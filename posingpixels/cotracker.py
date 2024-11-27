import numpy as np
from PIL import Image
from torch.cuda import is_available
from tqdm import tqdm
import os
import torch
from cotracker.predictor import CoTrackerOnlinePredictor
from posingpixels.datasets import (
    ImageBatchIterator,
    YCBinEOATDataset,
    load_video_images,
)

import cv2
from posingpixels.pnp import PnPSolver
from posingpixels.pointselector import SelectMostConfidentView
from posingpixels.query_refiner import QueryRefiner
from posingpixels.utils.cotracker import (
    get_tracks_outside_mask,
    sample_support_grid_points,
    unscale_by_crop,
)
from posingpixels.utils.geometry import (
    do_axis_rotation,
    interpolate_poses,
    rotation_matrix_y,
)
from posingpixels.alignment import CanonicalPointSampler, get_safe_query_points
from typing import Optional
from posingpixels.segmentation import get_bbox_from_mask, process_image_crop
from posingpixels.utils.cotracker import scale_by_crop

from posingpixels.utils.cotracker import get_ground_truths


class CoPoseTracker:
    cotracker_step = 8
    cotracker_window = 16
    cotracker_resolution = (384, 512)

    def __init__(
        self,
        canonical_point_sampler: CanonicalPointSampler,
        pnp_solver: Optional[PnPSolver] = None,
        downcast: bool = False,
        mask_threshold: float = 0.7,
        pose_interpolation_steps: int = 5,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        self.canonical_point_sampler = canonical_point_sampler
        self.pnp_solver = pnp_solver
        self.downcast = downcast
        self.mask_threshold = mask_threshold
        self.pose_interpolation_steps = pose_interpolation_steps
        self.device = device

    def get_canonical_points(
        self, dataset: YCBinEOATDataset, canonical_poses: np.ndarray
    ):
        canonical_points = []
        for pose in canonical_poses:
            rgb, depth, alpha = dataset.render_mesh_at_pose(pose)
            points_3d = self.canonical_point_sampler(rgb, alpha, depth, pose, dataset.K)
            canonical_points.append(points_3d)
        canonical_points = np.concatenate(canonical_points, axis=0)

        return canonical_points

    def _prepare_video_directory(self, video_dir: str):
        if not os.path.exists(video_dir):
            os.makedirs(video_dir)
        for f in os.listdir(video_dir):
            os.remove(os.path.join(video_dir, f))

    def create_prepend_video(
        self,
        dataset: YCBinEOATDataset,
        canonical_points: np.ndarray,
        query_poses: np.ndarray,
        start_pose: np.ndarray,
    ):
        init_video_dir = os.path.join(dataset.video_dir, "init_video")
        self._prepare_video_directory(init_video_dir)

        # Go through query poses and interpolate between them
        query_frames, prepend_poses = [], []
        for i, pose in enumerate(
            np.concatenate([query_poses[1:], [start_pose]], axis=0), 1
        ):
            query_frames.append(len(prepend_poses))
            prepend_poses.extend(
                interpolate_poses(
                    query_poses[i - 1, :3, :3],
                    query_poses[i - 1, :3, 3],
                    pose[:3, :3],
                    pose[:3, 3],
                    self.pose_interpolation_steps,
                )
            )
        # Ensure that there is at least `window` frames before the first real frame
        prepend_poses.extend([start_pose] * self.cotracker_window)
        prepend_poses = np.stack(prepend_poses, axis=0)
        prepend_length, num_points = len(prepend_poses), len(canonical_points)

        # Create video while checking for visibility of canonical points
        base_frame = dataset.get_rgb(0)
        base_frame[dataset.get_mask(0) > self.mask_threshold] = 0

        prepend_coords = np.zeros((prepend_length, num_points, 2))
        prepend_vis = np.zeros((prepend_length, num_points))

        for i, pose in enumerate(tqdm(prepend_poses, desc="Creating prepend video")):
            rgb, depth, _ = dataset.render_mesh_at_pose(pose)
            depth_rgb = depth[:, :, None]
            rgb = base_frame * (depth_rgb <= 0) + rgb * (depth_rgb > 0)
            # RGB
            cv2.imwrite(
                os.path.join(init_video_dir, f"{i:05d}.jpg"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )
            # Alpha/Mask
            cv2.imwrite(
                os.path.join(init_video_dir, f"{i:05d}.png"),
                (depth > 0).astype(np.uint8) * 255,
            )
            # Depth
            depth_image_32bit = Image.fromarray(depth.astype(np.float32))
            depth_image_32bit.save(os.path.join(init_video_dir, f"{i:05d}.tiff"))
            # Coodinates & Visibility
            prepend_coords[i], prepend_vis[i] = get_ground_truths(
                pose, dataset.K, canonical_points, depth > 0, depth
            )

        return prepend_coords, prepend_vis, prepend_poses, np.array(query_frames)

    def get_query_input(
        self,
        prepend_coords: np.ndarray,
        prepend_vis: np.ndarray,
        query_frames: np.ndarray,
    ):
        input_query = []
        query_to_point = []
        # TODO: Not only should they be visible, they should also be in the "safe zone" as defined for the canonical points
        # TODO: Check that we didn't choose too big of a threshold for visibility. Maybe we should make it very small in the get_ground_truths function
        for frame in query_frames:
            points = prepend_coords[frame]
            visibility = prepend_vis[frame]

            # Get idx of visible points
            visible_idx = visibility > 0
            points = points[visible_idx]
            # Prepend frame idx
            query_points = np.concatenate(
                [np.ones((len(points), 1)) * frame, points], axis=1
            )
            input_query.append(query_points)
            query_to_point.append(visible_idx)

        query_sizes = [len(q) for q in input_query]
        input_query = np.concatenate(input_query, axis=0)
        query_to_point = np.array(query_to_point)
        num_queries = len(input_query)
        # np.nonzero(query_to_point)
        query_to_point_indexes = np.array(np.nonzero(query_to_point)).T

        return input_query, np.array(query_sizes), num_queries, query_to_point_indexes


class CoTrackerInput:
    def __init__(
        self,
        dataset: YCBinEOATDataset,
        canonical_points: np.ndarray,
        prepend_poses: np.ndarray,
        prepend_coords: np.ndarray,
        prepend_vis: np.ndarray,
        input_query: np.ndarray,
        query_frames: np.ndarray,
        query_lengths: np.ndarray,
        query_to_point_indexes: np.ndarray,
        limit: Optional[int] = None,
    ) -> None:
        self.dataset = dataset
        self.limit = limit or len(dataset)
        self.canonical_points = canonical_points
        self.num_canonical_points = canonical_points.shape[0]
        self.prepend_dir = os.path.join(dataset.video_dir, "init_video")
        self.prepend_poses = prepend_poses
        self.prepend_coords = prepend_coords
        self.prepend_vis = prepend_vis
        self.prepend_length = prepend_poses.shape[0]
        self.video_dir = os.path.join(dataset.video_dir, "input")
        self.input_query = input_query
        self.query_frames = query_frames
        self.num_query_points = input_query.shape[0]
        self.query_lengths = query_lengths
        self.query_to_point_indexes = query_to_point_indexes
        self.bboxes: np.ndarray = None
        self.scaling: np.ndarray = None

    def __len__(self):
        return self.limit + self.prepend_length

    def get_rgb(self, idx: int) -> np.ndarray:
        if idx < self.prepend_length:
            return cv2.cvtColor(
                cv2.imread(
                    os.path.join(self.prepend_dir, f"{idx:05d}.jpg"),
                    cv2.IMREAD_COLOR,
                ),
                cv2.COLOR_BGR2RGB,
            )
        return self.dataset.get_rgb(idx - self.prepend_length)

    def get_mask(self, idx: int) -> np.ndarray:
        if idx < self.prepend_length:
            return (
                cv2.imread(
                    os.path.join(self.prepend_dir, f"{idx:05d}.png"),
                    cv2.IMREAD_GRAYSCALE,
                )
                / 255
            )
        return self.dataset.get_mask(idx - self.prepend_length)

    def get_gt_depth(self, idx: int) -> np.ndarray:
        if idx < self.prepend_length:
            depth_image = Image.open(os.path.join(self.prepend_dir, f"{idx:05d}.tiff"))
            return np.array(depth_image).astype(np.float32)
        return self.dataset.get_cad_depth(idx - self.prepend_length)

    @property
    def gt_poses(self) -> np.ndarray:
        return np.concatenate([self.prepend_poses, self.dataset.get_gt_poses()])

    def get_gt_pose(self, idx: int) -> np.ndarray:
        if idx < self.prepend_length:
            return self.prepend_poses[idx]
        return self.dataset.get_gt_pose(idx - self.prepend_length)

    def get_initialization(self, device: torch.device):
        flatten_query_to_point_indexes = self.query_to_point_indexes[:, 1].flatten()
        # Coords
        init_coords = np.zeros((self.prepend_length, self.num_query_points, 2))
        init_coords[...] = self.prepend_coords[
            :, flatten_query_to_point_indexes
        ]
        # Visibility
        init_vis = np.zeros((self.prepend_length, self.num_query_points))
        init_vis = (
            self.prepend_vis[:, flatten_query_to_point_indexes] - 0.5
        ) * 40
        # Confidence
        init_confidence = np.zeros((self.prepend_length, self.num_query_points))
        init_confidence[...] = 20

        return (
            torch.tensor(init_coords[None], device=device).float(),
            torch.tensor(init_vis[None], device=device).float(),
            torch.tensor(init_confidence[None], device=device).float(),
        )

    def get_queries(self, device: torch.device):
        return torch.tensor(self.input_query[None], device=device).float()


class CropCoPoseTracker(CoPoseTracker):
    def __call__(
        self,
        dataset: YCBinEOATDataset,
        canonical_poses: Optional[np.ndarray] = None,
        query_poses: Optional[np.ndarray] = None,
        start_pose: Optional[np.ndarray] = None,
        limit: Optional[int] = None,
        forced_coords: Optional[torch.Tensor] = None,
        forced_vis: Optional[torch.Tensor] = None,
        forced_conf: Optional[torch.Tensor] = None,
    ):
        if canonical_poses is None:
            canonical_poses = self.get_canonical_poses(dataset.get_canonical_pose())
        if query_poses is None:
            query_poses = canonical_poses
        else:
            query_poses = np.concatenate([canonical_poses, query_poses], axis=0)
        
            # for pose in canonical_poses:
            #     z_axis_poses = np.eye(4)[np.newaxis].repeat(
            #         3, axis=0
            #     )
            #     z_axis_poses[:, :3, :3] = do_axis_rotation(
            #         pose[:3, :3], 4, axis="y"
            #     )[1:]
            #     z_axis_poses[:, :, 3] = pose[:, 3]
            #     query_poses = np.concatenate([query_poses, z_axis_poses], axis=0)

        if start_pose is None:
            start_pose = dataset.get_gt_pose(0)
        # Get canonical points
        canonical_points = self.get_canonical_points(dataset, canonical_poses)
        # Get prepend video
        prepend_coords, prepend_vis, prepend_poses, query_frames = (
            self.create_prepend_video(
                dataset,
                canonical_points,
                query_poses,
                start_pose,
            )
        )
        # Prepare CoTracker input data
        # - Prepare query input
        input_query, query_sizes, num_queries, query_to_point_indexes = (
            self.get_query_input(prepend_coords, prepend_vis, query_frames)
        )
        # - Prepare video input
        cotracker_input = CoTrackerInput(
            dataset,
            canonical_points,
            prepend_poses,
            prepend_coords,
            prepend_vis,
            input_query,
            query_frames,
            query_sizes,
            query_to_point_indexes,
            limit,
        )

        # Prepare CoTracker input
        bboxes, scaling = self.create_video_input(cotracker_input)
        self.crop_input(cotracker_input, bboxes, scaling)

        # Run CoTracker
        query_refiner = self.prepare_query_refiner(cotracker_input, canonical_points)
        pred_coords, pred_vis, pred_conf = self.run_cotracker(
            cotracker_input, query_refiner, forced_coords, forced_vis, forced_conf
        )

        # Postprocess CoTracker output
        pred_coords, pred_vis, pred_conf, pred_coords_original = (
            self.postprocess_output(cotracker_input, pred_coords, pred_vis, pred_conf)
        )

        return pred_coords, pred_vis, pred_conf, pred_coords_original, cotracker_input

    def get_canonical_poses(self, canonical_pose: np.ndarray):
        canonical_poses = canonical_pose[np.newaxis]
        poses = []
        angles = [np.radians(angle) for angle in [90, 180, 270]]
        for angle in angles:
            pose = canonical_poses[0].copy()
            pose[:3, :3] = rotation_matrix_y(angle) @ pose[:3, :3]
            poses.append(pose)
        return np.concatenate([canonical_poses, poses], axis=0)

    def create_video_input(self, input: CoTrackerInput, limit: Optional[int] = None):
        self._prepare_video_directory(input.video_dir)
        bboxes, scaling = [], []
        for i in tqdm(range(len(input)), desc="Preparing images for CoTracker"):
            rgb = input.get_rgb(i)
            mask = input.get_mask(i) > self.mask_threshold
            rgb[mask == 0, :] = 0
            bbox = get_bbox_from_mask(mask)
            assert bbox
            rgb, processed_bbox, scaling_factor = process_image_crop(
                rgb,
                bbox,
                padding=10,
                target_size=self.cotracker_resolution,
            )
            bboxes.append(processed_bbox)
            scaling.append(scaling_factor)

            cv2.imwrite(
                os.path.join(input.video_dir, f"{i:05d}.jpg"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )

        return np.array(bboxes), np.array(scaling)

    def crop_input(
        self, input: CoTrackerInput, bboxes: np.ndarray, scaling: np.ndarray
    ):
        input.bboxes = bboxes
        input.scaling = scaling
        input.prepend_coords = (
            scale_by_crop(
                torch.tensor(input.prepend_coords),
                torch.tensor(input.bboxes[: input.prepend_length]),
                torch.tensor(input.scaling[: input.prepend_length]),
            )
            .detach()
            .cpu()
            .numpy()
        )

        frame_idx = np.repeat(input.query_frames, input.query_lengths)

        input.input_query[:, 1:] = input.prepend_coords[
            frame_idx, input.query_to_point_indexes[:, 1]
        ]

    def prepare_query_refiner(
        self, input: CoTrackerInput, canonical_points: np.ndarray
    ):
        if self.pnp_solver is None:
            return None
        point_selector_strategy = SelectMostConfidentView(
            input.num_canonical_points,
            torch.tensor(input.query_to_point_indexes, device=self.device),
            torch.tensor(input.query_lengths, device=self.device),
        )
        gt_poses = torch.tensor(input.prepend_poses, device=self.device)
        self.pnp_solver.K = torch.tensor(
            input.dataset.K[np.newaxis, :], device=self.device
        ).float()
        self.pnp_solver.X = torch.tensor(canonical_points, device=self.device).float()
        query_refiner = QueryRefiner(
            point_selector_strategy,
            self.pnp_solver,
            torch.tensor(input.bboxes, device=self.device),
            torch.tensor(input.scaling, device=self.device),
            gt_poses[:, :3, :3],
            gt_poses[:, :3, 3],
        )

        return query_refiner

    def run_cotracker(
        self,
        input: CoTrackerInput,
        query_refiner: QueryRefiner,
        forced_coords: Optional[torch.Tensor] = None,
        forced_vis: Optional[torch.Tensor] = None,
        forced_conf: Optional[torch.Tensor] = None,
    ):
        init_coords, init_vis, init_confidence = input.get_initialization(self.device)
        if (
            forced_coords is not None
            and forced_vis is not None
            and forced_conf is not None
        ):
            # forced_length = forced_coords.shape[1]
            # forced_point_num = forced_coords.shape[2]
            # init_coords[
            #     :,
            #     input.prepend_length : input.prepend_length + forced_length,
            #     :forced_point_num,
            # ] = forced_coords
            # init_vis[
            #     :,
            #     input.prepend_length : input.prepend_length + forced_length,
            #     :forced_point_num,
            # ] = forced_vis
            # init_confidence[
            #     :,
            #     input.prepend_length : input.prepend_length + forced_length,
            #     :forced_point_num,
            # ] = forced_conf
            init_coords = torch.cat((init_coords, forced_coords), dim=1)
            init_vis = torch.cat((init_vis, forced_vis), dim=1)
            init_confidence = torch.cat((init_confidence, forced_conf), dim=1)
        input_query = input.get_queries(self.device)

        batch_iterator = ImageBatchIterator(
            input.video_dir,
            batch_size=self.cotracker_step * 2,
            overlap=self.cotracker_step,
            device=self.device,
        )
        is_first_step = True
        with torch.autocast(
            device_type=self.device.type, dtype=torch.float16, enabled=self.downcast
        ):
            cotracker_online = CoTrackerOnlinePredictor(
                window_len=self.cotracker_step * 2, offline=False
            ).to(self.device)
            for batch in tqdm(batch_iterator, desc="Processing batches"):
                pred_coords, pred_vis, pred_conf = cotracker_online(
                    video_chunk=batch,
                    is_first_step=is_first_step,
                    grid_size=0,
                    queries=input_query,
                    init_coords=init_coords,
                    init_vis=init_vis,
                    init_confidence=init_confidence,
                    init_length=input.prepend_length,
                    query_refiner=query_refiner,
                )
                is_first_step = False

            del cotracker_online  # Free up memory

        return pred_coords, pred_vis, pred_conf

    def postprocess_output(
        self,
        input: CoTrackerInput,
        pred_coords: torch.Tensor,
        pred_vis: torch.Tensor,
        pred_conf: torch.Tensor,
    ):
        # Fix visibility for points outside the mask
        pred_coords_original = unscale_by_crop(
            pred_coords[0],
            torch.tensor(input.bboxes[: len(input)]).to(self.device),
            torch.tensor(input.scaling[: len(input)]).to(self.device),
        )
        masks_np = (
            np.stack([input.get_mask(i) for i in range(len(input))], axis=0)
            > self.mask_threshold
        )
        masks_torch = torch.tensor(masks_np).to(self.device)
        unmasked_indices = get_tracks_outside_mask(pred_coords_original, masks_torch)
        pred_vis[:, unmasked_indices[:, 0], unmasked_indices[:, 1]] = 0

        return pred_coords, pred_vis, pred_conf, pred_coords_original.unsqueeze(0)


# class CoMeshTracker:
#     def __init__(
#         self,
#         dataset: YCBinEOATDataset,
#         pnp_solver: PnPSolver,
#         visible_background: bool = False,
#         crop: bool = True,
#         offline: bool = True,
#         offline_limit: int = 500,
#         limit: Optional[int] = None,
#         support_grid: Optional[int] = None,
#         downcast: bool = False,
#         axis_rotation_steps: int = 20,
#         final_interpolation_steps: int = 20,
#         better_initialization: bool = True,
#         initialize_first_real_frame: bool = True,
#         mask_threshold: float = 0.7,
#         device: torch.device = torch.device(
#             "cuda:0" if torch.cuda.is_available() else "cpu"
#         ),
#     ):
#         if support_grid and self.better_initialization:
#             raise ValueError("Support grid initialization not yet implemented")
#         assert crop != (
#             support_grid is not None
#         ), "SUPPORT_GRID must be set if BLACK_BACKGROUND is False (and vice versa)"
#         # Dataset Config
#         self.dataset = dataset
#         self.K = dataset.K
#         self.H, self.W = dataset.image_size
#         # Model Config
#         self.pnp_solver = pnp_solver
#         self.visible_background = visible_background
#         self.crop = crop
#         self.offline = offline
#         self.offline_limit = offline_limit
#         self.canonical_poses = [dataset.get_canonical_pose()]
#         self.axis_rotation_steps = axis_rotation_steps
#         self.final_interpolation_steps = final_interpolation_steps
#         self.interpolation_steps = (
#             axis_rotation_steps * len(self.canonical_poses) + final_interpolation_steps
#         )  # TODO: Needs changing once self.canonical_poses is more than one
#         if query_frames is not None:
#             self.query_frames = query_frames
#         else:
#             self.query_frames = [0, axis_rotation_steps // 2]
#         self.limit = offline_limit if offline else limit or len(self)
#         self.support_grid = support_grid
#         self.mask_threshold = mask_threshold
#         self.model_resolution = (384, 512)
#         self.downcast = downcast
#         self.device = device
#         # Initialization Config
#         self.start_pose = dataset.get_gt_pose(0)
#         self.init_video_dir = os.path.join(self.dataset.video_dir, "init_video")
#         self.cotracker_input_dir = os.path.join(self.dataset.video_dir, "input")
#         self.better_initialization = better_initialization
#         self.initialize_first_real_frame = initialize_first_real_frame

#     def __call__(self):
#         # Get canonical points
#         self.get_canonical_points()
#         # Create init video
#         self.create_init_video()
#         # Prepare img directory (input for CoTracker)
#         self.prepare_img_directory()
#         # Prepare query points
#         self.get_query_points()
#         # Run CoTracker
#         return self.run_cotracker()

#     def __len__(self):
#         return len(self.dataset) + self.interpolation_steps

#     def get_rgb(self, idx: int) -> np.ndarray:
#         if idx < self.interpolation_steps:
#             return cv2.cvtColor(
#                 cv2.imread(
#                     os.path.join(self.init_video_dir, f"{idx:05d}.jpg"),
#                     cv2.IMREAD_COLOR,
#                 ),
#                 cv2.COLOR_BGR2RGB,
#             )
#         return self.dataset.get_rgb(idx - self.interpolation_steps)

#     def get_mask(self, idx: int) -> np.ndarray:
#         if idx < self.interpolation_steps:
#             return (
#                 cv2.imread(
#                     os.path.join(self.init_video_dir, f"{idx:05d}.png"),
#                     cv2.IMREAD_GRAYSCALE,
#                 )
#                 / 255
#             )
#         return self.dataset.get_mask(idx - self.interpolation_steps)

#     def get_gt_poses(self) -> np.ndarray:
#         return np.concatenate([self.interpolation_poses, self.dataset.get_gt_poses()])

#     def get_gt_pose(self, idx: int) -> np.ndarray:
#         if idx < self.interpolation_steps:
#             return self.interpolation_poses[idx]
#         return self.dataset.get_gt_pose(idx - self.interpolation_steps)

#     def get_gt_depth(self, idx: int) -> np.ndarray:
#         if idx < self.interpolation_steps:
#             depth_image = Image.open(
#                 os.path.join(self.init_video_dir, f"{idx:05d}.tiff")
#             )
#             return np.array(depth_image).astype(np.float32)
#         return self.dataset.get_cad_depth(idx - self.interpolation_steps)

#     def get_canonical_points(self):
#         all_canonical_points = []
#         self.num_object_points = 0
#         self.unposed_3d_points_lens = []
#         for pose in self.canonical_poses:
#             unposed_points, _ = get_safe_query_points(
#                 R=pose[:3, :3],
#                 T=pose[:3, 3],
#                 camK=self.K,
#                 H=self.H,
#                 W=self.W,
#                 mesh=self.dataset.get_mesh(),
#                 min_pixel_distance=10 if not self.interpolation_steps else 15,
#                 alpha_margin=5 if not self.interpolation_steps else 10,
#                 depth_margin=2 if not self.interpolation_steps else 6,
#             )
#             self.num_object_points += len(unposed_points)
#             all_canonical_points.append(unposed_points)
#             self.unposed_3d_points_lens.append(len(unposed_points))
#         self.unposed_3d_points = np.concatenate(all_canonical_points, axis=0)

#     def create_init_video(self):
#         print("Creating init video")
#         assert (
#             len(self.canonical_poses) == 1
#         ), "Not yet implemented for multiple query poses"
#         assert self.start_pose is not None

#         if not self.interpolation_steps:
#             return None, None, None
#         if not os.path.exists(self.init_video_dir):
#             os.makedirs(self.init_video_dir)
#         for f in os.listdir(self.init_video_dir):
#             os.remove(os.path.join(self.init_video_dir, f))

#         init_poses = []
#         for pose in self.canonical_poses:
#             # 1. Rotate the object around z-axis until it does a full circle
#             z_axis_poses = np.eye(4)[np.newaxis].repeat(
#                 self.axis_rotation_steps, axis=0
#             )
#             z_axis_poses[:, :3, :3] = do_axis_rotation(
#                 pose[:3, :3], self.axis_rotation_steps, axis="y"
#             )
#             z_axis_poses[:, :, 3] = pose[:, 3]
#             init_poses.append(z_axis_poses)
#         # 2. Interpolate between the last canonical_pose and the start_pose
#         final_transition = np.stack(
#             interpolate_poses(
#                 self.canonical_poses[-1][:3, :3],
#                 self.canonical_poses[-1][:3, 3],
#                 self.start_pose[:3, :3],
#                 self.start_pose[:3, 3],
#                 self.final_interpolation_steps,
#             ),
#             axis=0,
#         )
#         init_poses.append(final_transition)
#         self.interpolation_poses = np.concatenate(init_poses, axis=0)

#         # Create init video itself and check visibility of the canonical points in each frame
#         base_frame = self.dataset.get_rgb(0)
#         base_frame_mask = self.dataset.get_mask(0) > self.mask_threshold
#         base_frame[base_frame_mask] = 0
#         self.init_coords = np.zeros(
#             (self.interpolation_steps, self.num_object_points, 2)
#         )
#         self.init_vis = np.zeros((self.interpolation_steps, self.num_object_points))
#         for i, P_i in enumerate(
#             tqdm(self.interpolation_poses, desc="Creating init video")
#         ):
#             rgb, depth, _ = self.dataset.render_mesh_at_pose(pose=P_i)
#             depth_rgb = depth[:, :, None]
#             rgb = base_frame * (depth_rgb <= 0) + rgb * (depth_rgb > 0)

#             # Save RGB, Mask, and depth
#             cv2.imwrite(
#                 os.path.join(self.init_video_dir, f"{i:05d}.jpg"),
#                 cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
#             )
#             cv2.imwrite(
#                 os.path.join(self.init_video_dir, f"{i:05d}.png"),
#                 (depth > 0).astype(np.uint8) * 255,
#             )
#             depth_image_32bit = Image.fromarray(depth.astype(np.float32))
#             depth_image_32bit.save(os.path.join(self.init_video_dir, f"{i:05d}.tiff"))

#             self.init_coords[i], self.init_vis[i] = get_ground_truths(
#                 P_i, self.K, self.unposed_3d_points, depth > 0, depth
#             )

#     def prepare_img_directory(self):
#         # Clear directory
#         if not os.path.exists(self.cotracker_input_dir):
#             os.makedirs(self.cotracker_input_dir)
#         for f in os.listdir(self.cotracker_input_dir):
#             os.remove(os.path.join(self.cotracker_input_dir, f))
#         # Prepare images for CoTracker
#         self.bboxes, self.scaling = [], []
#         for i in tqdm(range(self.limit), desc="Preparing images for CoTracker"):
#             rgb = self.get_rgb(i)
#             mask = self.get_mask(i) > self.mask_threshold
#             if not self.visible_background:
#                 rgb[mask == 0, :] = 0
#             if self.crop:
#                 bbox = get_bbox_from_mask(mask)
#                 assert bbox
#                 rgb, processed_bbox, scaling_factor = process_image_crop(
#                     rgb,
#                     bbox,
#                     padding=10,
#                     target_size=self.model_resolution,
#                 )
#                 self.bboxes.append(processed_bbox)
#                 self.scaling.append(scaling_factor)

#             cv2.imwrite(
#                 os.path.join(self.cotracker_input_dir, f"{i:05d}.jpg"),
#                 cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
#             )

#         if self.crop:
#             self.init_coords = (
#                 scale_by_crop(
#                     torch.tensor(self.init_coords),
#                     torch.tensor(self.bboxes[: self.interpolation_steps]),
#                     torch.tensor(self.scaling[: self.interpolation_steps]),
#                 )
#                 .detach()
#                 .cpu()
#                 .numpy()
#             )

#     def get_query_points(self):
#         all_query_2d_points = []
#         query_to_point = []
#         for frame in self.query_frames:
#             points = self.init_coords[frame]
#             visibility = self.init_vis[frame]

#             # Get idx of visible points
#             visible_idx = visibility > 0
#             points = points[visible_idx]
#             # Prepend frame idx
#             query_points = np.concatenate(
#                 [np.ones((len(points), 1)) * frame, points], axis=1
#             )
#             all_query_2d_points.append(query_points)
#             query_to_point.append(visible_idx)
#         self.query_2d_points = np.concatenate(all_query_2d_points, axis=0)
#         self.queries_sizes = [len(q) for q in all_query_2d_points]
#         self.query_to_point = np.array(query_to_point)
#         self.num_query_points = len(self.query_2d_points)
#         tensor_query_to_point = torch.tensor(self.query_to_point)
#         self.query_indexes = torch.nonzero(tensor_query_to_point)
#         self.query_indexes = self.query_indexes[:, 1].flatten()

#         # Add support grid
#         if self.support_grid is not None:
#             support_grid_points = sample_support_grid_points(
#                 self.H,
#                 self.W,
#                 self.interpolation_steps,
#                 self.get_mask(0),
#                 grid_size=self.support_grid,
#             )
#             self.query_2d_points = np.concatenate(
#                 [self.query_2d_points, support_grid_points], axis=0
#             )

#     def run_cotracker(self):
#         # Prepare CoTracker initialization

#         cotracker_init_coords = np.zeros((self.limit, self.num_query_points, 2))
#         cotracker_init_coords[: self.interpolation_steps] = self.init_coords[
#             :, self.query_indexes
#         ]
#         cotracker_init_coords[self.interpolation_steps :, :] = self.init_coords[
#             -1, self.query_indexes
#         ]
#         cotracker_init_vis = np.zeros((self.limit, self.num_query_points))
#         cotracker_init_vis[: self.interpolation_steps] = (
#             self.init_vis[:, self.query_indexes] - 0.5
#         ) * 40
#         cotracker_init_confidence = np.zeros((self.limit, self.num_query_points))
#         cotracker_init_confidence[: self.interpolation_steps, :] = 20

#         cotracker_init_coords = (
#             torch.tensor(cotracker_init_coords).float().to(self.device)[None]
#         )
#         cotracker_init_vis = (
#             torch.tensor(cotracker_init_vis).float().to(self.device)[None]
#         )
#         cotracker_init_confidence = (
#             torch.tensor(cotracker_init_confidence).float().to(self.device)[None]
#         )
#         if not self.better_initialization:
#             cotracker_init_coords = None
#             cotracker_init_vis = None
#             cotracker_init_confidence = None

#         if self.pnp_solver is not None:
#             tensor_query_to_point = torch.tensor(
#                 self.query_to_point, device=self.device
#             )
#             true_indexes = torch.nonzero(tensor_query_to_point)
#             query_lengths = torch.tensor(self.queries_sizes, device=self.device)
#             self.point_selector = SelectMostConfidentView(
#                 self.num_object_points, true_indexes, query_lengths
#             )
#             gt_poses = torch.tensor(self.interpolation_poses, device=self.device)
#             self.pnp_solver.K = torch.tensor(
#                 self.K[np.newaxis, :], device=self.device
#             ).float()
#             self.pnp_solver.X = torch.tensor(
#                 self.unposed_3d_points, device=self.device
#             ).float()
#             self.query_refiner = QueryRefiner(
#                 self.point_selector,
#                 self.pnp_solver,
#                 torch.tensor(self.bboxes, device=self.device),
#                 torch.tensor(self.scaling, device=self.device),
#                 gt_poses[:, :3, :3],
#                 gt_poses[:, :3, 3],
#             )
#         else:
#             self.query_refiner = None

#         if not self.offline:
#             tracks, visibility, confidence = get_online_cotracker_predictions(
#                 self.cotracker_input_dir,
#                 grid_size=0,
#                 queries=self.query_2d_points,
#                 downcast=self.downcast,
#                 query_refiner=self.query_refiner,
#                 init_coords=cotracker_init_coords,
#                 init_vis=cotracker_init_vis,
#                 init_confidence=cotracker_init_confidence,
#                 init_length=self.interpolation_steps,
#             )
#         else:
#             tracks, visibility, confidence = get_offline_cotracker_predictions(
#                 self.cotracker_input_dir,
#                 grid_size=0,
#                 queries=self.query_2d_points,
#                 limit=self.offline_limit,
#                 init_coords=cotracker_init_coords,
#                 init_vis=cotracker_init_vis,
#                 init_confidence=cotracker_init_confidence,
#                 init_length=self.interpolation_steps,
#                 downcast=self.downcast,
#             )

#         # Fix visibility for points outside the mask
#         pred_tracks_original = unscale_by_crop(
#             tracks[0],
#             torch.tensor(self.bboxes[: self.limit]).to(self.device),
#             torch.tensor(self.scaling[: self.limit]).to(self.device),
#         )
#         masks_np = (
#             np.stack([self.get_mask(i) for i in range(self.limit)], axis=0)
#             > self.mask_threshold
#         )
#         masks_torch = torch.tensor(masks_np).to(self.device)
#         unmasked_indices = get_tracks_outside_mask(pred_tracks_original, masks_torch)
#         visibility[:, unmasked_indices[:, 0], unmasked_indices[:, 1]] = 0

#         return tracks, visibility, confidence


# def get_offline_cotracker_predictions(
#     video_img_directory,
#     grid_size=10,
#     queries: np.ndarray = None,
#     downcast=False,
#     limit=None,
#     device=torch.device("cuda"),
#     init_coords=None,
#     init_vis=None,
#     init_confidence=None,
#     init_length=None,
# ):
#     with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=downcast):
#         queries_torch = (
#             torch.tensor(queries).unsqueeze(0).to(device).float()
#             if queries is not None
#             else None
#         )
#         cotracker_offline = CoTrackerOnlinePredictor(offline=True, window_len=60).to(
#             device
#         )
#         pred_tracks, pred_visibility, confidence = cotracker_offline(
#             load_video_images(
#                 video_img_directory, limit=limit, device=device, dtype=torch.float32
#             ),
#             grid_size=grid_size,
#             queries=queries_torch,
#             is_first_step=True,
#             init_coords=init_coords,
#             init_vis=init_vis,
#             init_confidence=init_confidence,
#             init_length=init_length,
#         )  # B T N 2,  B T N 1, B T N 1
#     return pred_tracks, pred_visibility, confidence


# def get_online_cotracker_predictions(
#     video_img_directory,
#     grid_size=10,
#     queries: np.ndarray = None,
#     step=8,
#     downcast=False,
#     device=torch.device("cuda"),
#     query_refiner=None,
#     init_coords=None,
#     init_vis=None,
#     init_confidence=None,
#     init_length=None,
# ):
#     offline = step != 8
#     if queries is not None and grid_size:
#         raise ValueError(
#             "Cannot provide queries and grid_size at the same time. Provide support_grid_size instead."
#         )
#     queries_torch = (
#         torch.tensor(queries).unsqueeze(0).to(device).float()
#         if queries is not None
#         else None
#     )
#     batch_iterator = ImageBatchIterator(
#         video_img_directory, batch_size=step * 2, overlap=step, device=device
#     )
#     is_first_step = True
#     with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=downcast):
#         cotracker_online = CoTrackerOnlinePredictor(
#             window_len=step * 2, offline=offline
#         ).to(device)
#         for batch in tqdm(batch_iterator, desc="Processing batches"):
#             pred_tracks, pred_visibility, confidence = cotracker_online(
#                 video_chunk=batch,
#                 is_first_step=is_first_step,
#                 grid_size=grid_size,
#                 queries=queries_torch,
#                 init_coords=init_coords,
#                 init_vis=init_vis,
#                 init_confidence=init_confidence,
#                 init_length=init_length,
#                 query_refiner=query_refiner,
#             )
#             is_first_step = False
#     return pred_tracks, pred_visibility, confidence
