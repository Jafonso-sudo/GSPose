import numpy as np
from PIL import Image
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
from posingpixels.utils.cotracker import get_tracks_outside_mask, sample_support_grid_points, unscale_by_crop
from posingpixels.utils.geometry import (
    do_axis_rotation,
    interpolate_poses,
)
from posingpixels.alignment import get_safe_query_points
from typing import List, Optional
from posingpixels.segmentation import get_bbox_from_mask, process_image_crop
from posingpixels.utils.cotracker import scale_by_crop

from posingpixels.utils.cotracker import get_ground_truths


class CoMeshTracker:
    def __init__(
        self,
        dataset: YCBinEOATDataset,
        visible_background: bool = False,
        crop: bool = True,
        offline: bool = True,
        offline_limit: int = 500,
        limit: Optional[int] = None,
        support_grid: Optional[int] = None,
        downcast: bool = False,
        axis_rotation_steps: int = 20,
        final_interpolation_steps: int = 20,
        better_initialization: bool = True,
        initialize_first_real_frame: bool = True,
        query_frames: Optional[list] = None,
        mask_threshold: float = 0.7,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
        if support_grid and self.better_initialization:
            raise ValueError("Support grid initialization not yet implemented")
        assert crop != (
            support_grid is not None
        ), "SUPPORT_GRID must be set if BLACK_BACKGROUND is False (and vice versa)"
        # Dataset Config
        self.dataset = dataset
        self.K = dataset.K
        self.H, self.W = dataset.image_size
        # Model Config
        self.visible_background = visible_background
        self.crop = crop
        self.offline = offline
        self.offline_limit = offline_limit
        self.canonical_poses = [dataset.get_canonical_pose()]
        self.axis_rotation_steps = axis_rotation_steps
        self.final_interpolation_steps = final_interpolation_steps
        self.interpolation_steps = (
            axis_rotation_steps * len(self.canonical_poses) + final_interpolation_steps
        )  # TODO: Needs changing once self.canonical_poses is more than one
        if query_frames is not None:
            self.query_frames = query_frames
        else:
            self.query_frames = [0, axis_rotation_steps // 2]
        self.limit = offline_limit if offline else limit or len(self)
        self.support_grid = support_grid
        self.mask_threshold = mask_threshold
        self.model_resolution = (384, 512)
        self.downcast = downcast
        self.device = device
        # Initialization Config
        self.start_pose = dataset.get_gt_pose(0)
        self.init_video_dir = os.path.join(self.dataset.video_dir, "init_video")
        self.cotracker_input_dir = os.path.join(self.dataset.video_dir, "input")
        self.better_initialization = better_initialization
        self.initialize_first_real_frame = initialize_first_real_frame

    def __call__(self):
        # Get canonical points
        self.get_canonical_points()
        # Create init video
        self.create_init_video()
        # Prepare img directory (input for CoTracker)
        self.prepare_img_directory()
        # Prepare query points
        self.get_query_points()
        # Run CoTracker
        return self.run_cotracker()

    def __len__(self):
        return len(self.dataset) + self.interpolation_steps

    def get_rgb(self, idx: int) -> np.ndarray:
        if idx < self.interpolation_steps:
            return cv2.cvtColor(
                cv2.imread(
                    os.path.join(self.init_video_dir, f"{idx:05d}.jpg"),
                    cv2.IMREAD_COLOR,
                ),
                cv2.COLOR_BGR2RGB,
            )
        return self.dataset.get_rgb(idx - self.interpolation_steps)

    def get_mask(self, idx: int) -> np.ndarray:
        if idx < self.interpolation_steps:
            return (
                cv2.imread(
                    os.path.join(self.init_video_dir, f"{idx:05d}.png"),
                    cv2.IMREAD_GRAYSCALE,
                )
                / 255
            )
        return self.dataset.get_mask(idx - self.interpolation_steps)

    def get_gt_poses(self) -> np.ndarray:
        return np.concatenate([self.interpolation_poses, self.dataset.get_gt_poses()])

    def get_gt_pose(self, idx: int) -> np.ndarray:
        if idx < self.interpolation_steps:
            return self.interpolation_poses[idx]
        return self.dataset.get_gt_pose(idx - self.interpolation_steps)

    def get_gt_depth(self, idx: int) -> np.ndarray:
        if idx < self.interpolation_steps:
            depth_image = Image.open(
                os.path.join(self.init_video_dir, f"{idx:05d}.tiff")
            )
            return np.array(depth_image).astype(np.float32)
        return self.dataset.get_cad_depth(idx - self.interpolation_steps)

    def get_canonical_points(self):
        all_canonical_points = []
        self.num_object_points = 0
        self.unposed_3d_points_lens = []
        for pose in self.canonical_poses:
            unposed_points, _ = get_safe_query_points(
                R=pose[:3, :3],
                T=pose[:3, 3],
                camK=self.K,
                H=self.H,
                W=self.W,
                mesh=self.dataset.get_mesh(),
                min_pixel_distance=10 if not self.interpolation_steps else 15,
                alpha_margin=5 if not self.interpolation_steps else 10,
                depth_margin=2 if not self.interpolation_steps else 6,
            )
            self.num_object_points += len(unposed_points)
            all_canonical_points.append(unposed_points)
            self.unposed_3d_points_lens.append(len(unposed_points))
        self.unposed_3d_points = np.concatenate(all_canonical_points, axis=0)

    def create_init_video(self):
        print("Creating init video")
        assert (
            len(self.canonical_poses) == 1
        ), "Not yet implemented for multiple query poses"
        assert self.start_pose is not None

        if not self.interpolation_steps:
            return None, None, None
        if not os.path.exists(self.init_video_dir):
            os.makedirs(self.init_video_dir)
        for f in os.listdir(self.init_video_dir):
            os.remove(os.path.join(self.init_video_dir, f))

        init_poses = []
        for pose in self.canonical_poses:
            # 1. Rotate the object around z-axis until it does a full circle
            z_axis_poses = np.eye(4)[np.newaxis].repeat(
                self.axis_rotation_steps, axis=0
            )
            z_axis_poses[:, :3, :3] = do_axis_rotation(
                pose[:3, :3], self.axis_rotation_steps, axis="y"
            )
            z_axis_poses[:, :, 3] = pose[:, 3]
            init_poses.append(z_axis_poses)
        # 2. Interpolate between the last canonical_pose and the start_pose
        final_transition = np.stack(
            interpolate_poses(
                self.canonical_poses[-1][:3, :3],
                self.canonical_poses[-1][:3, 3],
                self.start_pose[:3, :3],
                self.start_pose[:3, 3],
                self.final_interpolation_steps,
            ),
            axis=0,
        )
        init_poses.append(final_transition)
        self.interpolation_poses = np.concatenate(init_poses, axis=0)

        # Create init video itself and check visibility of the canonical points in each frame
        base_frame = self.dataset.get_rgb(0)
        base_frame_mask = self.dataset.get_mask(0) > self.mask_threshold
        base_frame[base_frame_mask] = 0
        self.init_coords = np.zeros(
            (self.interpolation_steps, self.num_object_points, 2)
        )
        self.init_vis = np.zeros((self.interpolation_steps, self.num_object_points))
        for i, P_i in enumerate(
            tqdm(self.interpolation_poses, desc="Creating init video")
        ):
            rgb, depth = self.dataset.render_mesh_at_pose(pose=P_i)
            depth_rgb = depth[:, :, None]
            rgb = base_frame * (depth_rgb <= 0) + rgb * (depth_rgb > 0)

            # Save RGB, Mask, and depth
            cv2.imwrite(
                os.path.join(self.init_video_dir, f"{i:05d}.jpg"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )
            cv2.imwrite(
                os.path.join(self.init_video_dir, f"{i:05d}.png"),
                (depth > 0).astype(np.uint8) * 255,
            )
            depth_image_32bit = Image.fromarray(depth.astype(np.float32))
            depth_image_32bit.save(os.path.join(self.init_video_dir, f"{i:05d}.tiff"))

            self.init_coords[i], self.init_vis[i] = get_ground_truths(
                P_i, self.K, self.unposed_3d_points, depth > 0, depth
            )

    def prepare_img_directory(self):
        # Clear directory
        if not os.path.exists(self.cotracker_input_dir):
            os.makedirs(self.cotracker_input_dir)
        for f in os.listdir(self.cotracker_input_dir):
            os.remove(os.path.join(self.cotracker_input_dir, f))
        # Prepare images for CoTracker
        self.bboxes, self.scaling = [], []
        for i in tqdm(range(self.limit), desc="Preparing images for CoTracker"):
            rgb = self.get_rgb(i)
            mask = self.get_mask(i) > self.mask_threshold
            if not self.visible_background:
                rgb[mask == 0, :] = 0
            if self.crop:
                bbox = get_bbox_from_mask(mask)
                assert bbox
                rgb, processed_bbox, scaling_factor = process_image_crop(
                    rgb,
                    bbox,
                    padding=10,
                    target_size=self.model_resolution,
                )
                self.bboxes.append(processed_bbox)
                self.scaling.append(scaling_factor)

            cv2.imwrite(
                os.path.join(self.cotracker_input_dir, f"{i:05d}.jpg"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )

        if self.crop:
            self.init_coords = scale_by_crop(
                    torch.tensor(self.init_coords),
                    torch.tensor(self.bboxes[:self.interpolation_steps]),
                    torch.tensor(self.scaling[:self.interpolation_steps]),
                ).detach().cpu().numpy()

    def get_query_points(self):
        all_query_2d_points = []
        query_to_point = []
        for frame in self.query_frames:
            points = self.init_coords[frame]
            visibility = self.init_vis[frame]

            # Get idx of visible points
            visible_idx = visibility > 0
            points = points[visible_idx]
            # Prepend frame idx
            query_points = np.concatenate(
                [np.ones((len(points), 1)) * frame, points], axis=1
            )
            all_query_2d_points.append(query_points)
            query_to_point.append(visible_idx)
        self.query_2d_points = np.concatenate(all_query_2d_points, axis=0)
        self.queries_sizes = [len(q) for q in all_query_2d_points]
        self.query_to_point = np.array(query_to_point)
        self.num_query_points = len(self.query_2d_points)
        tensor_query_to_point = torch.tensor(self.query_to_point)
        self.query_indexes = torch.nonzero(tensor_query_to_point)
        self.query_indexes = self.query_indexes[:, 1].flatten()

        # Add support grid
        if self.support_grid is not None:
            support_grid_points = sample_support_grid_points(
                self.H,
                self.W,
                self.interpolation_steps,
                self.get_mask(0),
                grid_size=self.support_grid,
            )
            self.query_2d_points = np.concatenate(
                [self.query_2d_points, support_grid_points], axis=0
            )

    def run_cotracker(self):
        # Prepare CoTracker initialization
        
        cotracker_init_coords = np.zeros((self.limit, self.num_query_points, 2))
        cotracker_init_coords[: self.interpolation_steps] = self.init_coords[:, self.query_indexes]
        cotracker_init_coords[self.interpolation_steps :, :] = self.init_coords[-1, self.query_indexes]
        cotracker_init_vis = np.zeros((self.limit, self.num_query_points))
        cotracker_init_vis[: self.interpolation_steps] = (self.init_vis[:, self.query_indexes] - 0.5) * 40
        cotracker_init_confidence = np.zeros((self.limit, self.num_query_points))
        cotracker_init_confidence[: self.interpolation_steps, :] = 20

        cotracker_init_coords = (
            torch.tensor(cotracker_init_coords).float().to(self.device)[None]
        )
        cotracker_init_vis = (
            torch.tensor(cotracker_init_vis).float().to(self.device)[None]
        )
        cotracker_init_confidence = (
            torch.tensor(cotracker_init_confidence).float().to(self.device)[None]
        )
        if not self.better_initialization:
            cotracker_init_coords = None
            cotracker_init_vis = None
            cotracker_init_confidence = None
            
        if not self.offline:
            tracks, visibility, confidence = get_online_cotracker_predictions(
                self.cotracker_input_dir,
                grid_size=0,
                queries=self.query_2d_points,
                downcast=self.downcast,
                init_coords=cotracker_init_coords,
                init_vis=cotracker_init_vis,
                init_confidence=cotracker_init_confidence,
                init_length=self.interpolation_steps
            )
        else:
            tracks, visibility, confidence = get_offline_cotracker_predictions(
                self.cotracker_input_dir,
                grid_size=0,
                queries=self.query_2d_points,
                limit=self.offline_limit,
                init_coords=cotracker_init_coords,
                init_vis=cotracker_init_vis,
                init_confidence=cotracker_init_confidence,
                init_length=self.interpolation_steps,
                downcast=self.downcast,
            )
            
        # Fix visibility for points outside the mask
        pred_tracks_original = unscale_by_crop(
            tracks[0],
            torch.tensor(self.bboxes[:self.limit]).to(self.device),
            torch.tensor(self.scaling[:self.limit]).to(self.device),
        )
        masks_np = np.stack([self.get_mask(i) for i in range(self.limit)], axis=0) > self.mask_threshold
        masks_torch = torch.tensor(masks_np).to(self.device)
        unmasked_indices = get_tracks_outside_mask(pred_tracks_original, masks_torch)
        visibility[:, unmasked_indices[:, 0], unmasked_indices[:, 1]] = 0
        
        return tracks, visibility, confidence
        
        


def get_offline_cotracker_predictions(
    video_img_directory,
    grid_size=10,
    queries: np.ndarray = None,
    downcast=False,
    limit=None,
    device=torch.device("cuda"),
    init_coords=None,
    init_vis=None,
    init_confidence=None,
    init_length=None,
):
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=downcast):
        queries_torch = (
            torch.tensor(queries).unsqueeze(0).to(device).float()
            if queries is not None
            else None
        )
        cotracker_offline = CoTrackerOnlinePredictor(offline=True, window_len=60).to(
            device
        )
        pred_tracks, pred_visibility, confidence = cotracker_offline(
            load_video_images(
                video_img_directory, limit=limit, device=device, dtype=torch.float32
            ),
            grid_size=grid_size,
            queries=queries_torch,
            is_first_step=True,
            init_coords=init_coords,
            init_vis=init_vis,
            init_confidence=init_confidence,
            init_length=init_length
        )  # B T N 2,  B T N 1, B T N 1
    return pred_tracks, pred_visibility, confidence


def get_online_cotracker_predictions(
    video_img_directory,
    grid_size=10,
    queries: np.ndarray = None,
    step=8,
    downcast=False,
    device=torch.device("cuda"),
    init_coords=None,
    init_vis=None,
    init_confidence=None,
    init_length=None,
):
    offline = step != 8
    if queries is not None and grid_size:
        raise ValueError(
            "Cannot provide queries and grid_size at the same time. Provide support_grid_size instead."
        )
    queries_torch = (
        torch.tensor(queries).unsqueeze(0).to(device).float()
        if queries is not None
        else None
    )
    batch_iterator = ImageBatchIterator(
        video_img_directory, batch_size=step * 2, overlap=step, device=device
    )
    is_first_step = True
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=downcast):
        cotracker_online = CoTrackerOnlinePredictor(
            window_len=step * 2, offline=offline
        ).to(device)
        for batch in tqdm(batch_iterator, desc="Processing batches"):
            pred_tracks, pred_visibility, confidence = cotracker_online(
                video_chunk=batch,
                is_first_step=is_first_step,
                grid_size=grid_size,
                queries=queries_torch,
                init_coords=init_coords,
                init_vis=init_vis,
                init_confidence=init_confidence,
                init_length=init_length,
            )
            is_first_step = False
    return pred_tracks, pred_visibility, confidence