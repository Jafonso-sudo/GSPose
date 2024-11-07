import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import torch
from cotracker.predictor import CoTrackerOnlinePredictor
from posingpixels.datasets import ImageBatchIterator, YCBinEOATDataset, load_video_images

import cv2
from posingpixels.utils.cotracker import sample_support_grid_points
from posingpixels.utils.geometry import interpolate_poses
from posingpixels.alignment import get_safe_query_points
from typing import Optional
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
        interpolation_steps: int = 15,
        better_initialization: bool = True,
        initialize_first_real_frame: bool = True,
        mask_threshold: float = 0.7,
        device: torch.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        ),
    ):
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
        self.interpolation_steps = interpolation_steps
        self.limit = offline_limit if offline else limit or len(self)
        self.support_grid = support_grid
        self.mask_threshold = mask_threshold
        self.model_resolution = (384, 512)
        self.downcast = downcast
        self.device = device
        # Initialization Config
        self.start_pose = dataset.get_gt_pose(0)
        self.base_pose = (
            dataset.get_canonical_pose()
            if self.interpolation_steps > 1
            else self.start_pose
        )
        self.query_poses = [self.base_pose]
        self.init_video_dir = os.path.join(self.dataset.video_dir, "init_video")
        self.cotracker_input_dir = os.path.join(self.dataset.video_dir, "input")
        self.better_initialization = better_initialization
        self.initialize_first_real_frame = initialize_first_real_frame

    def __call__(self):
        # Create init video
        self.create_init_video()
        # Prepare img directory (input for CoTracker)
        self.prepare_img_directory()
        # Prepare query points
        self.get_query_points()
        # Prepare CoTracker initialization
        self.prepare_cotracker_initialization()
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

    def get_gt_pose(self, idx: int) -> Optional[np.ndarray]:
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

    def create_init_video(self):
        assert (
            len(self.query_poses) == 1
        ), "Not yet implemented for multiple query poses"
        assert self.base_pose is not None and self.start_pose is not None

        if not self.interpolation_steps:
            return None, None, None
        if not os.path.exists(self.init_video_dir):
            os.makedirs(self.init_video_dir)
        for f in os.listdir(self.init_video_dir):
            os.remove(os.path.join(self.init_video_dir, f))

        self.interpolation_poses = interpolate_poses(
            self.base_pose[:3, :3],
            self.base_pose[:3, 3],
            self.start_pose[:3, :3],
            self.start_pose[:3, 3],
            self.interpolation_steps,
        )

        base_frame = self.dataset.get_rgb(0)
        base_frame_mask = self.dataset.get_mask(0) > self.mask_threshold
        base_frame[base_frame_mask] = 0
        for i, P_i in enumerate(self.interpolation_poses):
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

    def get_query_points(self):
        assert (
            len(self.query_poses) == 1
        ), "Not yet implemented for multiple query poses"
        assert self.base_pose is not None
        # Get query points
        self.unposed_3d_points, self.query_2d_points = get_safe_query_points(
            R=self.base_pose[:3, :3],
            T=self.base_pose[:3, 3],
            camK=self.K,
            H=self.H,
            W=self.W,
            mesh=self.dataset.get_mesh(),
            min_pixel_distance=10 if not self.interpolation_steps else 15,
            alpha_margin=5 if not self.interpolation_steps else 10,
            depth_margin=2 if not self.interpolation_steps else 6,
        )
        self.object_query_points_num = len(self.unposed_3d_points)
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
        # Prepare query points for CoTracker
        self.input_query_points = self.query_2d_points.copy()
        if self.crop:
            self.input_query_points[:, 1:] = np.array(
                scale_by_crop(
                    torch.tensor(self.query_2d_points[:, 1:]).unsqueeze(0),
                    torch.tensor(self.bboxes[:1]),
                    torch.tensor(self.scaling[:1]),
                )[0]
            )

    def prepare_cotracker_initialization(self):
        if not self.better_initialization:
            self.init_coords = None
            self.init_vis = None
            self.init_confidence = None
            return
        extra_frame = int(self.initialize_first_real_frame)
        init_coords = np.zeros((self.limit, self.object_query_points_num, 2))
        init_vis = np.zeros((self.limit, self.object_query_points_num))
        init_confidence = np.zeros((self.limit, self.object_query_points_num))
        init_confidence[: self.interpolation_steps + extra_frame, :] = 20
        for i, pose in enumerate(self.interpolation_poses):
            mask = self.get_mask(i)
            depth = self.get_gt_depth(i)
            init_coords[i], init_vis[i] = get_ground_truths(
                pose, self.K, self.unposed_3d_points, mask, depth
            )
            init_vis[i] = (
                init_vis[i] - 0.5
            ) * 40  # [0, 1] -> [-20, 20] Why? Since this is pre-sigmoid
        # Scale by crop
        if self.crop:
            torch_bbox = torch.tensor(self.bboxes).to(self.device)[: self.limit]
            torch_scaling = torch.tensor(self.scaling).to(self.device)[: self.limit]
            init_coords = scale_by_crop(
                torch.tensor(init_coords).to(self.device)[: self.limit],
                torch_bbox,
                torch_scaling,
            ).float()[None]
        else:
            init_coords = torch.tensor(init_coords).float().to(self.device)[None]
        init_coords[:, self.interpolation_steps :] = init_coords[
            :, self.interpolation_steps - 1 : self.interpolation_steps, :, :
        ].repeat(1, self.limit - self.interpolation_steps, 1, 1)
        if extra_frame:
            init_vis[self.interpolation_steps] = init_vis[self.interpolation_steps - 1]
        init_vis = torch.tensor(init_vis).float().to(self.device)[None]
        init_confidence = torch.tensor(init_confidence).float().to(self.device)[None]
        self.init_vis = init_vis
        self.init_confidence = init_confidence
        self.init_coords = init_coords
        if len(self.query_2d_points) != self.object_query_points_num:
            num_grid_points = len(self.query_2d_points) - self.object_query_points_num
            support_coords = (
                torch.tensor(self.query_2d_points[self.object_query_points_num :, 1:])[
                    None
                ]
                .repeat(self.limit, 1, 1)
                .float()
                .to(self.device)[None]
            )
            self.init_coords = torch.cat([self.init_coords, support_coords], dim=2)
            init_zeroes = (
                torch.zeros((1, self.limit, num_grid_points)).float().to(self.device)
            )
            self.init_vis = torch.cat([self.init_vis, init_zeroes], dim=2)
            self.init_confidence = torch.cat([self.init_confidence, init_zeroes], dim=2)

    def run_cotracker(self):
        if not self.offline:
            return get_online_cotracker_predictions(
                self.cotracker_input_dir,
                grid_size=0,
                queries=self.input_query_points,
                downcast=self.downcast,
            )
        else:
            return get_offline_cotracker_predictions(
                self.cotracker_input_dir,
                grid_size=0,
                queries=self.input_query_points,
                limit=self.offline_limit,
                init_coords=self.init_coords,
                init_vis=self.init_vis,
                # init_confidence=self.init_confidence,
                downcast=self.downcast,
            )


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
        )  # B T N 2,  B T N 1, B T N 1
    return pred_tracks, pred_visibility, confidence


def get_online_cotracker_predictions(
    video_img_directory,
    grid_size=10,
    queries: np.ndarray = None,
    step=8,
    downcast=False,
    device=torch.device("cuda"),
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
            first_img = batch[0][0]
            pred_tracks, pred_visibility, confidence = cotracker_online(
                video_chunk=batch,
                is_first_step=is_first_step,
                grid_size=grid_size,
                queries=queries_torch,
            )
            is_first_step = False
    return pred_tracks, pred_visibility, confidence
