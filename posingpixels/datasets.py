from abc import ABC, abstractmethod
from enum import Enum
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import torch
from typing import Iterator, Optional, Tuple
import cv2

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import glob

from tqdm import tqdm
from misc_utils.gs_utils import egocentric_to_allocentric
from posingpixels.utils.gs_pose import (
    create_or_load_gaussian_splat_from_ycbineoat,
    load_model_net,
    render_gaussian_model_with_info,
)
from posingpixels.utils.offscreen_renderer import ModelRendererOffscreen
import trimesh
from posingpixels.utils.meshes import (
    get_bbox_from_size,
    get_diameter_from_mesh,
    get_size_from_mesh,
)
from posingpixels.segmentation import segment

proj_root = os.path.dirname(os.getcwd())


class ModelType(Enum):
    CAD = "CAD"
    GAUSSIAN = "Gaussian Splat"


class RenderableModel(ABC):
    def __init__(self, obj_dir: str, K: np.ndarray, H: int, W: int):
        self.obj_dir = obj_dir
        self.K = K
        self.H = H
        self.W = W

    @abstractmethod
    def render(self, pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Renders the given pose and returns the corresponding images.

        Args:
            pose (np.ndarray): The pose to be rendered.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing three numpy arrays representing the RGB, alpha, depth images.
        """
        pass


class CADModel(RenderableModel):
    def __init__(self, obj_dir: str, K: np.ndarray, H: int, W: int):
        super().__init__(obj_dir, K, H, W)
        self.renderer = ModelRendererOffscreen(K, H, W)
        self.mesh = trimesh.load_mesh(os.path.join(obj_dir, "textured_simple.obj"))
        self.obj_diameter = get_diameter_from_mesh(self.mesh)
        self.obj_size = get_size_from_mesh(self.mesh)
        self.bbox = get_bbox_from_size(self.obj_size)

    def render(self, pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rgb, depth = self.renderer.render(pose, self.mesh)
        alpha = (depth > 0).astype(float)
        return rgb, depth, alpha


class GaussianSplatModel(RenderableModel):
    def __init__(
        self,
        obj_dir: str,
        K: np.ndarray,
        H: int,
        W: int,
        dataset: "YCBinEOATDataset",
        device: torch.device,
    ):
        super().__init__(obj_dir, K, H, W)
        model_net = load_model_net(
            os.path.join(proj_root, "checkpoints/model_weights.pth")
        )
        self.ref_database = create_or_load_gaussian_splat_from_ycbineoat(
            dataset, model_net, device=device
        )
        self.gaussian_object = self.ref_database["obj_gaussians"]

    def render(self, pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        render = render_gaussian_model_with_info(
            self.gaussian_object, self.K, self.H, self.W, R=pose[:3, :3], T=pose[:3, 3]
        )

        rgb = render["image"]
        depth = render["depth"][0].detach().cpu().numpy()
        alpha = render["alpha"][0].detach().cpu().numpy()

        return rgb, depth, alpha


class CADModelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        object_dir: str,
        K: np.ndarray,
        H: int,
        W: int,
    ):
        self.K = K
        self.H = H
        self.W = W

        self.object_dir = object_dir
        self.obj_path = os.path.join(self.object_dir, "textured_simple.obj")
        mesh = trimesh.load_mesh(self.obj_path)

        self.obj_diameter = get_diameter_from_mesh(mesh)
        self.obj_size = get_size_from_mesh(mesh)
        self.bbox = get_bbox_from_size(self.obj_size)

        self.model = CADModel(self.object_dir, self.K, self.H, self.W)

        self.cad_rgb_dir = os.path.join(self.object_dir, "cad_rgb")
        self.cad_depth_dir = os.path.join(self.object_dir, "cad_depth")
        self.cad_mask_dir = os.path.join(self.object_dir, "cad_mask")

        self.poses = self._generate_dataset(200)
        self.mask_files = sorted(glob.glob(f"{self.cad_mask_dir}/*.png"))
        self.rgb_video_files = sorted(glob.glob(f"{self.cad_rgb_dir}/*.png"))
        self.cad_rgb_files = sorted(glob.glob(f"{self.cad_rgb_dir}/*.png"))

        # GSPose support
        self.obj_bbox3d = self.bbox.detach().cpu().numpy()
        self.diameter = self.obj_diameter
        self.bbox3d_diameter = torch.norm(self.obj_size).detach().cpu().numpy()
        self.bbox_diameter = self.bbox3d_diameter
        self.use_binarized_mask = False

    @staticmethod
    def random_rotation_matrix():
        # Step 1: Generate random quaternion
        u1, u2, u3 = np.random.rand(3)
        q_w = np.sqrt(1 - u1) * np.sin(2 * np.pi * u2)
        q_x = np.sqrt(1 - u1) * np.cos(2 * np.pi * u2)
        q_y = np.sqrt(u1) * np.sin(2 * np.pi * u3)
        q_z = np.sqrt(u1) * np.cos(2 * np.pi * u3)
        
        # Step 2: Convert quaternion to rotation matrix
        R = np.array([
            [1 - 2*(q_y**2 + q_z**2), 2*(q_x*q_y - q_w*q_z), 2*(q_x*q_z + q_w*q_y)],
            [2*(q_x*q_y + q_w*q_z), 1 - 2*(q_x**2 + q_z**2), 2*(q_y*q_z - q_w*q_x)],
            [2*(q_x*q_z - q_w*q_y), 2*(q_y*q_z + q_w*q_x), 1 - 2*(q_x**2 + q_y**2)]
        ])
        return R

    def _generate_dataset(self, num_samples: int):
        if (
            os.path.exists(self.cad_rgb_dir)
            and os.path.exists(self.cad_depth_dir)
            and len(os.listdir(self.cad_rgb_dir))
            == num_samples
            == len(os.listdir(self.cad_depth_dir))
        ):
            return np.load(os.path.join(self.object_dir, "poses.npy"))
        
        if not os.path.exists(self.cad_rgb_dir):
            os.makedirs(self.cad_rgb_dir)
        if not os.path.exists(self.cad_depth_dir):
            os.makedirs(self.cad_depth_dir)
        if not os.path.exists(self.cad_mask_dir):
            os.makedirs(self.cad_mask_dir)

        T = np.array([0, 0, self.obj_diameter * 1.3])
        poses = []
        for i in tqdm(range(num_samples)):
            R = self.random_rotation_matrix()
            pose = np.eye(4)
            pose[:3, :3] = R
            pose[:3, 3] = T
            poses.append(pose)

            rgb, depth, alpha = self.render(pose)

            cv2.imwrite(
                os.path.join(self.cad_rgb_dir, f"{i:05d}.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )

            depth_image_32bit = Image.fromarray(depth.astype(np.float32))
            depth_image_32bit.save(os.path.join(self.cad_depth_dir, f"{i:05d}.tiff"))

            cv2.imwrite(
                os.path.join(self.cad_mask_dir, f"{i:05d}.png"),
                (alpha * 255).astype(np.uint8)
            )

        poses = np.array(poses)
        np.save(os.path.join(self.object_dir, "poses.npy"), poses)

        return poses
    
    def render(self, pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.model.render(pose)
    
    def __len__(self):
        return len(self.poses)
    
    def get_gt_pose(self, idx: int) -> np.ndarray:
        return self.poses[idx]
    
    def get_rgb(self, idx: int) -> np.ndarray:
        return cv2.cvtColor(
            cv2.imread(os.path.join(self.cad_rgb_dir, f"{idx:05d}.png"), cv2.IMREAD_COLOR),
            cv2.COLOR_BGR2RGB
        )
    
    def get_depth(self, idx: int) -> np.ndarray:
        return cv2.imread(os.path.join(self.cad_depth_dir, f"{idx:05d}.tiff"), cv2.IMREAD_UNCHANGED) / 1e3
    
    def get_mask(self, idx: int) -> np.ndarray:
        return self.get_depth(idx) > 0
    

    
    
    # For GSPose support
    def __getitem__(self, idx):
        data_dict = dict()
        data_dict["camK"] = torch.tensor(self.K, dtype=torch.float32)
        data_dict["pose"] = torch.tensor(self.get_gt_pose(idx), dtype=torch.float32)
        data_dict["image"] = torch.tensor(self.get_rgb(idx), dtype=torch.float32)
        data_dict["allo_pose"] = torch.tensor(
            egocentric_to_allocentric(self.get_gt_pose(idx)), dtype=torch.float32
        )

        data_dict["image_ID"] = idx
        data_dict["image_path"] = self.rgb_video_files[idx]
        data_dict["mask"] = torch.tensor(self.get_mask(idx), dtype=torch.float32)
        data_dict["mask_path"] = self.mask_files[idx]
        data_dict["gt_mask_path"] = self.mask_files[idx]
        data_dict["coseg_mask_path"] = self.mask_files[idx]

        mask_binary = data_dict["mask"] > 0.5
        x1 = torch.min(torch.where(mask_binary)[1]).item()
        y1 = torch.min(torch.where(mask_binary)[0]).item()
        x2 = torch.max(torch.where(mask_binary)[1]).item()
        y2 = torch.max(torch.where(mask_binary)[0]).item()
        data_dict["gt_bbox_scale"] = max(x2 - x1, y2 - y1)
        data_dict["gt_bbox_center"] = torch.tensor(
            [(x1 + x2) / 2, (y1 + y2) / 2], dtype=torch.float32
        )

        # Fix: To combat the fact that YCBinEOAT doesn't have training data from the real object
        data_dict["cad_depth"] = torch.tensor(
            self.get_depth(idx), dtype=torch.float32
        )

        data_dict["depth"] = torch.tensor(self.get_depth(idx), dtype=torch.float32)

        return data_dict 



class YCBinEOATDataset(torch.utils.data.Dataset):
    videoname_to_object = {
        "bleach0": "bleach_cleanser",
        "bleach_hard_00_03_chaitanya": "bleach_cleanser",
        "cracker_box_reorient": "cracker_box",
        "cracker_box_yalehand0": "cracker_box",
        "mustard0": "mustard_bottle",
        "mustard_easy_00_02": "mustard_bottle",
        "sugar_box1": "sugar_box",
        "sugar_box_yalehand0": "sugar_box",
        "tomato_soup_can_yalehand0": "tomato_soup_can",
    }

    videoname_to_sam_prompt = {
        "bleach0": [(132, 351), (152, 327), (158, 368), (176, 405), (185, 417)],
        "bleach_hard_00_03_chaitanya": [(112, 375), (120, 335), (143, 306), (165, 270)],
        "cracker_box_reorient": [
            (125, 352),
            (170, 328),
            (153, 376),
            (143, 441),
            (197, 406),
        ],
        "cracker_box_yalehand0": [
            (306, 255),
            (363, 288),
            (318, 306),
            (294, 304),
            (363, 344),
        ],
        "mustard0": [(124, 292), (135, 304), (156, 336)],
        "mustard_easy_00_02": [(169, 269), (140, 288), (120, 310)],
        "sugar_box1": [(133, 362), (144, 381), (162, 403)],
        "sugar_box_yalehand0": [(297, 238), (320, 261), (311, 279)],
        "tomato_soup_can_yalehand0": [(336, 300), (351, 309)],
    }

    def __init__(
        self,
        video_dir: str,
        object_dir: str,
        model_type: ModelType = ModelType.CAD,
        use_cad_rgb: bool = False,
        use_cad_mask: bool = False,
        mask_threshold=0.7,
    ):
        # Experiments
        self.use_cad_rgb = use_cad_rgb
        self.use_cad_mask = use_cad_mask
        if use_cad_mask:
            print(
                "WARNING: Using CAD mask for YCBinEOATDataset. This is fine if being used to create a Gaussian Splat of the CAD, but not for running point tracker."
            )

        self.mask_threshold = mask_threshold
        # Video
        self.video_dir = video_dir
        self.video_rgb_dir = os.path.join(self.video_dir, "rgb")
        self.rgb_video_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.png"))
        self.depth_video_files = sorted(
            glob.glob(f"{self.video_dir}/depth_filled/*.png")
        )
        self.gt_pose_dir = os.path.join(self.video_dir, "annotated_poses")
        self.gt_pose_files = sorted(glob.glob(f"{self.video_dir}/annotated_poses/*"))
        self.gt_mask_files = sorted(glob.glob(f"{self.video_dir}/gt_mask/*"))

        self.K = np.loadtxt(os.path.join(self.video_dir, "cam_K.txt")).reshape(3, 3)
        self.H, self.W = cv2.imread(self.rgb_video_files[0], cv2.IMREAD_COLOR).shape[:2]
        self.masks_dir = os.path.join(self.video_dir, "masks")
        if not os.path.exists(self.masks_dir) or len(os.listdir(self.masks_dir)) == 0:
            segment(
                self.video_rgb_dir,
                self.masks_dir,
                prompts=self.videoname_to_sam_prompt[self.video_name],
            )
        self.mask_files = sorted(glob.glob(f"{self.masks_dir}/*.png"))

        # Video
        self.max_frames = len(self.rgb_video_files)
        self.start_frame = 0
        self.end_frame = self.max_frames
        # Object

        self.object_dir = object_dir
        self.obj_path = os.path.join(self.object_dir, "textured_simple.obj")
        mesh = self.get_mesh()
        self.obj_diameter = get_diameter_from_mesh(mesh)
        self.obj_size = get_size_from_mesh(mesh)
        self.bbox = get_bbox_from_size(self.obj_size)

        # CAD GT RGB and Depth
        self.cad_model = CADModel(self.object_dir, self.K, self.H, self.W)
        self.cad_rgb_dir = os.path.join(self.video_dir, "cad_rgb")
        self.cad_depth_dir = os.path.join(self.video_dir, "cad_depth")
        self._render_cad_gt_rgb_and_depth()  # Get CAD GT RGB and Depth
        self.cad_rgb_files = sorted(glob.glob(f"{self.cad_rgb_dir}/*.png"))
        self.cad_depth_files = sorted(glob.glob(f"{self.cad_depth_dir}/*.tiff"))

        # GSPose support
        self.obj_bbox3d = self.bbox.detach().cpu().numpy()
        self.diameter = self.obj_diameter
        self.bbox3d_diameter = torch.norm(self.obj_size).detach().cpu().numpy()
        self.bbox_diameter = self.bbox3d_diameter
        self.use_binarized_mask = False

        if model_type == ModelType.CAD:
            self.model = CADModel(self.object_dir, self.K, self.H, self.W)
        elif model_type == ModelType.GAUSSIAN:
            self.model = GaussianSplatModel(
                self.object_dir, self.K, self.H, self.W, self, torch.device("cuda")
            )
        else:
            raise ValueError("Invalid model type")

    def reset_frame_range(self):
        self.start_frame = 0
        self.end_frame = self.max_frames

    # For GSPose support
    def __getitem__(self, idx):
        data_dict = dict()
        data_dict["camK"] = torch.tensor(self.K, dtype=torch.float32)
        data_dict["pose"] = torch.tensor(self.get_gt_pose(idx), dtype=torch.float32)
        data_dict["image"] = torch.tensor(self.get_rgb(idx), dtype=torch.float32)
        data_dict["allo_pose"] = torch.tensor(
            egocentric_to_allocentric(self.get_gt_pose(idx)), dtype=torch.float32
        )

        data_dict["image_ID"] = idx
        data_dict["image_path"] = (
            self.rgb_video_files[idx]
            if not self.use_cad_rgb
            else self.cad_rgb_files[idx]
        )
        data_dict["mask"] = torch.tensor(self.get_mask(idx), dtype=torch.float32)
        data_dict["mask_path"] = (
            self.mask_files[idx] if not self.use_cad_mask else self.cad_depth_files[idx]
        )
        data_dict["gt_mask_path"] = (
            self.mask_files[idx] if not self.use_cad_mask else self.cad_depth_files[idx]
        )
        data_dict["coseg_mask_path"] = (
            self.mask_files[idx] if not self.use_cad_mask else self.cad_depth_files[idx]
        )

        mask_binary = data_dict["mask"] > self.mask_threshold
        x1 = torch.min(torch.where(mask_binary)[1]).item()
        y1 = torch.min(torch.where(mask_binary)[0]).item()
        x2 = torch.max(torch.where(mask_binary)[1]).item()
        y2 = torch.max(torch.where(mask_binary)[0]).item()
        data_dict["gt_bbox_scale"] = max(x2 - x1, y2 - y1)
        data_dict["gt_bbox_center"] = torch.tensor(
            [(x1 + x2) / 2, (y1 + y2) / 2], dtype=torch.float32
        )

        # Fix: To combat the fact that YCBinEOAT doesn't have training data from the real object
        data_dict["cad_depth"] = torch.tensor(
            self.get_cad_depth(idx), dtype=torch.float32
        )

        data_dict["depth"] = torch.tensor(self.get_depth(idx), dtype=torch.float32)

        return data_dict

    @property
    def poses(self):
        return self.get_gt_poses()

    @property
    def allo_poses(self):
        return np.stack(
            [egocentric_to_allocentric(pose) for pose in self.get_gt_poses()], axis=0
        )

    @property
    def video_name(self):
        return os.path.basename(self.video_dir)

    def __len__(self):
        return self.end_frame - self.start_frame

    def get_mesh(self) -> trimesh.Trimesh:
        return trimesh.load_mesh(self.obj_path)

    @property
    def image_size(self):
        if self.H is None or self.W is None:
            self.H, self.W = cv2.imread(
                os.listdir(self.video_rgb_dir)[0], cv2.IMREAD_COLOR
            ).shape[:2]
        return self.H, self.W

    def get_gt_poses(self) -> np.ndarray:
        return np.array([self.get_gt_pose(i) for i in range(len(self))])[
            self.start_frame : self.end_frame
        ]

    def get_gt_pose(self, idx: int) -> np.ndarray:
        idx += self.start_frame
        file = self.gt_pose_files[idx]
        return np.loadtxt(file).reshape(4, 4)

    def get_rgb(self, idx: int) -> np.ndarray:
        if self.use_cad_rgb:
            return self.get_cad_rgb(idx)
        idx += self.start_frame
        return cv2.cvtColor(
            cv2.imread(self.rgb_video_files[idx], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )

    def get_depth(self, idx: int) -> np.ndarray:
        if self.use_cad_rgb:
            return self.get_cad_depth(idx)
        idx += self.start_frame
        return cv2.imread(self.depth_video_files[idx], cv2.IMREAD_UNCHANGED) / 1e3

    def get_cad_rgb(self, idx: int) -> np.ndarray:
        idx += self.start_frame
        return cv2.cvtColor(
            cv2.imread(self.cad_rgb_files[idx], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )

    def get_mask(self, idx: int) -> np.ndarray:
        if self.use_cad_mask:
            return self.get_cad_depth(idx) > 0
        idx += self.start_frame
        return cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE) / 255

    def get_gt_mask(self, idx: int) -> np.ndarray:
        if self.use_cad_mask:
            return self.get_cad_depth(idx) > 0
        idx += self.start_frame
        return cv2.imread(self.gt_mask_files[idx], cv2.IMREAD_GRAYSCALE) / 255

    def get_cad_depth(self, idx: int) -> np.ndarray:
        idx += self.start_frame
        depth_image = Image.open(self.cad_depth_files[idx])

        return np.array(depth_image).astype(np.float32)

    def render_mesh_at_pose(
        self,
        pose: Optional[np.ndarray] = None,
        idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert (pose is None) != (idx is None)
        pose = self.get_gt_pose(idx) if pose is None else pose
        if not self.model:
            return self.cad_model.render(pose)
        return self.model.render(pose)

    def _get_safe_distance(self):
        # TODO: Implement smallest distance from the camera to the object such that it is fully visible
        f_x = self.K[0, 0]
        f_y = self.K[1, 1]
        c_x = self.K[0, 2]
        c_y = self.K[1, 2]

        d_x = 2 * min(c_x, self.W - c_x - 1)
        d_y = 2 * min(c_y, self.H - c_y - 1)

        return self.obj_diameter * 1.2

    def get_canonical_pose(self):
        canonical_pose = np.eye(4)
        diameter = self.obj_diameter

        # Translate along z-axis by diameter
        canonical_pose[:3, 3] = np.array([0, 0, self._get_safe_distance()])
        # Rotate 90 degrees around x-axis then rotate around y-axis 180 degrees
        canonical_pose[:3, :3] = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])

        return canonical_pose

    def _render_cad_gt_rgb_and_depth(self):
        if (
            os.path.exists(self.cad_rgb_dir)
            and os.path.exists(self.cad_depth_dir)
            and len(os.listdir(self.cad_rgb_dir))
            == self.max_frames
            == len(os.listdir(self.cad_depth_dir))
        ):
            return

        os.path.exists(self.cad_rgb_dir) or os.makedirs(self.cad_rgb_dir)
        os.path.exists(self.cad_depth_dir) or os.makedirs(self.cad_depth_dir)
        for file in os.listdir(self.cad_rgb_dir):
            os.remove(os.path.join(self.cad_rgb_dir, file))
        for file in os.listdir(self.cad_depth_dir):
            os.remove(os.path.join(self.cad_depth_dir, file))

        for i in tqdm(range(len(self)), desc="Rendering CAD GT RGB and Depth"):
            real_rgb = self.get_rgb(i)
            real_mask = self.get_mask(i)
            if (pose_i := self.get_gt_pose(i)) is not None:
                pose = pose_i
            rgb, depth, _ = self.render_mesh_at_pose(pose=pose)
            cv2.imwrite(
                os.path.join(self.cad_rgb_dir, f"{i:05d}.png"),
                cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
            )

            # overlap
            overlap = cv2.addWeighted(real_rgb, 0.5, rgb, 0.5, 0)

            depth_image_32bit = Image.fromarray(depth.astype(np.float32))
            depth_image_32bit.save(os.path.join(self.cad_depth_dir, f"{i:05d}.tiff"))


class ImageBatchIterator:
    """
    Iterator that loads images from a folder and yields batches of torch tensors.
    """

    def __init__(
        self,
        folder_path: str,
        batch_size: int,
        overlap: int = 0,
        limit: Optional[int] = None,
        image_size: Optional[Tuple[int, int]] = None,
        shuffle: bool = False,
        rgb_mode: bool = False,
        device: torch.device = torch.device("cuda"),
    ):
        """
        Initialize the iterator.

        Args:
            folder_path: Path to the folder containing images
            batch_size: Number of images per batch
            overlap: Number of images to overlap between consecutive batches
            limit: Maximum number of images to load
            image_size: Tuple of (height, width) to resize images to
            shuffle: Whether to shuffle the dataset each epoch
        """
        if overlap >= batch_size:
            raise ValueError("Overlap must be less than batch_size")

        self.folder_path = folder_path
        self.batch_size = batch_size
        self.overlap = overlap
        self.image_size = image_size
        self.shuffle = shuffle
        self.rgb_mode = rgb_mode
        self.device = device

        # Get image size from first image if not provided
        if image_size is None:
            with Image.open(
                os.path.join(folder_path, os.listdir(folder_path)[0])
            ) as img:
                self.image_size = tuple(img.size[::-1])

        # Get list of image files
        self.image_files = sorted(
            [
                f
                for f in os.listdir(folder_path)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp"))
            ]
        )
        if limit is not None:
            self.image_files = self.image_files[:limit]

        # Initialize transform pipeline
        self.transform = transforms.Compose(
            [
                # transforms.Resize(self.image_size),
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #               std=[0.229, 0.224, 0.225])
            ]
        )

        self.current_index = 0

        if self.shuffle:
            np.random.shuffle(self.image_files)

    def __iter__(self) -> Iterator[torch.Tensor]:
        return self

    def __next__(self) -> torch.Tensor:
        if self.current_index >= len(self.image_files):
            raise StopIteration

        # Calculate batch indices considering overlap
        start_idx = max(0, self.current_index - self.overlap)
        end_idx = min(start_idx + self.batch_size, len(self.image_files))

        batch_files = self.image_files[start_idx:end_idx]

        # Load and process images
        batch_tensors = []
        for img_file in batch_files:
            img_path = os.path.join(self.folder_path, img_file)
            try:
                with Image.open(img_path) as img:
                    if self.rgb_mode and img.mode != "RGB":
                        img = img.convert("RGB")
                    tensor = self.transform(img)
                    batch_tensors.append(tensor)
            except Exception as e:
                print(f"Error loading image {img_file}: {str(e)}")
                continue

        # Update current_index for next batch (considering non-overlapped portion)
        self.current_index += self.batch_size - self.overlap

        # Stack tensors into a batch
        if not batch_tensors:
            raise StopIteration

        return torch.stack(batch_tensors)[None].to(self.device)

    def __len__(self) -> int:
        if self.overlap == 0:
            return (len(self.image_files) + self.batch_size - 1) // self.batch_size
        else:
            # Calculate number of batches with overlap
            return max(
                1,
                (len(self.image_files) - self.overlap)
                // (self.batch_size - self.overlap),
            )


def load_video_images(
    video_img_directory: str,
    limit: Optional[int] = None,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    dtype=torch.uint8,
    file_type: Optional[str] = None,
) -> torch.Tensor:
    # Collect and sort filenames
    if file_type is None:
        video_files = (
            sorted(Path(video_img_directory).glob("*.jpg"))[:limit]
            + sorted(Path(video_img_directory).glob("*.png"))[:limit]
        )
    else:
        video_files = sorted(Path(video_img_directory).glob(f"*.{file_type}"))[:limit]

    # Function to read an image
    def read_image(file_path):
        return cv2.cvtColor(cv2.imread(str(file_path)), cv2.COLOR_BGR2RGB)

    # Parallel reading
    with ThreadPoolExecutor() as executor:
        images = list(executor.map(read_image, video_files))

    # Convert to a NumPy array
    video_np = np.stack(images, axis=0)

    # Convert to PyTorch tensor
    video = (
        torch.from_numpy(video_np)
        .to(dtype=dtype, device=device)
        .permute(0, 3, 1, 2)
        .unsqueeze(0)
    )

    return video
