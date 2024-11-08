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
from posingpixels.utils.offscreen_renderer import ModelRendererOffscreen
import trimesh
from posingpixels.utils.meshes import get_bbox_from_size, get_diameter_from_mesh, get_size_from_mesh
from posingpixels.segmentation import segment


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

    videoname_to_sam_prompt = {"mustard0": [(124, 292), (135, 304), (156, 336)]}

    def __init__(self, video_dir: str, object_dir: str):
        # Video
        self.video_dir = video_dir
        self.video_rgb_dir = os.path.join(self.video_dir, "rgb")
        self.rgb_video_files = sorted(glob.glob(f"{self.video_dir}/rgb/*.png"))
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

        # Object
        self.object_dir = object_dir
        self.obj_path = os.path.join(self.object_dir, "textured_simple.obj")
        mesh = self.get_mesh()
        self.obj_diameter = get_diameter_from_mesh(mesh)
        self.obj_size = get_size_from_mesh(mesh)
        self.bbox = get_bbox_from_size(self.obj_size)
        self.renderer = ModelRendererOffscreen(self.K, self.H, self.W)
        self.cad_rgb_dir = os.path.join(self.video_dir, "cad_rgb")
        self.cad_depth_dir = os.path.join(self.video_dir, "cad_depth")
        self._render_cad_gt_rgb_and_depth()  # Get CAD GT RGB and Depth
        self.cad_rgb_files = sorted(glob.glob(f"{self.cad_rgb_dir}/*.png"))
        self.cad_depth_files = sorted(glob.glob(f"{self.cad_depth_dir}/*.tiff"))

    @property
    def video_name(self):
        return os.path.basename(self.video_dir)

    def __len__(self):
        return len(self.rgb_video_files)

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
        return np.array([self.get_gt_pose(i) for i in range(len(self))])
        

    def get_gt_pose(self, idx: int) -> Optional[np.ndarray]:
        file = self.gt_pose_files[idx]
        return np.loadtxt(file).reshape(4, 4)

    def get_rgb(self, idx: int) -> np.ndarray:
        return cv2.cvtColor(
            cv2.imread(self.rgb_video_files[idx], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )

    def get_cad_rgb(self, idx: int) -> np.ndarray:
        return cv2.cvtColor(
            cv2.imread(self.cad_rgb_files[idx], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB
        )

    def get_mask(self, idx: int) -> np.ndarray:
        return cv2.imread(self.mask_files[idx], cv2.IMREAD_GRAYSCALE) / 255

    def get_gt_mask(self, idx: int) -> np.ndarray:
        return cv2.imread(self.gt_mask_files[idx], cv2.IMREAD_GRAYSCALE)

    def get_cad_depth(self, idx: int) -> np.ndarray:
        depth_image = Image.open(self.cad_depth_files[idx])

        return np.array(depth_image).astype(np.float32)

    def render_mesh_at_pose(
        self, pose: Optional[np.ndarray] = None, idx: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        assert (pose is None) != (idx is None)
        pose = self.get_gt_pose(idx) if pose is None else pose
        return self.renderer.render(pose, self.get_mesh())

    def get_canonical_pose(self):
        canonical_pose = np.eye(4)
        diameter = self.obj_diameter

        # Translate along z-axis by diameter
        canonical_pose[:3, 3] = np.array([0, 0, diameter])
        # Rotate 90 degrees around x-axis then rotate around y-axis 180 degrees
        canonical_pose[:3, :3] = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]])

        return canonical_pose

    def _render_cad_gt_rgb_and_depth(self):
        if (
            os.path.exists(self.cad_rgb_dir)
            and os.path.exists(self.cad_depth_dir)
            and len(os.listdir(self.cad_rgb_dir))
            == len(self)
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
            rgb, depth = self.render_mesh_at_pose(pose=pose)
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
