import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import os
import torch
from typing import Iterator, Optional, Tuple
import cv2

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor


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
) -> torch.Tensor:
    # Collect and sort filenames
    video_files = sorted(Path(video_img_directory).glob("*.jpg"))[:limit]

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
