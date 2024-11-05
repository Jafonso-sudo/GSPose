import subprocess
import os
import sys

import numpy as np
import torch

from typing import Tuple
import torch.nn.functional as F

file_dir = os.path.dirname(os.path.realpath(__file__))
proj_root = os.path.abspath(os.path.join(file_dir, os.pardir))
sys.path.append(proj_root)

sam2_dir = os.path.join(proj_root, os.pardir, "segment-anything-2")


def segment(input_image_dir, output_mask_dir, prompts=None):
    # Check existence of valid input image directory
    if (
        not os.path.exists(input_image_dir)
        or not os.path.isdir(input_image_dir)
        or len(os.listdir(input_image_dir)) == 0
    ):
        raise Exception(f"Input image directory not found at {input_image_dir}")
    # Create or clear output mask directory
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)
    for file in os.listdir(output_mask_dir):
        os.remove(os.path.join(output_mask_dir, file))
    prompts_str = f"--prompts '{prompts}'" if prompts else ""
    print(f"Running SAM2 on input image directory {input_image_dir}")
    command = f"""
source $(conda info --base)/etc/profile.d/conda.sh && \
conda activate sam2 && \
python {sam2_dir}/notebooks/posing_pixels.py {input_image_dir} {output_mask_dir} {prompts_str} && \
conda deactivate
""".strip()
    subprocess.run(command, shell=True, check=True, executable="/bin/bash")
    print(f"Finished running SAM2, masks saved to {output_mask_dir}")


def get_bbox_from_mask(mask):
    # Find indices of True values
    rows, cols = np.where(mask)

    if len(rows) == 0:
        return None  # Return None if mask is empty

    # Get bounding box coordinates
    min_row, max_row = rows.min(), rows.max()
    min_col, max_col = cols.min(), cols.max()

    return (min_col, min_row, max_col, max_row)


def process_image_crop(
    image: np.ndarray, bbox: tuple, padding: int, target_size: tuple
) -> Tuple[np.ndarray, tuple, tuple]:
    """
    Process an image by cropping around a bounding box with padding and resizing to target dimensions.

    Args:
        image: Input image as numpy array (H, W, C)
        bbox: Tuple of (x1, y1, x2, y2) coordinates
        padding: Initial padding amount in pixels
        target_size: Tuple of (target_height, target_width)

    Returns:
        tuple containing:
        - Processed image as numpy array
        - New bounding box coordinates
        - Zoom factor applied
    """
    orig_h, orig_w = image.shape[:2]
    target_h, target_w = target_size
    x1, y1, x2, y2 = bbox

    # Calculate initial crop dimensions with padding
    crop_x1 = max(0, x1 - padding)
    crop_y1 = max(0, y1 - padding)
    crop_x2 = min(orig_w, x2 + padding)
    crop_y2 = min(orig_h, y2 + padding)

    # Crop the image
    cropped = image[crop_y1:crop_y2, crop_x1:crop_x2]

    # Convert cropped image to tensor and add batch and channel dimensions
    cropped_tensor = torch.tensor(cropped.copy()).permute(2, 0, 1).unsqueeze(0).float()

    # Interpolate to target size
    scaled = F.interpolate(
        cropped_tensor, size=target_size, mode="bilinear", align_corners=True
    )

    # Remove batch and channel dimensions
    scaled = scaled.squeeze(0).permute(1, 2, 0)

    # Calculate scaling factors
    scale_x = target_w / (crop_x2 - crop_x1)
    scale_y = target_h / (crop_y2 - crop_y1)

    return (
        scaled.detach().cpu().numpy(),
        (crop_x1, crop_y1, crop_x2, crop_y2),
        (scale_x, scale_y),
    )
