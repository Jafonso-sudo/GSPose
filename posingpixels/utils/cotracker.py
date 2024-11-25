from cotracker.utils.visualizer import Visualizer
import numpy as np


from typing import Tuple

import torch

from posingpixels.utils.geometry import apply_pose_to_points, render_points_in_2d


def sample_support_grid_points(
    H: int, W: int, query_frame_idx: int, query_mask: np.ndarray, grid_size: int = 10
):
    """
    Generates a grid of sample support points within the given frame dimensions,
    excluding the border and avoiding points inside the query mask.

    Args:
        H (int): Height of the frame.
        W (int): Width of the frame.
        query_frame_idx (int): The frame index to which the points belong.
        query_mask (np.ndarray): A binary mask indicating the regions to avoid.
        grid_size (int, optional): The number of grid points along each dimension.
                                   Defaults to 10.

    Returns:
        np.ndarray: An array of shape (N, 3) where N is the number of valid grid points.
                    Each row contains [query_frame, x, y] coordinates of a grid point.
    """
    grid_size += 2
    x = np.linspace(0, W, grid_size, dtype=int)[1:-1]
    y = np.linspace(0, H, grid_size, dtype=int)[1:-1]
    xx, yy = np.meshgrid(x, y)
    pixels = np.stack([xx.flatten(), yy.flatten()], axis=1)
    # Prevent pixels from being inside the query_mask
    pixels = pixels[query_mask[yy.flatten(), xx.flatten()] == 0]
    return np.concatenate(
        [
            query_frame_idx * np.ones((len(pixels), 1)),
            pixels,
        ],
        axis=1,
    )


def get_ground_truths(
    pose, K, unposed_3d_points, prob_mask, depth
) -> Tuple[np.ndarray, np.ndarray]:
    posed_3d_points = apply_pose_to_points(unposed_3d_points, pose[:3, :3], pose[:3, 3])
    gt_coords = render_points_in_2d(posed_3d_points, K[:3, :3])
    # Get the depth values posed_3d_points[:, 2] at the 2D coordinates gt_coords
    gt_depth = depth[gt_coords[:, 1].astype(int), gt_coords[:, 0].astype(int)]
    # Calculate visibility based on depth values
    gt_visibility = np.abs(gt_depth - posed_3d_points[:, 2]) < 0.002
    # Apply prob_mask
    gt_visibility = (
        gt_visibility
        * prob_mask[gt_coords[:, 1].astype(int), gt_coords[:, 0].astype(int)]
    )

    return gt_coords, gt_visibility

def scale_by_crop(points, bboxes, scaling_factors):
    """
    Scale points by the scaling factor and crop bounding box.

    Args:
        points: 2D points to scale (B, N, 2)
        bboxes: Bounding boxes to crop (B, 4)
        scaling_factors: Scaling factors to apply (B, 2)

    Returns:
        Scaled and cropped points (B, N, 3)
    """
    points = points.clone()
    bboxes = bboxes.unsqueeze(1).repeat(1, points.shape[1], 1)
    scaling_factors = scaling_factors.unsqueeze(1).repeat(1, points.shape[1], 1)
    points -= bboxes[:, :, :2]
    points *= scaling_factors

    return points

def unscale_by_crop(points, bboxes, scaling_factors):
    """
    Reverse the scaling and cropping operations to get original coordinates.
    
    Args:
        points: 2D points in cropped space (B, N, 2)
        bboxes: Bounding boxes used for cropping (B, 4)
        scaling_factors: Scaling factors that were applied (B, 2)
        
    Returns:
        Original uncropped and unscaled points (B, N, 2)
    """
    points = points.clone()
    bboxes = bboxes.unsqueeze(1)#.repeat(1, points.shape[1], 1)
    scaling_factors = scaling_factors.unsqueeze(1)#.repeat(1, points.shape[1], 1)
    
    # First reverse the scaling
    points /= scaling_factors
    
    # Then reverse the cropping by adding back the top-left corner offset
    points += bboxes[:, :, :2]
    
    return points

def get_tracks_outside_mask(pred_tracks: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """
    Find indices of track points that fall on pixels where the mask is False for each timestamp.
    
    Args:
        pred_tracks: Tensor of shape (T, N, 2) containing pixel coordinates for each point at each timestamp
        masks: Tensor of shape (T, H, W) containing boolean values for each pixel at each timestamp
    
    Returns:
        Tensor of shape (M, 2) containing (timestamp, point_idx) pairs where M is the number of
        track points that fall outside their corresponding timestamp's mask
    """
    T, N, _ = pred_tracks.shape
    _, H, W = masks.shape
    
    # Convert track coordinates to integers for indexing
    track_coords = pred_tracks.round().long()
    
    # Clamp coordinates to valid image dimensions
    track_coords[..., 0] = torch.clamp(track_coords[..., 0], 0, W-1)
    track_coords[..., 1] = torch.clamp(track_coords[..., 1], 0, H-1)
    
    # Create time indices for all points
    time_indices = torch.arange(T, device=pred_tracks.device).unsqueeze(1).expand(T, N)
    
    # Get mask values for each track point at its corresponding timestamp
    track_mask_values = masks[
        time_indices,
        track_coords[..., 1],  # y coordinates
        track_coords[..., 0]   # x coordinates
    ]  # Shape: (T, N)
    
    # Find indices where track points are outside the mask (mask is False)
    unmasked_indices = torch.where(~track_mask_values)
    
    # Stack the indices to get (timestamp, point_idx) pairs
    return torch.stack(unmasked_indices, dim=1)

def visualize_results(
    video,
    pred_tracks,
    pred_visibility,
    pred_confidence,
    save_dir,
    num_of_main_queries=None,
    filename="video",
    threshold=0.6,
):
    if num_of_main_queries is None:
        num_of_main_queries = pred_tracks.shape[2]
    vis = Visualizer(save_dir=save_dir, pad_value=0, linewidth=3)
    vis.visualize(
        video,
        pred_tracks[:, :, :num_of_main_queries, :],
        (pred_visibility * pred_confidence > threshold)[:, :, :num_of_main_queries],
        filename=filename,
    )