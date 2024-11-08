from typing import Dict, Optional
from matplotlib import pyplot as plt
import torch
from pytorch3d.vis import plotly_vis
from pytorch3d.structures import Pointclouds
import numpy as np
import cv2
from gaussian_object.gaussian_model import GaussianModel
from inference import render_Gaussian_object_model
from misc_utils import gs_utils


def get_points_pointcloud(
    points: np.ndarray, color: Optional[np.ndarray] = None
) -> Pointclouds:
    """
    Create a point cloud from a list of points.

    Parameters
    ----------
    points : np.ndarray
        The list of points.
    color : Optional[np.ndarray], optional
        The color of the points, by default None.

    Returns
    -------
    Pointclouds
        The point cloud.
    """
    if color is None:
        color = np.array([255, 0, 0])
    if len(color.shape) == 1:
        colors = np.tile(color / 255, (points.shape[0], 1))
    else:
        colors = color / 255

    return Pointclouds(torch.tensor(points[None]), features=torch.tensor(colors[None]))


def get_ray_pointcloud(
    ray_origin: np.ndarray,
    ray_direction: np.ndarray,
    length: float,
    n_points: int,
    color: Optional[np.ndarray] = None,
) -> Pointclouds:
    """
    Create a point cloud along a ray.

    Parameters
    ----------
    ray_origin : np.ndarray
        The origin of the ray.
    ray_direction : np.ndarray
        The direction of the ray.
    length : float
        The length of the ray.
    n_points : int
        The number of points in the point cloud.
    color : Optional[np.ndarray], optional
        The color of the points, by default red.

    Returns
    -------
    Pointclouds
        The point cloud.
    """
    if color is None:
        color = np.array([255, 0, 0])
    if len(color.shape) == 1:
        colors = np.tile(color / 255, (n_points, 1))
    else:
        colors = color / 255

    points = ray_origin + np.outer(np.linspace(0, length, n_points), ray_direction)

    return Pointclouds(torch.tensor(points[None]), features=torch.tensor(colors[None]))


def get_gaussian_splat_pointcloud(
    gaussian_object: GaussianModel, pose: Optional[np.ndarray] = None
) -> Pointclouds:
    """
    Create a point cloud of the Gaussian object.

    Parameters
    ----------
    gaussian_object : GaussianModel
        The Gaussian object.
    pose : Optional[np.ndarray], optional
        The pose of the object, by default None.

    Returns
    -------
    Pointclouds
        The point cloud.
    """
    gaussian_object.initialize_pose()  # IMPORTANT: Otherwise the object will be rendered with the previous pose applied
    if pose is None:
        pose = np.eye(4)

    points = gaussian_object.get_xyz.squeeze().detach().cpu().numpy()
    colors = gaussian_object._features_dc.squeeze().detach().cpu().sigmoid().numpy()

    posed_points = points @ pose[:3, :3].T + pose[:3, 3][None, :]

    return Pointclouds(
        torch.tensor(posed_points[None]), features=torch.tensor(colors[None])
    )


def plot_pointclouds(
    pointclouds: Dict[str, Pointclouds], title: str = "Pointcloud"
) -> None:
    """
    Plot a point cloud.

    Parameters
    ----------
    pointclouds : List[Pointclouds]
        The point cloud.
    title : str, optional
        The title of the plot, by default "Pointcloud".
    """
    fig = plotly_vis.plot_scene(
        {title: pointclouds},
        xaxis={"backgroundcolor": "rgb(200, 200, 230)"},
        yaxis={"backgroundcolor": "rgb(230, 200, 200)"},
        zaxis={"backgroundcolor": "rgb(200, 230, 200)"},
        pointcloud_marker_size=3,
        pointcloud_max_points=30_000,
        axis_args=plotly_vis.AxisArgs(showgrid=True),
    )

    fig.update_layout(width=800, height=600)
    fig.show()


def video_to_grayscale(video: np.ndarray) -> np.ndarray:
    """
    Convert a video to grayscale.

    Parameters
    ----------
    video : np.ndarray
        The video.

    Returns
    -------
    np.ndarray
        The grayscale video.
    """
    return np.array(
        [
            cv2.cvtColor(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), cv2.COLOR_GRAY2RGB)
            for frame in video
        ]
    )


def plot_points_on_video(
    video_frames: np.ndarray, points: np.ndarray, colors: np.ndarray
) -> np.ndarray:
    """
    Plot points on a video.

    Parameters
    ----------
    video_frames : np.ndarray
        The video frames.
    points : np.ndarray
        The points to plot.
    colors : np.ndarray
        The colors of the points. Either a single color or a color for each point.
    """
    if len(colors.shape) == 1:
        colors = np.tile(colors[None], (points.shape[0], 1))
    assert len(points) == len(colors)
    new_video_frames = []
    for i, frame in enumerate(video_frames):
        frame = frame.copy()
        frame_points = points[i]
        for p, c in zip(frame_points, colors):
            frame = cv2.circle(frame, tuple(p.astype(int)), 3, c.tolist(), -1)
        new_video_frames.append(frame)

    return np.array(new_video_frames)


def overlay_gaussian_splat_on_video(
    video_frames: np.ndarray,
    gaussian_object: GaussianModel,
    camKs: np.ndarray,
    poses: Optional[np.ndarray] = None,
    original_opacity=0.4,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """
    Overlay the Gaussian object on a video.

    Parameters
    ----------
    video_frames : np.ndarray
        The video frames.
    gaussian_object : GaussianModel
        The Gaussian object.
    pose : Optional[np.ndarray], optional
        The pose of the object, by default None.

    Returns
    -------
    np.ndarray
        The video frames with the Gaussian object overlaid.
    """
    if not device:
        device = torch.device("cuda")
    if poses is None:
        poses = np.tile(np.eye(4), (len(video_frames), 1, 1))

    new_video_frames = []
    for frame, pose, camK in zip(video_frames, poses, camKs):
        render = render_Gaussian_object_model(
            gaussian_object, camK, pose, frame.shape[0], frame.shape[1], device
        )
        frame = cv2.addWeighted(
            cv2.cvtColor(render, cv2.COLOR_BGR2HSV),
            1 - original_opacity,
            frame,
            original_opacity,
            1,
        )
        new_video_frames.append(frame)

    return np.array(new_video_frames)


def overlay_bounding_box_on_video(
    video_frames: np.ndarray,
    cannon_3D_bbox: np.ndarray,
    camKs: np.ndarray,
    poses: np.ndarray,
    color=(0, 255, 0),
) -> np.ndarray:
    """
    Overlay the bounding box of the object on a video.

    Parameters
    ----------
    video_frames : np.ndarray
        The video frames.
    reference_database : dict
        The reference database for the Gaussian splat object.
    poses : np.ndarray
        The poses of the object.

    Returns
    -------
    np.ndarray
        The video frames with the bounding box overlaid.
    """
    new_video_frames = []
    for frame, pose, camK in zip(video_frames, poses, camKs):
        track_RT = torch.as_tensor(pose, dtype=torch.float32)
        track_bbox_KRT = (
            torch.einsum("ij,kj->ki", track_RT[:3, :3], cannon_3D_bbox)
            + track_RT[:3, 3][None, :]
        )
        track_bbox_KRT = torch.einsum("ij,kj->ki", camK, track_bbox_KRT)
        track_bbox_pts = (
            (track_bbox_KRT[:, :2] / track_bbox_KRT[:, 2:3]).type(torch.int64).numpy()
        )
        track_bbox3d_img = gs_utils.draw_3d_bounding_box(
            frame.copy(), track_bbox_pts, color=color, linewidth=5
        )
        new_video_frames.append(track_bbox3d_img)

    return np.array(new_video_frames)

def plot_per_point_losses(
    predictions: np.ndarray, target: np.ndarray, criterion: torch.nn.Module, title: str = "Per Point Losses",  device: Optional[torch.device] = None
):
    """
    Plot the per point losses.

    Parameters
    ----------
    predictions : np.ndarray
        The predictions.
    target : np.ndarray
        The target.
    criterion : torch.nn.Module
        The loss function.
    title : str, optional
        The title of the plot, by default "Per Point Losses".
    """
    if not device:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    criterion = criterion.to(device)
    
    predictions_torch = torch.tensor(predictions, device=device)
    target_torch = torch.tensor(target, device=device)
    
    losses = criterion(predictions_torch, target_torch)
    # Sum over the last dimension
    losses = losses.sum(-1)
    
    return losses.cpu().numpy()