from typing import Dict, List, Optional
import torch
import pytorch3d
from pytorch3d.vis import plotly_vis
from pytorch3d.structures import join_meshes_as_scene, Meshes, Pointclouds
import numpy as np
import math
from gaussian_object.gaussian_model import GaussianModel

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
    if not color:
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
    if not color:
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
    if not pose:
        pose = np.eye(4)

    points = gaussian_object.get_xyz.squeeze().detach().cpu().numpy()
    colors = gaussian_object._features_dc.squeeze().detach().cpu().sigmoid().numpy()

    posed_points = points @ pose[:3, :3].T + pose[:3, 3][None, :]

    return Pointclouds(torch.tensor(posed_points[None]), features=torch.tensor(colors[None]))


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
