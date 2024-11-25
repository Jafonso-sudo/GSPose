from typing import Optional, Tuple
import cv2
import numpy as np
import tqdm
import trimesh
from gaussian_object.gaussian_model import GaussianModel

from posingpixels.utils.geometry import (
    pixel_to_ray_dir,
    ray_splat_intersection,
    revert_pose_to_ray,
)

import matplotlib.pyplot as plt

from posingpixels.utils.offscreen_renderer import ModelRendererOffscreen
from posingpixels.visualization import get_gaussian_splat_pointcloud, plot_pointclouds
from posingpixels.visualization import get_points_pointcloud
from posingpixels.utils.gs_pose import render_gaussian_model_with_info
from posingpixels.utils.alignment import get_boolean_mask, sample_safe_zone

from posingpixels.utils.geometry import (
    apply_pose_to_points,
    render_points_in_2d,
)


class PixelToGaussianAligner:
    """
    A class to align pixel coordinates to a Gaussian splat based on a mask and tracked points.

    mask : np.ndarray
        A binary matrix indicating whether a pixel from the first frame belongs to the tracked object or not.
    tracks : np.ndarray
        A 3D array of shape (n_frames, n_points, 2) containing the pixel coordinates of the tracked points.
    """

    mask: np.ndarray
    tracks: np.ndarray

    def __init__(
        self,
        mask_path: str,
        pixeltracker_path: str,
        gaussian_object: GaussianModel,
        initial_T: np.ndarray,
        initial_R: np.ndarray,
        initial_cam_K: np.ndarray,
        pixeltracker_upscale: float = 1.0,
    ):
        """
        Initialize the PixelToGaussianAligner with a mask and tracked points.

        Parameters
        ----------
        mask_path : str
            Path to the segmentation mask of the object for the first frame.
        pixeltracker_path : str
            Path to the .npy file containing the tracked points.
        pixeltracker_upscale : float, optional
            Factor to upscale the pixeltracker coordinates, by default 1.0.
        """
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0.9
        self.H, self.W = self.mask.shape
        self.tracks = pixeltracker_upscale * np.load(pixeltracker_path)[:, :, :2]
        self.init_T, self.init_R, self.init_K = initial_T, initial_R, initial_cam_K
        self.gaussian_object = gaussian_object

        self.tracks = self.filter_object_tracks(self.mask, self.tracks)

    @staticmethod
    def filter_object_tracks(mask: np.ndarray, tracks: np.ndarray) -> np.ndarray:
        frame = tracks[0]
        object_idx = mask[frame[:, 1].astype(int), frame[:, 0].astype(int)]
        return tracks[:, object_idx]

    def align(self) -> tuple:
        """
        Align the tracked points to the Gaussian object.

        Returns
        -------
        tuple
            A tuple containing the intersections of the tracked points with the Gaussian object and the filtered tracks.
        """

        self.gaussian_object.initialize_pose()  # IMPORTANT: Otherwise the object will be rendered with the previous pose applied

        frame = self.tracks[0, :, :2]

        intersections = []
        filtered_idx = []

        for i, p in tqdm.tqdm(enumerate(frame), total=frame.shape[0]):
            pixel_ray_dir = pixel_to_ray_dir(p, self.init_K)
            ray_origin, ray_direction = revert_pose_to_ray(
                np.zeros(3), pixel_ray_dir, self.init_R, self.init_T
            )

            intersection_t = ray_splat_intersection(
                ray_origin, ray_direction, self.gaussian_object
            )
            if intersection_t is not None:
                intersection_point = ray_origin + intersection_t * ray_direction
                intersections.append(intersection_point)
            else:
                filtered_idx.append(i)

        # Filter out points that did not intersect with the Gaussian object
        filtered_tracks = np.delete(self.tracks, filtered_idx, axis=1)

        return intersections, filtered_tracks


class DepthInformedPixelToGaussianAligner:
    """
    A class to align pixel coordinates to a Gaussian splat based on predicted depth map.
    """

    def __init__(
        self,
        T: np.ndarray,
        R: np.ndarray,
        K: np.ndarray,
        depth_map: np.ndarray,
    ):
        """
        Initialize the PixelToGaussianAligner with a mask and tracked points.

        Parameters
        ----------
        mask_path : str
            Path to the segmentation mask of the object for the first frame.
        pixeltracker_path : str
            Path to the .npy file containing the tracked points.
        pixeltracker_upscale : float, optional
            Factor to upscale the pixeltracker coordinates, by default 1.0.
        """
        self.T, self.R, self.K = T, R, K
        self.depth_map = depth_map

    def align(self, points: np.ndarray) -> np.ndarray:
        """
        Align the tracked points to the Gaussian object.

        Parameters
        ----------
        points : np.ndarray
            A 2D array of shape (n_points, 2) containing the pixel coordinates of the tracked points.

        Returns
        -------
        np.ndarray
            A 2D array of shape (n_points, 3) containing the intersections of the points with the Gaussian object.
        """
        intersections = []
        for p in points:
            pixel_ray_dir = pixel_to_ray_dir(p, self.K)
            ray_origin, ray_direction = revert_pose_to_ray(
                np.zeros(3), pixel_ray_dir, self.R, self.T
            )
            depth = self.depth_map[p[1], p[0]]
            t = depth / pixel_ray_dir[2]
            point_3d = ray_origin + t * ray_direction
            intersections.append(point_3d)
        intersections = np.array(intersections)
        # intersection_identity = reverse_pose_to_points(intersections, self.R, self.T)

        return intersections


def get_safe_query_points(
    R: np.ndarray,
    T: np.ndarray,
    camK: np.ndarray,
    H: int,
    W: int,
    object: Optional[GaussianModel] = None,
    mesh: Optional[trimesh.Trimesh] = None,
    alpha_threshold: float = 0.9,
    depth_margin: int = 6,
    depth_change: float = 0.01,
    alpha_margin: int = 15,
    alpha_change: float = 0.01,
    min_pixel_distance: int = 25,
    frame_idx: int = 0,
    debug_vis: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Computes safe query points for alignment based on the provided object or mesh.
    Parameters:
    R (np.ndarray): Rotation matrix.
    T (np.ndarray): Translation vector.
    camK (np.ndarray): Camera intrinsic matrix.
    H (int): Height of the image.
    W (int): Width of the image.
    object (Optional[GaussianModel]): Gaussian model of the object. Default is None.
    mesh (Optional[trimesh.Trimesh]): Mesh of the object. Default is None.
    alpha_threshold (float): Threshold for alpha values to consider a pixel. Default is 0.9.
    depth_margin (int): Margin for depth consistency. Default is 6.
    depth_change (float): Allowed change in depth for consistency. Default is 0.01.
    alpha_margin (int): Margin for alpha consistency. Default is 15.
    alpha_change (float): Allowed change in alpha for consistency. Default is 0.01.
    min_pixel_distance (int): Minimum distance between sampled pixels. Default is 25.
    frame_idx (int): Frame index for the current image. Default is 0.
    debug_vis (bool): Flag to enable debug visualization. Default is False.
    Returns:
    Tuple[np.ndarray, np.ndarray]: Unposed intersections n x 3 and projected intersections with frame index n x 3.
    """
    assert object or mesh, "Either object or mesh must be provided."
    if object:
        render = render_gaussian_model_with_info(object, camK, H, W, R=R, T=T)
        rgb = render["image"]
        alpha = render["alpha"].detach().cpu().numpy().squeeze()
        depth = render["depth"].detach().cpu().numpy().squeeze()
    elif mesh:
        renderer = ModelRendererOffscreen(camK, H, W)
        pose = np.eye(4)
        pose[:3, :3] = R
        pose[:3, 3] = T
        rgb, depth = renderer.render(pose, mesh)
        alpha = (depth > 0).astype(float)

    if debug_vis:
        plt.imshow(alpha, cmap="rainbow")
        plt.title("Alpha")
        plt.show()
        plt.imshow(depth, cmap="rainbow")
        plt.title("Depth")
        plt.show()

    alpha[alpha < alpha_threshold] = 0
    depth[alpha < alpha_threshold] = 0

    if debug_vis:
        plt.imshow(alpha, cmap="rainbow")
        plt.title(f"Filtered Alpha (Alpha > {alpha_threshold})")
        plt.show()
        plt.imshow(depth, cmap="rainbow")
        plt.title(f"Filtered Depth (Alpha > {alpha_threshold})")
        plt.show()

    safe_region = (
        # Depth is constant within a margin
        get_boolean_mask(depth, depth_margin, depth_change)
        # Alpha is sufficiently high
        & (alpha > alpha_threshold)
        # Alpha is constant within a margin
        & get_boolean_mask(alpha, alpha_margin, alpha_change)
    )

    sampled_pixels = sample_safe_zone(safe_region, min_pixel_distance)

    if debug_vis:
        plt.imshow(alpha > alpha_threshold, cmap="gray")
        plt.imshow(safe_region, alpha=0.5, cmap="Reds")
        plt.scatter(sampled_pixels[:, 0], sampled_pixels[:, 1], c="r", s=1)
        plt.show()

        plt.imshow(rgb)
        plt.scatter(sampled_pixels[:, 0], sampled_pixels[:, 1], c="r", s=1)
        plt.show()

    depth_aligner = DepthInformedPixelToGaussianAligner(
        T=T,
        R=R,
        K=camK,
        depth_map=depth,
    )

    intersections = depth_aligner.align(sampled_pixels)

    if debug_vis:
        intersections_pointcloud = get_points_pointcloud(intersections)
        if object:
            object_pointcloud = get_gaussian_splat_pointcloud(object)
        if mesh:
            object_pointcloud = get_points_pointcloud(
                mesh.vertices, color=np.array([1, 0, 0])
            )

        plot_pointclouds(
            {"Object": object_pointcloud, "Intersections": intersections_pointcloud},
            "Aligned Pixels in Gaussian Object",
        )

    posed_intersections = apply_pose_to_points(intersections, R, T)
    projected_intersections = render_points_in_2d(posed_intersections, camK)

    if debug_vis:
        plt.imshow(alpha > alpha_threshold, cmap="gray")
        plt.imshow(safe_region, alpha=0.5, cmap="Reds")
        plt.scatter(
            projected_intersections[:, 0], projected_intersections[:, 1], c="r", s=1
        )
        plt.show()

    # Turn intersections_projected from N x 2 to N x 3 by prepending with frame_idx
    return intersections, np.concatenate(
        [
            frame_idx * np.ones((len(projected_intersections), 1)),
            projected_intersections,
        ],
        axis=1,
    )


class CanonicalPointSampler:
    def __init__(
        self,
        min_pixel_distance: int = 15,
        threshold: float = 0.9,
        alpha_margin: int = 10,
        max_alpha_change: float = 0.01,
        depth_margin: int = 6,
        depth_change_threshold: float = 0.01,
    ):
        """
        Args:
            min_pixel_distance (int): Minimum distance between points in pixels.
            threshold (float): Minimum alpha value for a point to be considered.
            alpha_margin (int): Only consider points that are at this many pixels away from "edges" in the alpha channel.
            max_alpha_change (float): Threshold for alpha change to consider an edge in the alpha channel.
            depth_margin (int): Only consider points that are at this many pixels away from "edges" in the depth channel.
            depth_change_threshold (float): Threshold for depth change to consider an edge in the depth channel.
        """
        self.min_pixel_distance = min_pixel_distance
        self.threshold = threshold
        self.alpha_margin = alpha_margin
        self.max_alpha_change = max_alpha_change
        self.depth_margin = depth_margin
        self.depth_change_threshold = depth_change_threshold

    def select_safe_pixels(self, alpha: np.ndarray, depth: np.ndarray) -> np.ndarray:
        """
        Select pixels based on alpha and depth channels.
        Args:
            alpha (np.ndarray): Alpha channel. Attention: Will be modified in place.
            depth (np.ndarray): Depth channel. Attention: Will be modified in place.
        Returns:
            np.ndarray: Selected pixel locations (N, 2).
        """
        alpha[alpha < self.threshold] = 0
        depth[alpha < self.threshold] = 0

        safe_region = (
            # Depth is constant within a margin
            get_boolean_mask(depth, self.depth_margin, self.depth_change_threshold)
            # Alpha is sufficiently high
            & (alpha > self.threshold)
            # Alpha is constant within a margin
            & get_boolean_mask(alpha, self.alpha_margin, self.max_alpha_change)
        )

        sampled_pixels = sample_safe_zone(safe_region, self.min_pixel_distance)

        return sampled_pixels
    
    @staticmethod
    def get_3d_locations(sampled_pixels: np.ndarray, depth: np.ndarray, pose: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        Get 3D locations of the selected pixels.
        Args:
            sampled_pixels (np.ndarray): Selected pixel locations (N, 2).
            depth (np.ndarray): Depth channel.
            pose (np.ndarray): Pose of the object.
            K (np.ndarray): Camera intrinsic matrix.
        Returns:
            np.ndarray: 3D locations of the selected pixels (N, 3).
        """
        # TODO: Should use camera intrinsics to get 3D location of ray origin instead of assuming it is at (0, 0, 0)
        # TODO: Should batch this operation
        intersections = []
        for p in sampled_pixels:
            pixel_ray_dir = pixel_to_ray_dir(p, K)
            ray_origin, ray_direction = revert_pose_to_ray(
                np.zeros(3), pixel_ray_dir, pose[:3, :3], pose[:3, 3]
            )
            depth_value = depth[p[1], p[0]]
            t = depth_value / pixel_ray_dir[2]
            point_3d = ray_origin + t * ray_direction
            intersections.append(point_3d)
        
        intersections = np.array(intersections)

        return intersections
        
    
    def __call__(self, rgb: np.ndarray, alpha: np.ndarray, depth: np.ndarray, pose: np.ndarray, K: np.ndarray) -> np.ndarray:
        """
        Select points based on alpha and depth channels.
        Args:
            rgb (np.ndarray): RGB image.
            alpha (np.ndarray): Alpha channel. Attention: Will be modified in place.
            depth (np.ndarray): Depth channel. Attention: Will be modified in place.
            pose (np.ndarray): Pose of the object.
            K (np.ndarray): Camera intrinsic matrix.
        Returns:
            np.ndarray: Selected pixel locations (N, 2).
        """
        pixel_locations = self.select_safe_pixels(alpha, depth)
        points_3d = self.get_3d_locations(pixel_locations, depth, pose, K)
        
        return points_3d