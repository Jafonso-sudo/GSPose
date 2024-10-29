import cv2
import numpy as np
import tqdm
from gaussian_object.gaussian_model import GaussianModel

from posingpixels.utils.geometry import (
    pixel_to_ray_dir,
    ray_splat_intersection,
    revert_pose_to_ray,
)

import matplotlib.pyplot as plt

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
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0.5
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
    object: GaussianModel,
    R: np.ndarray,
    T: np.ndarray,
    camK: np.ndarray,
    H: int,
    W: int,
    alpha_threshold: float = 0.9,
    depth_margin: int = 6,
    depth_change: float = 0.01,
    alpha_margin: int = 15,
    alpha_change: float = 0.01,
    min_pixel_distance: int = 25,
    frame_idx: int = 0,
    debug_vis: bool = False,
):
    render = render_gaussian_model_with_info(object, camK, H, W, R=R, T=T)

    alpha = render["alpha"].detach().cpu().numpy().squeeze()
    depth = render["depth"].detach().cpu().numpy().squeeze()

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

        plt.imshow(render["image"])
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
        object_pointcloud = get_gaussian_splat_pointcloud(object)

        plot_pointclouds(
            {"Object": object_pointcloud, "Intersections": intersections_pointcloud},
            "Aligned Pixels in Gaussian Object",
        )

    posed_intersections = apply_pose_to_points(intersections, R, T)
    projected_intersections = render_points_in_2d(posed_intersections.T, camK)

    if debug_vis:
        plt.imshow(alpha > alpha_threshold, cmap="gray")
        plt.imshow(safe_region, alpha=0.5, cmap="Reds")
        plt.scatter(
            projected_intersections[:, 0], projected_intersections[:, 1], c="r", s=1
        )
        plt.show()

    # Turn intersections_projected from N x 2 to N x 3 by prepending with frame_idx
    return np.concatenate(
        [
            frame_idx * np.ones((len(projected_intersections), 1)),
            projected_intersections,
        ],
        axis=1,
    )
