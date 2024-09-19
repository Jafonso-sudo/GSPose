import cv2
import numpy as np
import tqdm
from gaussian_object.gaussian_model import GaussianModel
from pytorch3d.renderer import (
    HeterogeneousRayBundle,
    ray_bundle_to_ray_points,
    RayBundle,
    TexturesAtlas,
    TexturesVertex,
)

from posingpixels.utils.geometry import pixel_to_ray_dir, ray_splat_intersection, revert_pose_to_ray

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
    
    def __init__(self, mask_path: str, pixeltracker_path: str, gaussian_object: GaussianModel, initial_T: np.ndarray, initial_R: np.ndarray, initial_cam_K: np.ndarray, pixeltracker_upscale: float = 1.0):
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
        self.mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) > 0
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
        
        self.gaussian_object.initialize_pose() # IMPORTANT: Otherwise the object will be rendered with the previous pose applied
        
        frame = self.tracks[0, :, :2]
        
        intersections = []
        filtered_idx = []
        
        # Pixel to camera ray (assuming the camera is at the origin and rotated to identity)
        def pixel_to_camera_ray(pixel, camK):
            p_pixel = np.array([pixel[0], pixel[1], 1])
            p_camera = np.linalg.inv(camK) @ p_pixel
            d_ray = p_camera / np.linalg.norm(p_camera)
            return d_ray

        def apply_pose_to_ray(ray, pose):
            R = pose[:3, :3]
            t = pose[:3, 3]
            ray = R.T @ ray
            t = -R.T @ t
            return ray, t
        
        for i, p in tqdm.tqdm(enumerate(frame), total=frame.shape[0]):
            # pixel_ray_dir = pixel_to_ray_dir(p, self.init_K)
            # ray_origin, ray_direction = revert_pose_to_ray(np.zeros(3), pixel_ray_dir, self.init_R, self.init_T)
            
            # intersection_t = ray_splat_intersection(ray_origin, ray_direction, self.gaussian_object)
            # if intersection_t is not None:
            #     intersection_point = ray_origin + intersection_t * ray_direction
            #     intersections.append(intersection_point)
            # else:
            #     filtered_idx.append(i)
            
            og_ray_direction = pixel_to_camera_ray(p[:2], self.init_K)
            init_RT = np.eye(4)
            init_RT[:3, :3] = self.init_R
            init_RT[:3, 3] = self.init_T
            ray_direction, ray_origin = apply_pose_to_ray(og_ray_direction, init_RT)
            
            intersection, intersected_point = ray_splat_intersection(ray_origin, ray_direction, self.gaussian_object)
            if intersection is not None:
                intersections.append(intersection)
            else:
                filtered_idx.append(i)
                
        # Filter out points that did not intersect with the Gaussian object
        filtered_tracks = np.delete(self.tracks, filtered_idx, axis=1)
        
        return intersections, filtered_tracks