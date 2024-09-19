from typing import Optional, Tuple
import numpy as np
from gaussian_object.gaussian_model import GaussianModel

def pixel_to_ray_dir(pixel: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Convert a pixel coordinate to a ray.

    Parameters
    ----------
    pixel : np.ndarray
        The pixel coordinate.
    K : np.ndarray
        The camera matrix.

    Returns
    -------
    np.ndarray
        The normalized ray.
    """
    
    pixel_cord = np.append(pixel, 1)
    ray = np.linalg.inv(K) @ pixel_cord
    return ray / np.linalg.norm(ray)

def pixel_to_ray_batch(pixels: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Convert multiple pixel coordinates to rays.

    Parameters
    ----------
    pixels : np.ndarray
        The pixel coordinates. Shape: (N, 2), where N is the number of pixels.
    K : np.ndarray
        The camera matrix.

    Returns
    -------
    np.ndarray
        The normalized rays. Shape: (N, 3).
    """
    pixels = np.hstack((pixels, np.ones((pixels.shape[0], 1))))
    rays = np.linalg.inv(K) @ pixels.T
    return (rays / np.linalg.norm(rays[:2], axis=0)).T

def apply_pose_to_ray(ray_origin: np.ndarray, ray_direction: np.ndarray, R: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply a pose to a ray.

    Parameters
    ----------
    ray_origin : np.ndarray
        The origin of the ray.
    ray_direction : np.ndarray
        The direction of the ray.
    R : np.ndarray
        The rotation matrix.
    T : np.ndarray
        The translation vector.

    Returns
    -------
    np.ndarray
        The transformed ray origin.
    np.ndarray
        The transformed ray direction.
    """
    return R @ ray_origin + T, R @ ray_direction

def revert_pose_to_ray(ray_origin: np.ndarray, ray_direction: np.ndarray, R: np.ndarray, T: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Revert a pose from a ray.

    Parameters
    ----------
    ray_origin : np.ndarray
        The origin of the ray.
    ray_direction : np.ndarray
        The direction of the ray.
    R : np.ndarray
        The rotation matrix.
    T : np.ndarray
        The translation vector.

    Returns
    -------
    np.ndarray
        The reverted ray origin.
    np.ndarray
        The reverted ray direction.
    """
    return R.T @ (ray_origin - T), R.T @ ray_direction

def apply_pose_to_points(points: np.ndarray, R: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Apply a pose to a set of points.

    Parameters
    ----------
    points : np.ndarray
        The points to transform. Shape: (N, 3), where N is the number of points.
    R : np.ndarray
        The rotation matrix.
    T : np.ndarray
        The translation vector.

    Returns
    -------
    np.ndarray
        The transformed points.
    """
    return R @ points + T

def render_points_in_2d(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    """
    Render 3D points in 2D using a camera matrix.

    Parameters
    ----------
    points : np.ndarray
        The 3D points. Shape: (N, 3), where N is the number of points.
    K : np.ndarray
        The camera matrix.

    Returns
    -------
    np.ndarray
        The rendered 2D points. Shape: (N, 2).
    """
    pixels = K @ points
    return pixels[:2] / pixels[2]

# def ray_sphere_intersection(ray_origin: np.ndarray, ray_direction: np.ndarray, sphere_center: np.ndarray, sphere_radius: float) -> Optional[Tuple[float, float]]:
#     """
#     Compute the intersection of a ray with a sphere.

#     Parameters
#     ----------
#     ray_origin : np.ndarray
#         The origin of the ray.
#     ray_direction : np.ndarray
#         The direction of the ray.
#     sphere_center : np.ndarray
#         The center of the sphere.
#     sphere_radius : float
#         The radius of the sphere.

#     Returns
#     -------
#     Tuple[Optional[float], Optional[float]]
#         A tuple containing the two intersection points distances from the ray origin.
#     """
#     # ray_direction = ray_direction / np.linalg.norm(ray_direction)
#     oc = ray_origin - sphere_center
#     a = np.dot(ray_direction, ray_direction)
#     b = 2.0 * np.dot(oc, ray_direction)
#     c = np.dot(oc, oc) - sphere_radius**2
#     discriminant = b**2 - 4 * a * c
#     if discriminant < 0:
#         return None
    
#     t1 = (-b - np.sqrt(discriminant)) / (2.0 * a)
#     t2 = (-b + np.sqrt(discriminant)) / (2.0 * a)
    
#     return t1, t2

def ray_sphere_intersection(ray_origin, ray_direction, sphere_center, sphere_radius):
    if sphere_radius > 0.02:
        sphere_radius = 0.02
    ray_origin = np.array(ray_origin)
    ray_direction = np.array(ray_direction)
    # Direction should be normalized
    ray_direction = ray_direction / np.linalg.norm(ray_direction)
    
    # Vector from the ray origin to the sphere center
    oc = ray_origin - sphere_center
    
    # Coefficients of the quadratic equation
    a = np.dot(ray_direction, ray_direction)
    b = 2.0 * np.dot(oc, ray_direction)
    c = np.dot(oc, oc) - sphere_radius ** 2
    
    # Discriminant
    discriminant = b ** 2 - 4 * a * c
    
    # If the discriminant is negative, the ray does not intersect the ellipsoid
    if discriminant < 0:
        return None

    # Calculate the intersection points
    t1 = (-b + np.sqrt(discriminant)) / (2 * a)
    t2 = (-b - np.sqrt(discriminant)) / (2 * a)

    # Find the minimum positive t value
    t_values = np.array([t1, t2])
    t_positive = t_values[t_values >= 0]

    # If there are no positive t values, the ray does not intersect the ellipsoid
    if t_positive.size == 0:
        return None

    # Use the smallest positive t value
    t = np.min(t_positive)
    
    # Use average of the two t values if it is positive
    if t_avg := np.mean(t_values):
        t = t_avg
    
    # Get the intersection point
    intersection_point = ray_origin + t * ray_direction
    
    return intersection_point, t


# def ray_splat_intersection(ray_origin: np.ndarray, ray_direction: np.ndarray, gaussian_object: GaussianModel) -> Optional[float]:
#     """
#     Calculate the intersection of a ray with a splat represented by a Gaussian model.
#     Args:
#         ray_origin (np.ndarray): The origin of the ray.
#         ray_direction (np.ndarray): The direction of the ray.
#         gaussian_object (GaussianModel): The Gaussian model representing the splat.
#     Returns:
#         Optional[float]: The distance from the ray origin to the intersection point, or None if there is no intersection.
#     """
#     positions = gaussian_object.get_xyz.squeeze().detach().cpu().numpy()
#     rotations = gaussian_object.get_rotation.squeeze().detach().cpu().numpy()
#     scalings = gaussian_object.get_scaling.squeeze().detach().cpu().numpy()
    
#     t = None
#     for position, rotation, scaling in zip(positions, rotations, scalings):
#         max_scaling = min(np.max(scaling), 0.02)
#         intersection = ray_sphere_intersection(ray_origin, ray_direction, position, max_scaling)
#         if intersection is None:
#             continue
        
#         proposal_t = np.mean(intersection)
#         if proposal_t < 0:
#             continue
        
#         if t is None or proposal_t < t:
#             t = proposal_t
    
#     return float(t) if t is not None else None

def ray_splat_intersection(ray_origin, ray_direction, obj_gaussians):
    # Get the Gaussian points and features
    gaussian_points = obj_gaussians.get_xyz.squeeze().detach().cpu().numpy()
    gaussian_rotations = obj_gaussians.get_rotation.squeeze().detach().cpu().numpy()
    gaussian_scalings = obj_gaussians.get_scaling.squeeze().detach().cpu().numpy()
    
    intersection, t_min, intersected_point = None, None, None
    for position, rotation, scaling in zip(gaussian_points, gaussian_rotations, gaussian_scalings):
        intersection_t = ray_sphere_intersection(ray_origin, ray_direction, position, np.max(scaling))
        if intersection_t is not None:
            intersection_point, t = intersection_t
            if t_min is None or t < t_min:
                t_min = t
                intersection = intersection_point
                intersected_point = position
        
    return intersection, intersected_point
    
    

def ray_sphere_intersection_batch(ray_origins: np.ndarray, ray_directions: np.ndarray, sphere_centers: np.ndarray, sphere_radii: np.ndarray) -> np.ndarray:
    """
    Compute the intersection of multiple rays with multiple spheres.

    Parameters
    ----------
    ray_origins : np.ndarray
        The origins of the rays. Shape: (N, 3), where N is the number of rays.
    ray_directions : np.ndarray
        The directions of the rays. Shape: (N, 3).
    sphere_centers : np.ndarray
        The centers of the spheres. Shape: (M, 3), where M is the number of spheres.
    sphere_radii : np.ndarray
        The radii of the spheres. Shape: (M,).

    Returns
    -------
    np.ndarray
        A 3D array of shape (N, M, 2), containing the two intersection distances for each ray-sphere pair.
        If no intersection, NaN is returned in place of the distances.
    """
    # Broadcast arrays to make sure they have the right shape
    ray_origins = ray_origins[:, np.newaxis, :]  # (N, 1, 3)
    ray_directions = ray_directions[:, np.newaxis, :]  # (N, 1, 3)
    sphere_centers = sphere_centers[np.newaxis, :, :]  # (1, M, 3)
    sphere_radii = sphere_radii[np.newaxis, :]  # (1, M)

    # Vectorized computation of oc, a, b, and c for the quadratic formula
    oc = ray_origins - sphere_centers  # (N, M, 3)
    a = np.sum(ray_directions**2, axis=-1)  # (N, 1)
    b = 2.0 * np.sum(oc * ray_directions, axis=-1)  # (N, M)
    c = np.sum(oc**2, axis=-1) - sphere_radii**2  # (N, M)
    
    # Discriminant
    discriminant = b**2 - 4 * a[:, np.newaxis] * c  # (N, M)
    
    # Initialize result with NaNs (no intersection case)
    t1 = np.full_like(discriminant, np.nan)
    t2 = np.full_like(discriminant, np.nan)
    
    # Compute the two intersection distances where discriminant >= 0
    mask = discriminant >= 0
    sqrt_discriminant = np.sqrt(discriminant[mask])
    a_expanded = a[:, np.newaxis][mask]
    
    t1[mask] = (-b[mask] - sqrt_discriminant) / (2.0 * a_expanded)
    t2[mask] = (-b[mask] + sqrt_discriminant) / (2.0 * a_expanded)
    
    return np.stack((t1, t2), axis=-1)  # (N, M, 2)