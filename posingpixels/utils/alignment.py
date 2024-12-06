from matplotlib import pyplot as plt
import numpy as np
from scipy.ndimage import maximum_filter, minimum_filter
from scipy.spatial import distance



def get_boolean_mask(image: np.ndarray, m: int, v: float) -> np.ndarray:
    """
    Given an image (2D depth map), margin m (in pixels), and threshold value v,
    returns a boolean array where True indicates pixels that have no neighbors
    in the m-radius that differ by more than v.

    Args:
    - image (np.ndarray): 2D array representing the depth map.
    - m (int): Radius margin in pixels.
    - v (float): Threshold value for pixel comparison.

    Returns:
    - np.ndarray: Boolean array with the same shape as image.
    """
    # Create a local maximum filter with a window size of (2m + 1)
    local_max = maximum_filter(image, size=2 * m + 1, mode="reflect")

    # Create a local minimum filter with a window size of (2m + 1)
    local_min = minimum_filter(image, size=2 * m + 1, mode="reflect")

    # Compare the difference between the local max and local min to the threshold v
    diff = local_max - local_min

    # Create a boolean mask where the difference is less than or equal to v
    mask = diff <= v

    return mask


def sample_safe_zone(safe_zone: np.ndarray, min_dist: int) -> np.ndarray:
    """
    Sample points from the 'safe zone' such that any two sampled points are at least
    `min_dist` pixels away from each other.

    Args:
    - safe_zone (np.ndarray): Boolean array where True represents the safe zone.
    - min_dist (int): Minimum distance between any two sampled points (in pixels).

    Returns:
    - np.ndarray: Array of shape (N, 2) where N is the number of sampled points.
    """
    # Get indices of all pixels in the safe zone
    safe_points = np.argwhere(safe_zone)

    # List to store selected points
    selected_points = []

    # Create a function to check distance between a point and all selected points
    def is_far_enough(point, selected_points, min_dist):
        if not selected_points:
            return True
        dists = distance.cdist([point], selected_points)
        return np.all(dists >= min_dist)

    # Iterate through safe points and select points that are far enough apart
    for point in safe_points:
        if is_far_enough(point, selected_points, min_dist):
            selected_points.append(point)

    selected_np = np.array(selected_points)
    
    # DEBUG VIS
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.imshow(safe_zone)
    # ax.scatter(selected_np[:, 1], selected_np[:, 0], c='b', s=5)
    # resulting_figure = plt.gcf()
    
    
    # Make point[0] as y and point[1] as x
    return np.array([selected_np[:, 1], selected_np[:, 0]]).T
