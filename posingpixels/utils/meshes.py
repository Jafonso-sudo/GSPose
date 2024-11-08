import torch

# Util function for loading meshes

# Data structures and functions for rendering
from scipy.spatial import ConvexHull
import trimesh
import numpy as np


def get_size_from_mesh(mesh: trimesh.Trimesh) -> torch.Tensor:
    """
    Calculate the size of the object from the mesh.
    Args:
        mesh: The mesh of the object.
    Returns:
        The size of the object.
    """
    min_coords = torch.tensor(mesh.bounds[0])
    max_coords = torch.tensor(mesh.bounds[1])
    size = max_coords - min_coords
    return size

def get_bbox_from_size(size: torch.Tensor) -> torch.Tensor:
    """
    Calculate the bounding box of the object from the size.
    Assumes the object is centered at the origin.
    Args:
        size: The size of the object.
    Returns:
        The bounding box of the object.
    """
    ex, ey, ez = size / 2
    obj_bbox3d = torch.tensor([
        [-ex, -ey, -ez],  # Front-top-left corner
        [ex, -ey, -ez],  # Front-top-right corner
        [ex, ey, -ez],  # Front-bottom-right corner
        [-ex, ey, -ez],  # Front-bottom-left corner
        [-ex, -ey, ez],  # Back-top-left corner
        [ex, -ey, ez],  # Back-top-right corner
        [ex, ey, ez],  # Back-bottom-right corner
        [-ex, ey, ez],  # Back-bottom-left corner
    ])
    return obj_bbox3d
def get_diameter_from_mesh(mesh: trimesh.Trimesh) -> float:
    """
    Calculate the diameter of the object from the mesh.
    Args:
        mesh: The mesh of the object.
    Returns:
        The diameter of the object.
    """
    vertices = np.array(mesh.vertices)
    hull = ConvexHull(vertices)

    hull_vertices = vertices[hull.vertices]
    max_distance = 0

    # Compute pairwise differences using broadcasting
    differences = hull_vertices[:, None] - hull_vertices  # Shape: (N, N, 3)

    # Compute the Euclidean distances for each pair
    distances = np.linalg.norm(differences, axis=2)  # Shape: (N, N)

    # Find the maximum distance
    max_distance = np.max(distances)
    return max_distance


def get_size_from_vertices(vertices: torch.Tensor) -> torch.Tensor:
    """
    Calculate the size of the object from the vertices.
    Args:
        vertices: The vertices of the object.
    Returns:
        The size of the object.
    """
    min_coords, _ = torch.min(vertices, dim=0)
    max_coords, _ = torch.max(vertices, dim=0)
    size = max_coords - min_coords
    return size


def get_diameter_from_vertices(vertices: torch.Tensor) -> float:
    """
    Calculate the diameter of the object from the vertices.
    Args:
        vertices: The vertices of the object.
    Returns:
        The diameter of the object.
    """

    hull = ConvexHull(vertices.cpu().numpy())

    hull_vertices = vertices[hull.vertices]
    max_distance = 0

    # Compute pairwise differences using broadcasting
    differences = hull_vertices.unsqueeze(1) - hull_vertices.unsqueeze(
        0
    )  # Shape: (N, N, 3)

    # Compute the Euclidean distances for each pair
    distances = torch.norm(differences, dim=2)  # Shape: (N, N)

    # Find the maximum distance (ignoring the diagonal which is 0)
    max_distance = distances.max().item()
    return max_distance
