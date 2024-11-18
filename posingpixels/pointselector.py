from abc import ABC, abstractmethod
from typing import Optional
import torch


class PointSelectorStrategy(ABC):
    def __init__(
        self, N: int, query_to_point_mapping: torch.Tensor, view_sizes: torch.Tensor
    ):
        """
        Initialize the PointSelectorStrategy.

        Args:
            Q (int): Number of queries
            N (int): Number of points
            query_to_point_mapping (torch.Tensor): Shape Qx2, view index and point index for each query
            view_sizes (torch.Tensor): Shape V, number of queries in each view
        """
        self.N = N
        self.Q = query_to_point_mapping.shape[0]
        self.V = view_sizes.shape[0]

        self.query_to_point_mapping = query_to_point_mapping
        self.view_sizes = view_sizes

    @abstractmethod
    def query_to_point(
        self,
        coordinates: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
        metric: Optional[torch.Tensor] = None,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
        """
        Transform query predictions into point predictions.

        Args:
            coordinates (torch.Tensor): Shape TxQx2, coordinates of queries
            visibility (torch.Tensor): Shape TxQ, visibility scores
            confidence (torch.Tensor): Shape TxQ, confidence scores
            metric (torch.Tensor): Shape TxQ, custom metric to select points

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: Point predictions
                - coordinates: Shape TxNx2
                - visibility: Shape TxN
                - confidence: Shape TxN
                - indexes: Shape Q
        """
        # Implementation details would go here
        raise NotImplementedError()

    def point_to_query(
        self,
        coordinates: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor]":
        """
        Transform point predictions into query predictions.

        Args:
            coordinates (torch.Tensor): Shape TxNx2, coordinates of points
            visibility (torch.Tensor): Shape TxN, visibility scores
            confidence (torch.Tensor): Shape TxN, confidence scores

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Query predictions
                - coordinates: Shape TxQx2
                - visibility: Shape TxQ
                - confidence: Shape TxQ
        """
        T = coordinates.shape[0]

        point_indices = self.query_to_point_mapping[..., 1].flatten()
        time_indices = torch.arange(T, device=coordinates.device).unsqueeze(1)

        return (
            coordinates[time_indices, point_indices],
            visibility[time_indices, point_indices],
            confidence[time_indices, point_indices],
        )


class SelectMostConfidentPoint(PointSelectorStrategy):
    """
    SelectMostConfidentPoint is a strategy that selects the query with the highest
        confidence for each point in isolation at each timestamp.
    """

    def query_to_point(
        self,
        coordinates: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
        metric: Optional[torch.Tensor] = None,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
        T = coordinates.shape[0]

        best_query_indices = get_most_confident_queries(
            metric if metric is not None else confidence,
            self.query_to_point_mapping[..., 1],
            self.N,
        )
        time_indices = torch.arange(T, device=coordinates.device).unsqueeze(1)
        point_coords = coordinates[time_indices, best_query_indices]
        point_vis = visibility[time_indices, best_query_indices]
        point_conf = confidence[time_indices, best_query_indices]

        return point_coords, point_vis, point_conf, best_query_indices


class SelectMostConfidentView(PointSelectorStrategy):
    def query_to_point(
        self,
        coordinates: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
        metric: Optional[torch.Tensor] = None,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]":
        T = coordinates.shape[0]

        _, _, best_query_indices = get_most_confident_views(
            metric if metric is not None else confidence,
            self.query_to_point_mapping[..., 1],
            self.N,
            self.view_sizes,
        )

        time_indices = torch.arange(T, device=coordinates.device).unsqueeze(1)
        point_coords = coordinates[time_indices, best_query_indices]
        point_vis = visibility[time_indices, best_query_indices]
        point_conf = confidence[time_indices, best_query_indices]

        return point_coords, point_vis, point_conf, best_query_indices


def get_most_confident_queries(
    pred_confidence: torch.Tensor, query_to_point: torch.Tensor, num_points: int
) -> torch.Tensor:
    """
    Find the query indices with highest confidence scores for each point at each timestamp.

    Args:
        pred_confidence: Tensor of shape (T, Q) containing confidence scores for each query at each timestamp
        query_to_point: Tensor of shape (Q,) mapping each query index to its corresponding point index
        num_points: Number of unique points (N)

    Returns:
        Tensor of shape (T, N) containing the query indices with highest confidence for each point at each timestamp
    """
    T, Q = pred_confidence.shape

    # Create a mask of shape (Q, N) where mask[q, p] = 1 if query q corresponds to point p
    query_point_mask = torch.zeros(Q, num_points, device=pred_confidence.device)
    query_indices = torch.arange(Q, device=pred_confidence.device)
    query_point_mask[query_indices, query_to_point] = 1

    # Reshape confidence scores to (T, Q, 1) for broadcasting
    confidence_expanded = pred_confidence.unsqueeze(-1)  # Shape: (T, Q, 1)

    # Broadcast confidence scores across points and mask out irrelevant queries
    # Shape: (T, Q, N)
    masked_confidence = confidence_expanded * query_point_mask.unsqueeze(0)

    # Find the query indices with maximum confidence for each point
    # Use a very low value for masking to ensure masked values aren't selected
    masked_confidence[masked_confidence == 0] = float("-inf")

    # Get the query indices with maximum confidence for each point at each timestamp
    # Shape: (T, N)
    best_query_indices = torch.argmax(masked_confidence, dim=1)

    return best_query_indices


def get_most_confident_views(
    pred_confidence: torch.Tensor,
    query_to_point: torch.Tensor,
    num_points: int,
    queries_lengths: torch.Tensor,
) -> torch.Tensor:
    """
    Find the query indices with highest confidence scores for each point at each timestamp.

    Args:
        pred_confidence: Tensor of shape (T, Q) containing confidence scores for each query at each timestamp
        query_to_point: Tensor of shape (Q,) mapping each query index to its corresponding point index
        num_points: Number of unique points (N)
        queries_lengths: List of lengths of each query sequence

    Returns:
        Tensor of shape (T, N) containing the query indices with highest confidence for each point at each timestamp
    """
    T, Q = pred_confidence.shape
    V = len(queries_lengths)

    query_to_view = torch.repeat_interleave(
        torch.arange(len(queries_lengths), device=pred_confidence.device),
        torch.tensor(queries_lengths),
    )

    # Step 1: Find the most confident view for each timestamp
    # Create a mask of shape (Q, V) where mask[q, v] = 1 if query q belongs to view v
    query_view_mask = torch.zeros(Q, V, device=pred_confidence.device).int()
    query_indices = torch.arange(Q, device=pred_confidence.device).int()
    query_view_mask[query_indices, query_to_view] = 1

    # Calculate average confidence per view at each timestamp
    # Shape: (T, Q, 1)
    confidence_expanded = pred_confidence.unsqueeze(-1)
    # Shape: (T, Q, V)
    view_confidence = confidence_expanded * query_view_mask.unsqueeze(0)

    # Get the sum and count of confidence scores per view
    view_conf_sum = view_confidence.sum(dim=1)  # Shape: (T, V)
    view_conf_count = query_view_mask.sum(dim=0)  # Shape: (V,)

    # Calculate average confidence per view
    view_conf_avg = view_conf_sum / view_conf_count  # Shape: (T, V)

    # Get the most confident view for each timestamp
    best_view_indices = torch.argmax(view_conf_avg, dim=1)  # Shape: (T,)

    # Step 2: Create a mask for queries belonging to the best view at each timestamp
    best_view_mask = query_view_mask[:, best_view_indices]  # Shape: (1, T, Q)

    # Step 3: Find the most confident queries for points in the best view
    # Create a mask for queries belonging to each point
    query_point_mask = torch.zeros(Q, num_points, device=pred_confidence.device)
    query_point_mask[query_indices, query_to_point] = 1

    # Mask confidence scores for queries not in the best view
    masked_confidence = pred_confidence.unsqueeze(-1) * query_point_mask.unsqueeze(
        0
    )  # Shape: (T, Q, N)
    masked_confidence = masked_confidence * best_view_mask.T.unsqueeze(-1)

    # Find the query indices with maximum confidence for each point
    masked_confidence[masked_confidence == 0] = float("-inf")
    best_query_indices = torch.argmax(masked_confidence, dim=1)  # Shape: (T, N)

    return best_view_indices, best_view_mask, best_query_indices
