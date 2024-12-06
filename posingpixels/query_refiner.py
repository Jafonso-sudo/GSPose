import torch

from posingpixels.pnp import GradientResults, PnPSolver
from posingpixels.pointselector import PointSelectorStrategy
from posingpixels.utils.cotracker import scale_by_crop, unscale_by_crop


class QueryRefiner:
    def __init__(
        self,
        point_selector: PointSelectorStrategy,
        pnp_solver: PnPSolver,
        bboxes: torch.Tensor,
        scaling: torch.Tensor,
        prepend_R: torch.Tensor,
        prepend_T: torch.Tensor,
        threshold: float = 0.6,
        pad_inputs: bool = True,
    ):
        """
        Initialize the QueryRefiner.

        Args:
            point_selector (PointSelectorStrategy): Instance of PointSelectorStrategy
            pnp_solver (PnPSolver): Instance of PnPSolver
            bboxes (torch.Tensor): Shape Bx4, bounding boxes for crop operation
            scaling (torch.Tensor): Shape Bx2, scaling factors
            prepend_R (torch.Tensor): Shape Px3x3, initial rotation matrices
            prepend_T (torch.Tensor): Shape Px3, initial translation vectors
            threshold (float, optional): Threshold for visibility x confidence. Defaults to 0.6.
        """
        self.step = 8
        self.window = 16
        
        self.B = bboxes.shape[0]
        self.point_selector = point_selector
        self.pnp_solver = pnp_solver
        self.bboxes = bboxes
        self.scaling = scaling
        self.threshold = threshold
        
        if pad_inputs:
            self._pad_inputs()

        
        self.prepend_R = prepend_R
        self.prepend_T = prepend_T
        self.prepend_length = prepend_R.shape[0]
        
        
        self.reset()
        
    def _pad_inputs(self):
        real_B = self.B
        # Round B up such that it is a multiple of window
        self.B = (self.B + self.window - 1) // self.window * self.window
        if self.B > real_B:
            self.bboxes = torch.cat(
                [self.bboxes, self.bboxes[-1:].repeat(self.B - real_B, 1)],
                dim=0,
            )
            self.scaling = torch.cat(
                [self.scaling, self.scaling[-1:].repeat(self.B - real_B, 1)],
                dim=0,
            )

    def reset(self):
        self.current = 0
        self.R = (
            torch.eye(3, device=self.bboxes.device).unsqueeze(0).repeat(self.B, 1, 1)
        )
        self.T = torch.zeros(self.B, 3, device=self.bboxes.device)

    def __call__(
        self,
        coordinates: torch.Tensor,
        visibility: torch.Tensor,
        confidence: torch.Tensor,
    ) -> "tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[GradientResults]]":
        """
        Refine query predictions.

        Args:
            coordinates (torch.Tensor): Shape WxQx2, coordinates of queries
            visibility (torch.Tensor): Shape WxQ, visibility scores
            confidence (torch.Tensor): Shape WxQ, confidence scores

        Returns:
            tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Refined predictions
                - coordinates: Shape WxQx2
                - visibility: Shape WxQ
                - confidence: Shape WxQ
        """
        device = coordinates.device
        left, right = self.current, min(self.current + self.window, self.B)
        # Select points from queries to be used for PnP
        coordinates, visibility, confidence, indexes = (
            self.point_selector.query_to_point(coordinates, visibility, confidence)
        )
        
        # Uncrop the coordinates
        uncroped_coordinates = unscale_by_crop(
            coordinates, self.bboxes[left:right], self.scaling[left:right]
        )

        # Solve PnP
        # TODO: If we have ground truths (prepend video), we can skip PnP for them
        # - Weights are calculated as the product of visibility and confidence
        # TODO: Ensure there is a minimum number of points to solve
        weights = visibility * confidence
        weights[weights < self.threshold] = 0
        # - R, T are initialized as the previous predictions (when available)
        if left > 0:
            copy = min(left + self.step, right)
            initial_R = torch.cat(
                [
                    self.R[left : copy],
                    self.R[copy - 1 : copy].repeat(
                        self.window - (copy - left), 1, 1
                    ),
                ],
                dim=0,
            )
            initial_T = torch.cat(
                [
                    self.T[left : copy],
                    self.T[copy - 1 : copy].repeat(
                        self.window - (copy - left), 1
                    ),
                ],
                dim=0,
            )
        else:
            initial_R = torch.eye(3, device=device).unsqueeze(0).repeat(self.window, 1, 1)
            initial_T = torch.zeros(self.window, 3, device=device)
        if self.prepend_length and left < self.prepend_length:
            right_prepend = max(left, min(right, self.prepend_length))
            initial_R[:right_prepend - left] = self.prepend_R[left:right_prepend]
            initial_T[:right_prepend - left] = self.prepend_T[left:right_prepend]
            
        # - Update R, T
        self.R[left:right], self.T[left:right], gradient_results = self.pnp_solver(
            Y=uncroped_coordinates,
            weights=weights,
            R=initial_R,
            T=initial_T,
        )
        
        # Apply pose to points
        coordinates = self.pnp_solver.pose_and_render_points(self.pnp_solver.X, self.R[left:right], self.T[left:right], self.pnp_solver.K)[..., :2]

        # Crop the coordinates
        coordinates = scale_by_crop(coordinates, self.bboxes[left:right], self.scaling[left:right])
        
        # Update visibility and confidence TODO: Update based on reprojection error or something else
        
        # Update query predictions based on the points
        coordinates, visibility, confidence = self.point_selector.point_to_query(coordinates, visibility, confidence)
        
        # Update current
        self.current += self.step
        
        return coordinates, visibility, confidence, gradient_results

