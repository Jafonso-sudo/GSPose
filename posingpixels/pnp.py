from abc import ABC, abstractmethod
from typing import Generic, List, Literal, Optional, Tuple, TypeVar
from attr import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from misc_utils.warmup_lr import CosineAnnealingWarmupRestarts
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
from pytorch3d.ops.perspective_n_points import efficient_pnp
import numpy as np
import cv2

# Define a type variable for the additional return type
T_Extra = TypeVar("T_Extra")


class PnPSolver(Generic[T_Extra], ABC):
    def __init__(
        self,
        X: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
    ):
        self.X = X
        self.K = K
        self.R = R
        self.T = T

    @abstractmethod
    def __call__(
        self,
        Y: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        X: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, T_Extra]:
        """
        Solve the PnP problem.

        Args:
            Y (torch.Tensor): Shape BxNx2, 2D point estimates
            weights (torch.Tensor, optional): Shape BxN, weights for each point
            X (torch.Tensor, optional): Shape Nx3, 3D canonical points
            K (torch.Tensor, optional): Shape 3x3 or Bx3x3, camera intrinsics
            R (torch.Tensor, optional): Shape Bx3x3, initial rotation matrices
            T (torch.Tensor, optional): Shape Bx3, initial translation vectors

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - rotation_matrices: Shape Bx3x3
                - translation_vectors: Shape Bx3
        """
        raise NotImplementedError()

    def _input_parser(
        self,
        X: Optional[torch.Tensor],
        Y: torch.Tensor,
        K: Optional[torch.Tensor],
        weights: Optional[torch.Tensor],
        R: Optional[torch.Tensor],
        T: Optional[torch.Tensor],
    ):
        X = self.X if X is None else X
        K = self.K if K is None else K
        R = self.R if R is None else R
        T = self.T if T is None else T
        assert X is not None, "3D points X must be provided"
        assert K is not None, "Camera intrinsics K must be provided"

        if len(Y.shape) == 2:
            Y = Y[..., :2].unsqueeze(0)
        B, Q, _ = Y.shape
        if len(X.shape) == 2:
            X = X.unsqueeze(0)
        if weights is not None and len(weights.shape) == 1:
            weights = weights.unsqueeze(0)
        if len(K.shape) == 2:
            K = K[:3, :3].unsqueeze(0)

        return X, Y[..., :2], K[:3, :3], weights, R, T, B, Q

    @staticmethod
    def pose_points(points, R, T):
        return torch.matmul(points, R.transpose(1, 2)) + T.unsqueeze(1)

    @staticmethod
    def render_points(points, K):
        return torch.matmul(points, K.transpose(1, 2)) / points[:, :, 2:3]

    @staticmethod
    def pose_and_render_points(points, R, T, K):
        return PnPSolver.render_points(PnPSolver.pose_points(points, R, T), K)

class OpenCVePnP(PnPSolver[None]):
    """
    OpenCVePnP class that solves the Perspective-n-Point (PnP) problem using the built-in OpenCV ePnP implementation.
    """

    def __init__(
        self,
        flags: int = cv2.SOLVEPNP_EPNP,
        use_ransac: bool = True,
        weight_threshold: float = 0.9,
        ransac_inliner_threshold: float = 2.0,
        ransac_iterations: int = 500,
        min_inliers: int = 20,
        loss_fn: nn.Module = nn.L1Loss(reduction="none"),
        X: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
    ):
        super().__init__(X, K, R, T)
        self.flags = flags
        self.use_ransac = use_ransac
        self.weight_threshold = weight_threshold
        self.min_inliers = min_inliers
        self.ransac_inliner_threshold = ransac_inliner_threshold
        self.ransac_iterations = ransac_iterations
        self.loss_fn = loss_fn

    def __call__(
        self,
        Y: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        X: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
    ) -> "tuple[torch.Tensor, torch.Tensor, None]":
        X, Y, K, weights, R, T, B, Q = self._input_parser(X, Y, K, weights, R, T)

        x = X[0].cpu().numpy()
        K_np = K[0].cpu().numpy()
        Y_np = Y.cpu().numpy()
        weights_np = weights.cpu().numpy() if weights is not None else None

        all_R = []
        all_T = []
        errors = []

        for i in range(B):
            y = Y_np[i]
            x_i = x
            if weights_np is not None:
                w = weights_np[i]
                above_threshold = w > self.weight_threshold
                num_taken = np.sum(above_threshold)
                if num_taken < self.min_inliers:
                    sorted_indices = np.argsort(w, axis=0)[::-1]
                    x_i = x[sorted_indices[: self.min_inliers]]
                    y = y[sorted_indices[: self.min_inliers]]
                else:
                    x_i = x[above_threshold]
                    y = y[above_threshold]

            _, rvec, tvec, err = cv2.solvePnPRansac(
                x_i,
                y,
                K_np,
                None,
                flags=cv2.SOLVEPNP_EPNP,
                reprojectionError=self.ransac_inliner_threshold,
                iterationsCount=self.ransac_iterations,
                confidence=0.9999
            )
            # _, rvec, tvec = cv2.solvePnP(x_i, y, K_np, distCoeffs=None, flags=cv2.SOLVEPNP_EPNP)
            # _, rvec, tvec = self.method(x, y, K_np, distCoeffs=None, flags=self.flags)
            # rvec, tvec = cv2.solvePnPRefineLM(x, y, K_np, None, rvec, tvec)
            R_np = cv2.Rodrigues(rvec)[0]
            T_np = tvec.squeeze()
            all_R.append(torch.tensor(R_np, device=Y.device, dtype=torch.float32))
            all_T.append(torch.tensor(T_np, device=Y.device, dtype=torch.float32))
            
            weights_i = weights[i] if weights is not None else torch.ones(Q, device=Y.device)
            
            rendered_prediction = self.pose_and_render_points(X, all_R[-1].unsqueeze(0), all_T[-1].unsqueeze(0), K)[0, :, :2]
            errors.append(torch.mean(weights_i.unsqueeze(-1) * self.loss_fn(rendered_prediction, Y[i])))
        
        prev_R, prev_T = None, None
        for i in range(B):
            if prev_R is not None and prev_T is not None:
                weights_i = weights[i] if weights is not None else torch.ones(Q, device=Y.device)
                rendered_with_prev = self.pose_and_render_points(X, prev_R.unsqueeze(0), prev_T.unsqueeze(0), K)[0, :, :2]
                error_with_prev = torch.mean(weights_i.unsqueeze(-1) * self.loss_fn(rendered_with_prev, Y[i]))
                if error_with_prev < errors[i]:
                    all_R[i], all_T[i] = prev_R, prev_T
                    errors[i] = error_with_prev
                
            prev_R, prev_T = all_R[i], all_T[i]
        
        post_R, post_T = None, None
        for i in range(B - 1, -1, -1):
            if post_R is not None and post_T is not None:
                weights_i = weights[i] if weights is not None else torch.ones(Q, device=Y.device)
                rendered_with_post = self.pose_and_render_points(X, post_R.unsqueeze(0), post_T.unsqueeze(0), K)[0, :, :2]
                error_with_post = torch.mean(weights_i.unsqueeze(-1) * self.loss_fn(rendered_with_post, Y[i]))
                if error_with_post < errors[i]:
                    all_R[i], all_T[i] = post_R, post_T
                    errors[i] = error_with_post
                
            post_R, post_T = all_R[i], all_T[i]
            
        
        return torch.stack(all_R), torch.stack(all_T), torch.stack(errors)


@dataclass
class GradientResults:
    losses: torch.Tensor
    rotations: torch.Tensor
    translations: torch.Tensor
    learning_rates: torch.Tensor


class GradientPnP(PnPSolver[GradientResults]):
    def __init__(
        self,
        epochs: int = 4000,
        warmup_epochs: int = 100,
        max_lr: float = 0.02,
        min_lr: float = 0.0,
        early_stop_epochs: int = 100,
        early_stop_loss_grad_norm: float = 1e-4,
        temporal_consistency_weight: float = 0.2,
        reconstruction_loss: Literal["mse", "l1", "huber"] = "huber",
        reconstruction_loss_clip: Optional[float] = None,
        reconstruction_loss_clip_start_epoch: int = 200,
        X: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
        disable_tqdm: bool = False,
    ):
        super().__init__(X, K, R, T)
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.early_stop_epochs = max(early_stop_epochs, warmup_epochs)
        self.early_stop_loss_grad_norm = early_stop_loss_grad_norm
        self.temporal_consistency_weight = temporal_consistency_weight
        self.reconstruction_loss_type = reconstruction_loss
        self.reconstruction_loss = self._get_reconstruction_loss(reconstruction_loss)
        self.reconstruction_loss_clip = reconstruction_loss_clip
        self.reconstruction_loss_clip_start_epoch = reconstruction_loss_clip_start_epoch
        self.disable_tqdm = disable_tqdm

    @staticmethod
    def _get_reconstruction_loss(
        reconstruction_loss_type: Literal["mse", "l1", "huber"],
    ):
        if reconstruction_loss_type == "mse":
            return nn.MSELoss(reduction="none")
        elif reconstruction_loss_type == "l1":
            return nn.L1Loss(reduction="none")
        elif reconstruction_loss_type == "huber":
            return nn.HuberLoss(reduction="none", delta=0.1)
        else:
            raise ValueError(f"Unknown reconstruction loss: {reconstruction_loss_type}")

    @staticmethod
    def _initialize_parameters(R, T, B):
        if R is None:
            R_params = nn.Parameter(
                torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], device=R.device).repeat(
                    B, 1
                )
            )
        else:
            if not torch.is_tensor(R):
                R = torch.tensor(R, device=R.device)
            if R.dim() == 2:
                R = R.unsqueeze(0).repeat(B, 1, 1)
            R_params = nn.Parameter(matrix_to_rotation_6d(R))
        if T is None:
            T_params = nn.Parameter(torch.zeros(B, 3, device=R.device))
        else:
            if not torch.is_tensor(T):
                T = torch.tensor(T, device=R.device)
            if T.dim() == 1:
                T = T.unsqueeze(0).repeat(B, 1)
            T_params = nn.Parameter(T)
        return R_params, T_params

    @staticmethod
    def _initialize_result_logger(B, epochs, device):
        return GradientResults(
            losses=torch.zeros(epochs, device=device),
            rotations=torch.zeros(epochs, B, 6, device=device),
            translations=torch.zeros(epochs, B, 3, device=device),
            learning_rates=torch.zeros(epochs, device=device),
        )

    @staticmethod
    def _temporal_consistency_loss(rotations_6d, translations):
        rotation_diff = rotations_6d[1:] - rotations_6d[:-1]
        translation_diff = translations[1:] - translations[:-1]

        rotation_loss = torch.norm(rotation_diff, dim=1).mean()
        translation_loss = torch.norm(translation_diff, dim=1).mean()

        return rotation_loss + translation_loss

    def __call__(
        self,
        Y: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        X: Optional[torch.Tensor] = None,
        K: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, GradientResults]:
        X, Y, K, weights, R, T, B, Q = self._input_parser(X, Y, K, weights, R, T)
        if weights is None:
            weights = torch.ones_like(Y)
        elif len(weights.shape) == 2:
            weights = weights.unsqueeze(-1)

        R_params, T_params = self._initialize_parameters(R, T, B)

        optimizer = optim.Adam([R_params, T_params])
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            self.epochs,
            warmup_steps=self.warmup_epochs,
            max_lr=self.max_lr,
            min_lr=self.min_lr,
        )

        results = self._initialize_result_logger(B, self.epochs, Y.device)

        for epoch in tqdm.tqdm(range(self.epochs), disable=self.disable_tqdm):
            optimizer.zero_grad()
            results.learning_rates[epoch] = lr_scheduler.get_lr()[0]
            rendered_points = self.pose_and_render_points(
                X, rotation_6d_to_matrix(R_params), T_params, K
            )

            reconstruction_loss = (
                self.reconstruction_loss(rendered_points[:, :, :2], Y) * weights
            )
            if (
                self.reconstruction_loss_clip is not None
                and epoch > self.reconstruction_loss_clip_start_epoch
            ):
                reconstruction_loss = torch.clamp(
                    reconstruction_loss, max=self.reconstruction_loss_clip
                )
            reconstruction_loss = torch.mean(reconstruction_loss)
            consistency_loss = self._temporal_consistency_loss(R_params, T_params)

            total_loss = (
                reconstruction_loss
                + self.temporal_consistency_weight * consistency_loss
            )

            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            results.losses[epoch] = total_loss.item()
            results.rotations[epoch] = R_params.detach()
            results.translations[epoch] = T_params.detach()
            if epoch > self.early_stop_epochs:
                loss_grads = results.losses[1:] - results.losses[:-1]
                loss_grad = loss_grads[-self.early_stop_epochs : epoch].abs().mean()
                if loss_grad < self.early_stop_loss_grad_norm:
                    break

        return rotation_6d_to_matrix(R_params.detach()), T_params.detach(), results