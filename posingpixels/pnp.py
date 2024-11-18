from abc import ABC, abstractmethod
from typing import Literal, Optional, Tuple
from attr import dataclass
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from misc_utils.warmup_lr import CosineAnnealingWarmupRestarts
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
from pytorch3d.ops.perspective_n_points import efficient_pnp


class PnPSolver(ABC):
    @abstractmethod
    def __call__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        K: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        """
        Solve the PnP problem.

        Args:
            X (torch.Tensor): Shape Nx3, 3D canonical points
            Y (torch.Tensor): Shape BxNx2, 2D point estimates
            K (torch.Tensor): Shape 3x3 or Bx3x3, camera intrinsics
            weights (torch.Tensor, optional): Shape BxN, weights for each point
            R (torch.Tensor, optional): Shape Bx3x3, initial rotation matrices
            T (torch.Tensor, optional): Shape Bx3, initial translation vectors

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - rotation_matrices: Shape Bx3x3
                - translation_vectors: Shape Bx3
        """
        raise NotImplementedError()

    def _input_parser(self, X, Y, K, weights, R, T):
        if len(Y.shape) == 2:
            Y = Y[..., :2].unsqueeze(0)
        B, Q, _ = Y.shape
        if len(X.shape) == 2:
            X = X.unsqueeze(0).repeat(B, 1, 1)
        if weights is not None and len(weights.shape) == 1:
            weights = weights.unsqueeze(0).repeat(B, 1)
        if len(K.shape) == 2:
            K = K[:3, :3].unsqueeze(0).repeat(B, 1, 1)

        return X, Y, K, weights, R, T, B, Q

    @staticmethod
    def pose_points(points, R, T):
        return torch.matmul(points, R.transpose(1, 2)) + T.unsqueeze(1)

    @staticmethod
    def render_points(points, K):
        return torch.matmul(points, K.transpose(1, 2)) / points[:, :, 2:3]

    @staticmethod
    def pose_and_render_points(points, R, T, K):
        return PnPSolver.render_points(PnPSolver.pose_points(points, R, T), K)


class ePnP(PnPSolver):
    """
    ePnP class that solves the Perspective-n-Point (PnP) problem using the built-in PyTorch3D ePnP implementation.
    """

    @abstractmethod
    def __call__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        K: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        X, Y, K, weights, R, T, B, Q = self._input_parser(X, Y, K, weights, R, T)

        Y_h = torch.cat((Y, torch.ones(B, Q, 1, device=Y.device)), dim=-1)
        K_inv = torch.inverse(K).transpose(1, 2)
        Y_uncal = torch.matmul(Y_h, K_inv)

        transform = efficient_pnp(X, Y_uncal[..., :2], weights=weights)

        return transform.R.transpose(1, 2), transform.T


class RANSACePnP(PnPSolver):
    def __init__(self, num_iterations: int = 10, subset_size: int = 6):
        self.num_iterations = num_iterations
        self.subset_size = subset_size

    def __call__(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        K: torch.Tensor,
        weights: Optional[torch.Tensor] = None,
        R: Optional[torch.Tensor] = None,
        T: Optional[torch.Tensor] = None,
    ) -> "tuple[torch.Tensor, torch.Tensor]":
        X, Y, K, weights, R, T, B, Q = self._input_parser(X, Y, K, weights, R, T)

        Y_h = torch.cat((Y, torch.ones(B, Q, 1, device=Y.device)), dim=-1)
        K_inv = torch.inverse(K).transpose(1, 2)
        Y_uncal = torch.matmul(Y_h, K_inv)

        if weights is not None:
            sorted_indices = torch.argsort(weights, descending=True)
            sorted_confidence = torch.gather(weights, 1, sorted_indices)
        else:
            sorted_indices = torch.arange(Q, device=Y.device).unsqueeze(0).repeat(B, 1)
            sorted_confidence = torch.ones_like(sorted_indices, device=Y.device)

        best_rotation = torch.eye(3, device=Y.device).unsqueeze(0).repeat(B, 1, 1)
        best_translation = torch.zeros(3, device=Y.device).unsqueeze(0).repeat(B, 1)
        best_score = -torch.ones(B, device=Y.device) * 1e6

        subset_size = min(self.subset_size, Q)
        for _ in range(self.num_iterations):
            # Weighted random sampling for the subset
            subset_indices = torch.multinomial(
                sorted_confidence, subset_size, replacement=False
            )
            subset_indices = torch.gather(sorted_indices, 1, subset_indices)

            X_subset = torch.gather(X, 1, subset_indices.unsqueeze(-1).repeat(1, 1, 3))
            Y_uncal_subset = torch.gather(
                Y_uncal, 1, subset_indices.unsqueeze(-1).repeat(1, 1, 3)
            )

            transform = efficient_pnp(X_subset, Y_uncal_subset[..., :2])

            X_projected = PnPSolver.pose_and_render_points(
                X, transform.R.transpose(1, 2), transform.T, K
            )[..., :2]
            score = -torch.mean(weights * torch.norm(X_projected - Y, dim=-1), dim=-1)

            improved_mask = score > best_score
            best_rotation[improved_mask] = transform.R[improved_mask]
            best_translation[improved_mask] = transform.T[improved_mask]
            best_score[improved_mask] = score[improved_mask]

        return best_rotation.transpose(1, 2), best_translation


# class PnPSolver(ABC):
#     def __init__(self, canonical_points, K=None, device=None):
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
#         self.canonical_points = nn.Parameter(canonical_points.to(self.device), requires_grad=False)
#         self.K = nn.Parameter(K.to(self.device), requires_grad=False) if K is not None else None

#     @staticmethod
#     def pose_points(points, R, T):
#         """
#         Applies a pose to points.

#         Args:
#             points (torch.Tensor): The points to be transformed, of shape (N, 3).
#             R (torch.Tensor): The rotation matrices, of shape (B, 3, 3).
#             T (torch.Tensor): The translation vectors, of shape (B, 3).

#         Returns:
#             torch.Tensor: The transformed points, of shape (N, 3).
#         """
#         transformed = torch.matmul(points, R.transpose(1, 2))
#         return transformed + T.unsqueeze(1)

#     @staticmethod
#     def render_points(points, K):
#         """
#         Projects points using the camera intrinsics matrix.

#         Args:
#             points (torch.Tensor): The points to be projected, of shape (N, 3).
#             K (torch.Tensor): The camera intrinsics matrix, of shape (B, 3, 3).

#         Returns:
#             torch.Tensor: The projected points, of shape (N, 2).
#         """
#         K = K.unsqueeze(0) if len(K.shape) == 2 else K
#         projected = torch.matmul(points, K.transpose(1, 2))
#         return projected / points[:, :, 2:3]

#     def pose_and_render_points(self, R, T, K=None, points=None):
#         """
#         Applies a pose to points and projects them using the camera intrinsics matrix.

#         Args:
#             R (torch.Tensor): The rotation matrices, of shape (B, 3, 3).
#             T (torch.Tensor): The translation vectors, of shape (B, 3).
#             K (torch.Tensor): The camera intrinsics matrix, of shape (B, 3, 3).
#             points (torch.Tensor): The points to be transformed, of shape (N, 3).

#         Returns:
#             torch.Tensor: The projected points, of shape (N, 2).
#         """
#         if points is None:
#             points = self.canonical_points
#         if K is None:
#             if self.K is None: raise ValueError("Camera intrinsics matrix not provided.")
#             K = self.K
#         return PnPSolver.render_points(PnPSolver.pose_points(points, R, T), K)

#     @abstractmethod
#     def forward(self, R, T, K=None) -> Tuple[torch.Tensor, torch.Tensor]:
#         """
#         Forward pass of the PnP solver.

#         Args:
#             R (torch.Tensor): The rotation matrices, of shape (B, 3, 3).
#             T (torch.Tensor): The translation vectors, of shape (B, 3).
#             K (torch.Tensor): The camera intrinsics matrix, of shape (B, 3, 3).

#         Returns:
#             tuple[torch.Tensor, torch.Tensor]: The rotation matrices and the translation vectors.
#         """
#         pass


@dataclass
class OptimizerSettings:
    num_epochs: int = 4000
    warmup_steps: int = 100
    max_lr: float = 0.02
    min_lr: float = 0.0
    early_stop_min_steps: int = 100
    early_stop_loss_grad_norm: float = 1e-4
    temporal_consistency_weight: float = 0.2
    reconstruction_loss_type: Literal["mse", "l1", "huber"] = "huber"
    reconstruction_loss_clip: Optional[float] = 50.0
    reconstruction_loss_clip_start_epoch: int = 200

    @property
    def min_steps(self):
        return max(self.warmup_steps, self.early_stop_min_steps)

    @property
    def reconstruction_loss(self) -> nn.Module:
        if self.reconstruction_loss_type == "mse":
            return nn.MSELoss(reduction="none")
        elif self.reconstruction_loss_type == "l1":
            return nn.L1Loss(reduction="none")
        elif self.reconstruction_loss_type == "huber":
            return nn.HuberLoss(reduction="none")
        else:
            raise ValueError(
                f"Unknown reconstruction loss: {self.reconstruction_loss_type}"
            )


class GradientBatchPnP(PnPSolver):
    def __init__(
        self,
        canonical_points,
        K=None,
        optimizer_settings: Optional[OptimizerSettings] = None,
        device=None,
    ):
        super(GradientBatchPnP, self).__init__(canonical_points, K, device)
        self.optimizer_settings = optimizer_settings or OptimizerSettings()

    def forward(
        self, y, weights=None, R=None, T=None, K=None, B=None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if K is None:
            if self.K is None:
                raise ValueError("Camera intrinsics matrix not provided.")
            K = self.K
        if B is None:
            B = 1
        R_params, T_params = self._initialize_parameters(R, T, B)
        if weights is None:
            weights = torch.ones_like(y)
        elif len(weights.shape) == 2:
            weights = weights.unsqueeze(-1)

        optimizer = optim.Adam([R_params, T_params])
        lr_scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            self.optimizer_settings.num_epochs,
            warmup_steps=self.optimizer_settings.warmup_steps,
            max_lr=self.optimizer_settings.max_lr,
            min_lr=self.optimizer_settings.min_lr,
        )
        min_steps = self.optimizer_settings.min_steps
        reconstruction_loss = self.optimizer_settings.reconstruction_loss

        losses = torch.zeros(self.optimizer_settings.num_epochs, device=self.device)

        for epoch in tqdm.tqdm(range(self.optimizer_settings.num_epochs)):
            optimizer.zero_grad()
            rendered_points = self.pose_and_render_points(
                R_params, T_params, K, self.canonical_points
            )

            reconstruction_loss = (
                reconstruction_loss(rendered_points[:, :, :2], y) * weights
            )
            if (
                self.optimizer_settings.reconstruction_loss_clip is not None
                and epoch > self.optimizer_settings.reconstruction_loss_clip_start_epoch
            ):
                reconstruction_loss = torch.clamp(
                    reconstruction_loss,
                    max=self.optimizer_settings.reconstruction_loss_clip,
                )
            reconstruction_loss = torch.mean(reconstruction_loss)
            consistency_loss = temporal_consistency_loss(R_params, T_params)

            total_loss = (
                reconstruction_loss
                + self.optimizer_settings.temporal_consistency_weight * consistency_loss
            )

            total_loss.backward()
            optimizer.step()
            lr_scheduler.step()

            losses[epoch] = total_loss.item()
            if epoch > min_steps:
                loss_grads = torch.tensor(
                    losses[1:], device=self.device
                ) - torch.tensor(losses[:-1], device=self.device)
                loss_grad = loss_grads[-min_steps:].abs().mean()
                if loss_grad < self.optimizer_settings.early_stop_loss_grad_norm:
                    break

        return rotation_6d_to_matrix(R_params.detach()), T_params.detach()

    def _initialize_parameters(self, R, T, B):
        if R is None:
            R_params = nn.Parameter(
                torch.tensor(
                    [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], device=self.device
                ).repeat(B, 1)
            )
        else:
            if not torch.is_tensor(R):
                R = torch.tensor(R, device=self.device)
            if R.dim() == 2:
                R = R.unsqueeze(0).repeat(B, 1, 1)
            R_params = nn.Parameter(matrix_to_rotation_6d(R))
        if T is None:
            T_params = nn.Parameter(torch.zeros(B, 3, device=self.device))
        else:
            if not torch.is_tensor(T):
                T = torch.tensor(T, device=self.device)
            if T.dim() == 1:
                T = T.unsqueeze(0).repeat(B, 1)
            T_params = nn.Parameter(T)
        return R_params, T_params


class OldGradientBatchPnP(nn.Module):
    def __init__(
        self, points, num_timestamps, initial_T=None, initial_R=None, device=None
    ):
        super(OldGradientBatchPnP, self).__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)
        self.num_timestamps = num_timestamps

        if initial_R is None:
            identity_rotation_6d = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], device=self.device
            )
            self.rotations_6d = nn.Parameter(
                identity_rotation_6d.repeat(num_timestamps, 1)
            )
        else:
            if not torch.is_tensor:
                initial_R = torch.tensor(initial_R, device=self.device)
            if initial_R.dim() == 2:
                initial_R = initial_R.unsqueeze(0).repeat(num_timestamps, 1, 1)
            self.rotations_6d = nn.Parameter(matrix_to_rotation_6d(initial_R))

        if initial_T is None:
            self.translation_vectors = nn.Parameter(
                torch.zeros(num_timestamps, 3, device=self.device)
            )
        else:
            if not torch.is_tensor:
                initial_T = torch.tensor(initial_T, device=self.device)
            if initial_T.dim() == 1:
                initial_T = initial_T.unsqueeze(0).repeat(num_timestamps, 1)
            self.translation_vectors = nn.Parameter(initial_T)

        self.canonical_points = nn.Parameter(
            points.to(self.device), requires_grad=False
        )

    def transform_points(self, points, rotation_matrices, translation_vectors):
        # points: [N, 3], rotation_matrices: [T, 3, 3], translation_vectors: [T, 3]
        transformed = torch.matmul(points, rotation_matrices.transpose(1, 2))
        return transformed + translation_vectors.unsqueeze(1)

    @staticmethod
    def render_points(points, camKs):
        # points: [T, N, 3], camKs: [T, 3, 3]
        projected = torch.matmul(points, camKs.transpose(1, 2))
        return projected / points[:, :, 2:3]

    def get_rotation_matrices(self):
        return rotation_6d_to_matrix(100 * self.rotations_6d)

    def get_translation_vectors(self):
        return self.translation_vectors

    def get_pose_matrices(self):
        rotation_matrices = self.get_rotation_matrices()
        translation_vectors = self.get_translation_vectors().unsqueeze(2)
        return torch.cat((rotation_matrices, translation_vectors), dim=2)

    def forward(self, camKs):
        rotation_matrices = self.get_rotation_matrices()
        transformed_points = self.transform_points(
            self.canonical_points, rotation_matrices, self.translation_vectors
        )
        return self.render_points(transformed_points, camKs)


def temporal_consistency_loss(rotations_6d, translations):
    rotation_diff = rotations_6d[1:] - rotations_6d[:-1]
    translation_diff = translations[1:] - translations[:-1]

    rotation_loss = torch.norm(rotation_diff, dim=1).mean()
    translation_loss = torch.norm(translation_diff, dim=1).mean()

    return rotation_loss + translation_loss


def render_train_model(
    model,
    camKs,
    target_points,
    weights: Optional[torch.Tensor] = None,
    num_epochs=2000,
    warmup_steps=100,
    max_lr=5e-3,
    min_lr=0.0,
    early_stop_min_steps=5,
    early_stop_loss_grad_norm=1e-4,
    temporal_consistency_weight=0.1,
    reconstruction_loss="mse",
    reconstruction_loss_clip=None,
    reconstruction_loss_clip_start_epoch=200,
    verbose=False,
    return_all=False,
):
    device = model.device
    camKs = camKs.to(device)  # [T, 3, 3]
    target_points = target_points.to(device)  # [T, N, 2]
    optimizer = optim.Adam(model.parameters())
    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer, num_epochs, warmup_steps=warmup_steps, max_lr=max_lr, min_lr=min_lr
    )
    early_stop_min_steps = max(warmup_steps, early_stop_min_steps)
    if reconstruction_loss == "mse":
        reconstruction_criterion = nn.MSELoss(reduction="none")
    elif reconstruction_loss == "l1":
        reconstruction_criterion = nn.L1Loss(reduction="none")
    elif reconstruction_loss == "huber":
        reconstruction_criterion = nn.HuberLoss(reduction="none")
    else:
        raise ValueError(f"Unknown reconstruction loss: {reconstruction_loss}")

    losses = []

    all_rotations = []
    all_translations = []
    all_rotations.append(model.rotations_6d.detach().clone())
    all_translations.append(model.translation_vectors.detach().clone())

    if weights is None:
        weights = torch.ones_like(target_points)
    elif len(weights.shape) == 2:
        weights = weights.unsqueeze(-1)

    for epoch in tqdm.tqdm(range(num_epochs)):
        optimizer.zero_grad()
        rendered_points = model(camKs)

        reconstruction_loss = (
            reconstruction_criterion(rendered_points[:, :, :2], target_points) * weights
        )
        if (
            reconstruction_loss_clip is not None
            and epoch > reconstruction_loss_clip_start_epoch
        ):
            reconstruction_loss = torch.clamp(
                reconstruction_loss, max=reconstruction_loss_clip
            )
        reconstruction_loss = torch.mean(reconstruction_loss)
        consistency_loss = temporal_consistency_loss(
            model.rotations_6d, model.translation_vectors
        )

        total_loss = (
            reconstruction_loss + temporal_consistency_weight * consistency_loss
        )

        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        all_rotations.append(model.rotations_6d.detach().clone())
        all_translations.append(model.translation_vectors.detach().clone())
        losses.append(total_loss.item())
        if epoch > early_stop_min_steps:
            loss_grads = torch.tensor(losses[1:]) - torch.tensor(losses[:-1])
            loss_grad = loss_grads[-early_stop_min_steps:].abs().mean()
            if loss_grad < early_stop_loss_grad_norm:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

    return all_rotations, all_translations, losses
