from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from misc_utils.warmup_lr import CosineAnnealingWarmupRestarts
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d


class RenderPoseModel6D(nn.Module):
    def __init__(self, points, initial_T=None, initial_R=None):
        super(RenderPoseModel6D, self).__init__()

        # Learnable 6D rotation parameters and translation vector
        if initial_R is None:
            identity_rotation_6d = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]]
            )  # First two columns of identity matrix
            self.rotation_6d = nn.Parameter(
                identity_rotation_6d
            )  # Rotation matrix in 6D representation
        else:
            self.rotation_6d = nn.Parameter(
                matrix_to_rotation_6d(torch.tensor(initial_R))
            )

        if initial_T is None:
            self.translation_vector = nn.Parameter(
                torch.zeros(3)
            )  # Translation vector: [tx, ty, tz]
        else:
            self.translation_vector = nn.Parameter(torch.tensor(initial_T))

        self.canonical_points = nn.Parameter(
            points, requires_grad=False
        )  # Points to transform in canonical space

    @staticmethod
    def transform_points(points, rotation_matrix, translation_vector):
        return (points @ rotation_matrix.T) + translation_vector

    @staticmethod
    def render_points(points, camK):
        return (points @ camK.T) / points[:, 2][:, None]

    def get_rotation_matrix(self):
        return rotation_6d_to_matrix(self.rotation_6d).squeeze(0)

    def get_translation_vector(self):
        return self.translation_vector

    def get_pose_matrix(self):
        return torch.cat(
            (self.get_rotation_matrix(), self.get_translation_vector().unsqueeze(0).T),
            dim=1,
        )

    def forward(self, camK):
        # Convert 6D representation to a rotation matrix using PyTorch3D
        rotation_matrix = rotation_6d_to_matrix(self.rotation_6d).squeeze(0)

        # Apply the rotation and translation to the points
        transformed_points = self.transform_points(
            self.canonical_points, rotation_matrix, self.translation_vector
        )

        # Render the points in 2D using the camera intrinsics
        rendered_points = self.render_points(transformed_points, camK)

        return rendered_points


def render_train_model(
    model,
    camK,
    target_points,
    num_epochs=400,
    warmup_steps=10,
    max_lr=5e-3,
    min_lr=0.0,
    early_stop_min_steps=5,
    early_stop_loss_grad_norm=1e-4,
    verbose=False,
    device: Optional[torch.device] = None,
    return_all: bool = False,
):
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define optimizer
    optimizer = optim.Adam(model.parameters())

    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer, num_epochs, warmup_steps=warmup_steps, max_lr=max_lr, min_lr=min_lr
    )

    # Define loss function (Mean Squared Error)
    criterion = nn.MSELoss(reduction="mean").to(device)
    # criterion = nn.L1Loss(reduction='mean').to(device)

    target_points = target_points[:, :2]

    # Return all rendered points epochs?
    if return_all:
        rendered_points = []

    losses = []

    # Training loop
    for epoch in range(num_epochs):
        optimizer.zero_grad()

        # Forward pass: Transform the points
        rendered_points = model.forward(camK)

        # Compute the loss (MSE between transformed points and target points)
        loss = criterion(rendered_points[:, :2], target_points)

        # Backward pass: Compute gradients
        loss.backward()

        # Update the parameters
        optimizer.step()
        lr_scheduler.step()

        if return_all:
            rendered_points.append(rendered_points.detach())

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

        losses.append(loss)
        if epoch > early_stop_min_steps:
            loss_grads = (
                torch.as_tensor(losses)[1:] - torch.as_tensor(losses)[:-1]
            ).abs()
            loss_grad = loss_grads[-early_stop_min_steps:].mean()
            if loss_grad < early_stop_loss_grad_norm:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

    with torch.no_grad():
        rendered_points = model.forward(camK)
        if return_all:
            rendered_points.append(rendered_points.detach())
            rendered_points = torch.stack(rendered_points, dim=0)

    return model, rendered_points
