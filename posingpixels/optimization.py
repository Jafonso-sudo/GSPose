import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
from misc_utils.warmup_lr import CosineAnnealingWarmupRestarts
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d

class MultiTimestampRenderPoseModel6D(nn.Module):
    def __init__(self, points, num_timestamps, initial_T=None, initial_R=None, device=None):
        super(MultiTimestampRenderPoseModel6D, self).__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.num_timestamps = num_timestamps

        if initial_R is None:
            identity_rotation_6d = torch.tensor([[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], device=self.device)
            self.rotations_6d = nn.Parameter(identity_rotation_6d.repeat(num_timestamps, 1))
        else:
            initial_R = torch.tensor(initial_R, device=self.device)
            if initial_R.dim() == 2:
                initial_R = initial_R.unsqueeze(0).repeat(num_timestamps, 1, 1)
            self.rotations_6d = nn.Parameter(matrix_to_rotation_6d(initial_R))

        if initial_T is None:
            self.translation_vectors = nn.Parameter(torch.zeros(num_timestamps, 3, device=self.device))
        else:
            initial_T = torch.tensor(initial_T, device=self.device)
            if initial_T.dim() == 1:
                initial_T = initial_T.unsqueeze(0).repeat(num_timestamps, 1)
            self.translation_vectors = nn.Parameter(initial_T)

        self.canonical_points = nn.Parameter(points.to(self.device), requires_grad=False)

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
        return rotation_6d_to_matrix(self.rotations_6d)

    def get_translation_vectors(self):
        return self.translation_vectors

    def get_pose_matrices(self):
        rotation_matrices = self.get_rotation_matrices()
        translation_vectors = self.get_translation_vectors().unsqueeze(2)
        return torch.cat((rotation_matrices, translation_vectors), dim=2)

    def forward(self, camKs):
        rotation_matrices = self.get_rotation_matrices()
        transformed_points = self.transform_points(
            self.canonical_points,
            rotation_matrices,
            self.translation_vectors
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
    num_epochs=1000,
    warmup_steps=10,
    max_lr=5e-3,
    min_lr=0.0,
    early_stop_min_steps=5,
    early_stop_loss_grad_norm=1e-4,
    temporal_consistency_weight=0.1,
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
    reconstruction_criterion = nn.MSELoss(reduction="mean")

    losses = []
    
    if return_all:
        rendered_points_history = []

    for epoch in tqdm.tqdm(range(num_epochs)):
        optimizer.zero_grad()
        rendered_points = model(camKs)
        
        reconstruction_loss = reconstruction_criterion(rendered_points[:, :, :2], target_points)
        consistency_loss = temporal_consistency_loss(model.rotations_6d, model.translation_vectors)
        
        total_loss = reconstruction_loss + temporal_consistency_weight * consistency_loss

        total_loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if return_all:
            rendered_points_history.append(rendered_points.detach())

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}: Total Loss = {total_loss.item()}, "
                  f"Reconstruction Loss = {reconstruction_loss.item()}, "
                  f"Consistency Loss = {consistency_loss.item()}")

        losses.append(total_loss.item())
        if epoch > early_stop_min_steps:
            loss_grads = torch.tensor(losses[1:]) - torch.tensor(losses[:-1])
            loss_grad = loss_grads[-early_stop_min_steps:].abs().mean()
            if loss_grad < early_stop_loss_grad_norm:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

    with torch.no_grad():
        final_rendered_points = model(camKs)
        if return_all:
            rendered_points_history.append(final_rendered_points)
            rendered_points_history = torch.stack(rendered_points_history, dim=0)

    return model, rendered_points_history if return_all else final_rendered_points

class RenderPoseModel6D(nn.Module):
    def __init__(self, points, initial_T=None, initial_R=None, device=None):
        super(RenderPoseModel6D, self).__init__()
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.to(self.device)

        if initial_R is None:
            identity_rotation_6d = torch.tensor(
                [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0]], device=self.device
            )
            self.rotation_6d = nn.Parameter(identity_rotation_6d)
        else:
            self.rotation_6d = nn.Parameter(
                matrix_to_rotation_6d(torch.tensor(initial_R, device=self.device))
            )

        if initial_T is None:
            self.translation_vector = nn.Parameter(torch.zeros(3, device=self.device))
        else:
            self.translation_vector = nn.Parameter(
                torch.tensor(initial_T, device=self.device)
            )

        self.canonical_points = nn.Parameter(
            points.to(self.device), requires_grad=False
        )

    def transform_points(self, points, rotation_matrix, translation_vector):
        # Performs: (points @ rotation_matrix.T) + translation_vector (but more efficient)
        return torch.addmm(translation_vector, points, rotation_matrix.T)

    @staticmethod
    def render_points(points, camK):
        return (points @ camK.T) / points[:, 2:3]

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
        rotation_matrix = rotation_6d_to_matrix(self.rotation_6d).squeeze(0)
        transformed_points = self.transform_points(
            self.canonical_points, rotation_matrix, self.translation_vector
        )
        return self.render_points(transformed_points, camK)


def render_train_model_single(
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
    return_all=False,
):
    device = model.device
    camK = camK.to(device)
    target_points = target_points[:, :2].to(device)
    # TODO: Different lr for R and T
    optimizer = optim.Adam(model.parameters())
    lr_scheduler = CosineAnnealingWarmupRestarts(
        optimizer, num_epochs, warmup_steps=warmup_steps, max_lr=max_lr, min_lr=min_lr
    )
    criterion = nn.MSELoss(reduction="mean")
    # criterion = nn.L1Loss(reduction="mean")

    losses = []

    if return_all:
        rendered_points_history = []

    for epoch in range(num_epochs):
        optimizer.zero_grad()
        rendered_points = model(camK)
        loss = criterion(rendered_points[:, :2], target_points)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        if return_all:
            rendered_points_history.append(rendered_points.detach())

        if verbose and epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {loss.item()}")

        losses.append(loss.item())
        if epoch > early_stop_min_steps:
            loss_grads = torch.tensor(losses[1:]) - torch.tensor(losses[:-1])
            loss_grad = loss_grads[-early_stop_min_steps:].abs().mean()
            if loss_grad < early_stop_loss_grad_norm:
                if verbose:
                    print(f"Early stopping at epoch {epoch}")
                break

    with torch.no_grad():
        final_rendered_points = model(camK)
        if return_all:
            rendered_points_history.append(final_rendered_points)
            rendered_points_history = torch.stack(rendered_points_history, dim=0)

    return model, rendered_points_history if return_all else final_rendered_points