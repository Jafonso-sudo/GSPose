# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import numpy as np
import pyrender
from tqdm import tqdm
from multiprocessing import Pool

cvcam_in_glcam = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


class ModelRendererOffscreen:
    def __init__(self, cam_K, H, W, zfar=100):
        """
        @window_sizes: H,W
        """
        self.K = cam_K
        self.scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[0, 0, 0])
        self.camera = pyrender.IntrinsicsCamera(
            fx=cam_K[0, 0],
            fy=cam_K[1, 1],
            cx=cam_K[0, 2],
            cy=cam_K[1, 2],
            znear=0.001,
            zfar=zfar,
        )
        self.cam_node = self.scene.add(self.camera, pose=np.eye(4), name="cam")
        self.mesh_nodes = []
        self.zfar = zfar

        self.H = H
        self.W = W
        self.r = pyrender.OffscreenRenderer(self.W, self.H)

    def render(self, ob_in_cvcam, mesh=None):
        if mesh is not None:
            mesh = mesh.copy()
            mesh.apply_transform(cvcam_in_glcam @ ob_in_cvcam)
            mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
            mesh_node = self.scene.add(
                mesh, pose=np.eye(4), name="ob"
            )  # Object pose parent is cam
        color, depth = self.r.render(self.scene)  # depth: float
        if mesh is not None:
            self.scene.remove_node(mesh_node)

        return color, depth

    def render_batch(self, ob_in_cvcam, mesh=None, num_workers=4):
        # Prepare arguments for parallel processing
        args = [(pose, mesh, self.K, self.H, self.W, self.zfar) for pose in ob_in_cvcam]

        # Run parallel rendering with progress bar
        results = _imap_unordered_bar(_render_single, args, num_workers=num_workers)

        # Separate colors and depths
        colors, depths = zip(*results)
        return list(colors), list(depths)


def _render_single(args):
    """Helper function for parallel rendering."""
    pose, mesh, cam_K, H, W, zfar = args
    renderer = ModelRendererOffscreen(cam_K, H, W, zfar)
    return renderer.render(mesh, pose)


def _imap_unordered_bar(func, args, total=None, num_workers=4):
    """
    Wrapper function to add tqdm to imap_unordered.
    """
    if total is None:
        total = len(args)

    with Pool(processes=num_workers) as pool:
        results = []
        with tqdm(total=total, desc="Rendering") as pbar:
            for result in pool.imap_unordered(func, args):
                results.append(result)
                pbar.update()

        return results
