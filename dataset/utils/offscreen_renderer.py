# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
import numpy as np
import pyrender
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
cvcam_in_glcam = np.array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])


class ModelRendererOffscreen:
    def __init__(self, cam_K, H, W, zfar=100):
        """
        @window_sizes: H,W
        """
        self.K = cam_K
        self.scene = pyrender.Scene(ambient_light=[1.0, 1.0, 1.0], bg_color=[0, 0, 0, 0])
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

    def render(self, ob_in_cvcam, mesh=None, return_alpha=False):
        if mesh is not None:
            mesh = mesh.copy()
            mesh.apply_transform(cvcam_in_glcam @ ob_in_cvcam)
            assert hasattr(mesh, 'visual') and hasattr(mesh.visual, 'material')
            mesh = pyrender.Mesh.from_trimesh(mesh, smooth=False)
            mesh_node = self.scene.add(
                mesh, pose=np.eye(4), name="ob"
        )  # Object pose parent is cam
        color, depth = self.r.render(self.scene, flags=pyrender.RenderFlags.RGBA)  # depth: float
        if mesh is not None:
            self.scene.remove_node(mesh_node)
            
        alpha = color[:, :, 3] / 255.0
        color = color[:, :, :3]
        if return_alpha:
            return color, alpha, depth
        else:
            return color, depth 