import os
import pickle
import sys
import time
from argparse import ArgumentParser
from typing import Optional

import mediapy as media
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from config import inference_cfg as CFG
from dataset.demo_dataset import OnePoseCap_Dataset
from gaussian_object.gaussian_model import GaussianModel
from inference import (
    GS_Tracker,
    ModelParams,
    OptimizationParams,
    PipelineParams,
    create_3D_Gaussian_object,
    create_reference_database_from_RGB_images,
    multiple_initial_pose_inference,
    perform_segmentation_and_encoding,
    render_Gaussian_object_model,
    render_Gaussian_object_model_and_get_radii,
)
from model.network import model_arch as ModelNet


def load_model_net(ckpt_file: str, device: Optional[torch.device] = None):
    if not device:
        device = torch.device("cuda")
    model_net = ModelNet()
    model_net.load_state_dict(torch.load(ckpt_file))
    model_net = model_net.to(device)
    model_net.eval()
    return model_net


def load_test_data(video_directory_path: str):
    query_video_camKs = list()
    with open(os.path.join(video_directory_path, "Frames.txt"), "r") as cf:
        for row in cf.readlines():
            if len(row) > 0 and row[0] != "#":
                camk_dat = np.array([float(c) for c in row.strip().split(",")])
                camk = np.eye(3)
                camk[0, 0] = camk_dat[-4]
                camk[1, 1] = camk_dat[-3]
                camk[0, 2] = camk_dat[-2]
                camk[1, 2] = camk_dat[-1]
                query_video_camKs.append(camk)
    query_video_frames = media.read_video(
        os.path.join(video_directory_path, "Frames.m4v")
    )  # NxHxWx3
    query_video_frames.shape

    return query_video_frames, query_video_camKs


def load_existing_gaussian_splat(
    reference_path: str, device: Optional[torch.device] = None
):
    if not device:
        device = torch.device("cuda")
    with open(reference_path, "rb") as df:
        reference_database = pickle.load(df)

    for _key, _val in reference_database.items():
        if isinstance(_val, np.ndarray):
            reference_database[_key] = torch.as_tensor(_val, dtype=torch.float32).to(
                device
            )

    gs_ply_path = reference_database["obj_gaussians_path"]
    obj_gaussians = GaussianModel(sh_degree=3)
    obj_gaussians.load_ply(gs_ply_path)
    print("load 3D-OGS model from ", gs_ply_path)
    reference_database["obj_gaussians"] = obj_gaussians

    return reference_database


def create_or_load_gaussian_splat_from_images(
    demo_data_dir: str,
    obj_name: str,
    model_net: ModelNet,
    device: Optional[torch.device] = None,
    obj_refer_dataset: Optional[Dataset] = None,
):
    if not device:
        device = torch.device("cuda")

    
    if not obj_refer_dataset:
        refer_seq_dir = os.path.join(
            demo_data_dir, f"{obj_name}-annotate"
        )  # reference sequence directory
        obj_database_dir = os.path.join(
            demo_data_dir, f"{obj_name}-database"
        )  # object database directory
        obj_database_path = os.path.join(
            obj_database_dir, "reference_database.pkl"
        )  # object database file path
        obj_refer_dataset = OnePoseCap_Dataset(
            obj_data_dir=refer_seq_dir,
            obj_database_dir=obj_database_dir,
            use_binarized_mask=CFG.BINARIZE_MASK,
        )
    else:
        obj_dir = obj_refer_dataset.obj_dir
        obj_name = obj_refer_dataset.obj_name
        obj_database_dir = os.path.join(
            demo_data_dir, f"{obj_dir}-database"
        )
        obj_database_path = os.path.join(
            obj_database_dir, "reference_database.pkl"
        )

    if not os.path.exists(obj_database_path):
        print(f"Generate object reference database for {obj_name} ...")

        reference_database = create_reference_database_from_RGB_images(
            model_net, obj_refer_dataset, save_pred_mask=True, device=device
        )

        obj_bbox3D = torch.as_tensor(obj_refer_dataset.obj_bbox3d, dtype=torch.float32)
        bbox3d_diameter = torch.as_tensor(
            obj_refer_dataset.bbox3d_diameter, dtype=torch.float32
        )
        reference_database["obj_bbox3D"] = obj_bbox3D
        reference_database["bbox3d_diameter"] = bbox3d_diameter

        parser = ArgumentParser(description="Training script parameters")
        ###### arguments for 3D-Gaussian Splatting Refiner ########
        gaussian_ModelP = ModelParams(parser)
        gaussian_PipeP = PipelineParams(parser)
        gaussian_OptimP = OptimizationParams(parser)
        # gaussian_BG = torch.zeros((3), device=device)

        if "ipykernel_launcher.py" in sys.argv[0]:
            args = parser.parse_args(sys.argv[3:])  # if run in ipython notebook
        else:
            args = parser.parse_args()  # if run in terminal

        print(f"Creating 3D-OGS model for {obj_name} ")
        gs_pipeData = gaussian_PipeP.extract(args)
        gs_modelData = gaussian_ModelP.extract(args)
        gs_optimData = gaussian_OptimP.extract(args)

        gs_modelData.model_path = obj_database_dir
        gs_modelData.referloader = obj_refer_dataset
        gs_modelData.queryloader = obj_refer_dataset

        obj_gaussians = create_3D_Gaussian_object(
            gs_modelData, gs_optimData, gs_pipeData, return_gaussian=True
        )

        reference_database["obj_gaussians_path"] = f"{obj_database_dir}/3DGO_model.ply"

        for _key, _val in reference_database.items():
            if isinstance(_val, torch.Tensor):
                reference_database[_key] = _val.detach().cpu().numpy()
        with open(obj_database_path, "wb") as df:
            pickle.dump(reference_database, df)
        print("save database to ", obj_database_path)

    print("Load database from ", obj_database_path)
    with open(obj_database_path, "rb") as df:
        reference_database = pickle.load(df)

    for _key, _val in reference_database.items():
        if isinstance(_val, np.ndarray):
            reference_database[_key] = torch.as_tensor(_val, dtype=torch.float32).to(
                device
            )

    gs_ply_path = reference_database["obj_gaussians_path"]
    obj_gaussians = GaussianModel(sh_degree=3)
    obj_gaussians.load_ply(gs_ply_path)
    print("load 3D-OGS model from ", gs_ply_path)
    reference_database["obj_gaussians"] = obj_gaussians

    return reference_database


def perform_pose_estimation(
    model_net,
    reference_database,
    query_video_frames,
    query_video_camKs,
    frame_reinit=False,
    device: Optional[torch.device] = None,
    epochs: Optional[int] = None,
):
    if not device:
        device = torch.device("cuda")
    if epochs:
        temp_epochs = CFG.MAX_STEPS
        CFG.MAX_STEPS = epochs

    # Setup configuration and constants
    start_idx = 0
    num_frames = len(query_video_frames)
    frame_interval = 1

    # Initialize tracking variables
    camKs, images = [], []
    track_outputs = []
    track_accum_runtime = 0

    # Helper function to prepare the image and camera intrinsics
    def prepare_image_and_camK(index):
        camK = torch.as_tensor(query_video_camKs[index], dtype=torch.float32)
        image = (
            torch.as_tensor(np.array(query_video_frames[index]), dtype=torch.float32)
            / 255.0
        )
        return camK, image

    # Rescale the image to models input size
    # -- Get the initial image size
    camK, image = prepare_image_and_camK(start_idx)
    raw_hei, raw_wid = image.shape[:2]
    raw_long_size = max(raw_hei, raw_wid)
    raw_short_size = min(raw_hei, raw_wid)
    raw_aspect_ratio = raw_short_size / raw_long_size

    # -- Calculate new image size
    new_wid, new_hei = (
        (CFG.query_longside_scale, int(CFG.query_longside_scale * raw_aspect_ratio))
        if raw_hei < raw_wid
        else (
            int(CFG.query_longside_scale * raw_aspect_ratio),
            CFG.query_longside_scale,
        )
    )
    query_rescaling_factor = CFG.query_longside_scale / raw_long_size

    # Get the initial pose estimate
    def initial_pose_inference(image):
        # -- Scale the image
        que_image = image[None, ...].permute(0, 3, 1, 2).to(device)
        que_image = F.interpolate(
            que_image, size=(new_hei, new_wid), mode="bilinear", align_corners=True
        )
        # -- Perform segmentation and pose inference
        obj_data = perform_segmentation_and_encoding(
            model_net, que_image, reference_database, device=device
        )
        obj_data.update({"camK": camK, "img_scale": max(image.shape[:2])})
        obj_data["bbox_scale"] /= query_rescaling_factor
        obj_data["bbox_center"] /= query_rescaling_factor
        # -- Initial pose estimation
        try:
            init_RTs = multiple_initial_pose_inference(
                obj_data, ref_database=reference_database, device=device
            )
        except Exception as e:
            print(e)
            init_RTs = torch.eye(4)[None].numpy()
        return init_RTs[0]

    track_pose = initial_pose_inference(image=image)

    # Loop through each frame in the video
    for view_idx in range(start_idx, num_frames, frame_interval):
        camK, image = prepare_image_and_camK(view_idx)

        # Reinitialize the pose estimate
        if frame_reinit:
            track_pose = initial_pose_inference(image=image)

        # Refine the pose estimate (with GS tracker)
        track_timer = time.time()
        track_outp = GS_Tracker(
            model_net,
            frame=image,
            prev_pose=track_pose,
            camK=camK,
            ref_database=reference_database,
        )
        track_accum_runtime += time.time() - track_timer

        # Update tracking data
        track_pose = track_outp["track_pose"]
        # bbox_scale, bbox_center = track_outp['bbox_scale'], track_outp['bbox_center']
        track_outputs.append(track_outp)
        images.append(image)
        camKs.append(camK)

        # Log progress every 30 frames
        if (view_idx + 1) % 30 == 0:
            print(
                f"[{view_idx+1}/{num_frames}], \t{(view_idx - start_idx) / track_accum_runtime:.1f} FPS"
            )

    # Restore default CFG
    if epochs:
        CFG.MAX_STEPS = temp_epochs

    return track_outputs, images, camKs


def render_gaussian_model_with_info(
    gaussian_object: GaussianModel,
    camK: np.ndarray,
    H,
    W,
    R: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
):
    if not device:
        device = torch.device("cuda")
    if R is None:
        R = np.eye(3)
    if T is None:
        T = np.zeros(3)

    pose = np.eye(4)
    pose[:3, :3] = R
    pose[:3, 3] = T

    return render_Gaussian_object_model_and_get_radii(
        gaussian_object, camK, pose, H, W, device=device
    )


def render_gaussian_model(
    gaussian_object: GaussianModel,
    camK: np.ndarray,
    H,
    W,
    R: Optional[np.ndarray] = None,
    T: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
):
    return render_gaussian_model_with_info(gaussian_object, camK, H, W, R, T, device)['image']
