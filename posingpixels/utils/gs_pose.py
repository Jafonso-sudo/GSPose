import os
import pickle
import sys
import time
from argparse import ArgumentParser
from typing import TYPE_CHECKING, Optional

import mediapy as media
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from config import inference_cfg as CFG
from dataset.demo_dataset import OnePoseCap_Dataset
from dataset.inference_datasets import YCBInEOAT_Dataset
from gaussian_object.gaussian_model import GaussianModel
from gaussian_object.arguments import ModelParams, OptimizationParams, PipelineParams
from gaussian_object.build_3DGaussianObject import create_3D_Gaussian_object
from gaussian_object.utils.graphics_utils import focal2fov
from inference import (
    GS_Tracker,
    create_reference_database_from_RGB_images,
    create_reference_database_from_RGB_images_YCB,
    multiple_initial_pose_inference,
    perform_segmentation_and_encoding,
    render_Gaussian_object_model_and_get_radii,
)
from model.network import model_arch as ModelNet
from torch import optim
# from misc_utils.metric_utils import *
from misc_utils.warmup_lr import CosineAnnealingWarmupRestarts

from gaussian_object.gaussian_render import render as GS_Renderer
from gaussian_object.cameras import Camera as GS_Camera
from pytorch_msssim import SSIM, MS_SSIM

# Only for typing
if TYPE_CHECKING:
    from posingpixels.datasets import YCBinEOATDataset


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


def create_or_load_gaussian_splat_from_ycbineoat(
    dataset: "YCBinEOATDataset",
    model_net: ModelNet,
    device: Optional[torch.device] = None,
):
    if not device:
        device = torch.device("cuda")
    obj_database_dir = dataset.object_dir
    obj_database_path = os.path.join(obj_database_dir, "reference_database.pkl")
    if not os.path.exists(obj_database_path):
        reference_database = create_reference_database_from_RGB_images_YCB(model_net, dataset, device=device)
        obj_bbox3D = torch.as_tensor(dataset.bbox, dtype=torch.float32)
        bbox3d_diameter = torch.as_tensor(dataset.bbox_diameter, dtype=torch.float32)
        reference_database["obj_bbox3D"] = obj_bbox3D
        reference_database["bbox3d_diameter"] = bbox3d_diameter
        
        parser = ArgumentParser(description="Training script parameters")
        ###### arguments for 3D-Gaussian Splatting Refiner ########
        gaussian_ModelP = ModelParams(parser)
        gaussian_PipeP = PipelineParams(parser)
        gaussian_OptimP = OptimizationParams(parser)
        # gaussian_BG = torch.zeros((3), device=device)

        if "ipykernel_launcher.py" in sys.argv[0] or "create_object.py" in sys.argv[0]:
            args = parser.parse_args(sys.argv[3:])  # if run in ipython notebook
        else:
            args = parser.parse_args()  # if run in terminal

        print(f"Creating 3D-OGS model") # for {dataset.videoname_to_object[dataset.video_name]} ")
        gs_pipeData = gaussian_PipeP.extract(args)
        gs_modelData = gaussian_ModelP.extract(args)
        gs_optimData = gaussian_OptimP.extract(args)

        gs_modelData.model_path = obj_database_dir
        gs_modelData.referloader = dataset
        gs_modelData.queryloader = dataset

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
        obj_refer_dataset = OnePoseCap_Dataset(
            obj_data_dir=refer_seq_dir,
            obj_database_dir=obj_database_dir,
            use_binarized_mask=CFG.BINARIZE_MASK,
        )
    elif isinstance(obj_refer_dataset, OnePoseCap_Dataset):
        obj_database_dir: str = obj_refer_dataset.obj_database_dir # type: ignore
        obj_name = obj_refer_dataset.obj_name
    elif isinstance(obj_refer_dataset, YCBInEOAT_Dataset):
        obj_dir = obj_refer_dataset.obj_dir
        obj_name = obj_refer_dataset.obj_name
        obj_database_dir = os.path.join(
            demo_data_dir, f"{obj_dir}-database"
        )
    else:
        raise ValueError("obj_refer_dataset must be an instance of OnePoseCap_Dataset or YCBInEOAT_Dataset")
    
    obj_database_path = os.path.join(obj_database_dir, "reference_database.pkl")

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

        if "ipykernel_launcher.py" in sys.argv[0] or "create_object.py" in sys.argv[0]:
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


def unzoomed_GS_Tracker(ref_database, frame, mask, camK, prev_poses_proposals, device):
    parser = ArgumentParser()
    gaussian_PipeP = PipelineParams(parser)
    gaussian_BG = torch.zeros((3), device=device)
    
    L1Loss = torch.nn.L1Loss(reduction='none')
    SSIM_METRIC = SSIM(data_range=1, size_average=False, channel=3) # channel=1 for grayscale images
    
    height, width = frame.shape[:2]
           
    cx_offset, cy_offset = (camK[0, 2] - width / 2) / (width / 2), (camK[1, 2] - height / 2) / (height / 2)
    cam_fx, cam_fy = camK[0, 0], camK[1, 1]
    FovX = focal2fov(cam_fx, width)
    FovY = focal2fov(cam_fy, height)
        
    if not isinstance(frame, torch.Tensor):
        frame = torch.as_tensor(frame, dtype=torch.float32)
    if frame.shape[0] != 3:
        frame = frame.permute(2, 0, 1)
    frame = frame.float().to(device) / 255.0
    if not isinstance(mask, torch.Tensor):
        mask = torch.as_tensor(mask, dtype=torch.float32)
    mask = mask.float().to(device)
    if isinstance(prev_poses_proposals, torch.Tensor):
        assert False, "prev_pose should be a numpy array"
        # prev_pose = prev_pose.detach().cpu().numpy()
    frame *= mask
    obj_gaussians = ref_database['obj_gaussians']
    obj_gaussians.initialize_pose()
    best_proposal, best_start_loss, best_proposal_index, time_added = None, None, None, 0
    with torch.no_grad():
        time_start = time.time()
        for idx, prev_pose in enumerate(prev_poses_proposals):
            track_camera = GS_Camera(T=prev_pose[:3, 3],
                                R=prev_pose[:3, :3].T, 
                                FoVx=FovX, FoVy=FovY,
                                cx_offset=cx_offset, cy_offset=cy_offset,
                                image=frame, colmap_id=0, uid=0, image_name='', 
                                mask=mask, gt_alpha_mask=None, data_device=device)
            render_img = GS_Renderer(track_camera, obj_gaussians, gaussian_PipeP, gaussian_BG)['render'] * mask
            loss = 0
            
            rgb_loss = (L1Loss(render_img, frame) * mask).mean()
            loss += rgb_loss
        
            ssim_loss = ((1 - SSIM_METRIC(render_img[None], frame[None])) * mask).mean()
            loss += ssim_loss
            
            if best_proposal is None or loss < best_start_loss:
                best_proposal = prev_pose
                best_start_loss = loss
                best_proposal_index = idx
        time_end = time.time()
        time_added = time_end - time_start
            
        
    track_camera = GS_Camera(T=best_proposal[:3, 3],
                            R=best_proposal[:3, :3].T, 
                            FoVx=FovX, FoVy=FovY,
                            cx_offset=cx_offset, cy_offset=cy_offset,
                            image=frame, colmap_id=0, uid=0, image_name='', 
                            mask=mask, gt_alpha_mask=None, data_device=device)

    obj_gaussians.initialize_pose()
    
    optimizer = optim.AdamW([obj_gaussians._delta_R, obj_gaussians._delta_T])

    lr_scheduler = CosineAnnealingWarmupRestarts(optimizer, 
                                                 CFG.MAX_STEPS, 
                                                 warmup_steps=CFG.WARMUP, 
                                                 max_lr=CFG.START_LR, min_lr=CFG.END_LR)
    losses = list()
    frame *= mask
    norms = list()
    params = list()
    for iter_step in range(CFG.MAX_STEPS):
        optimizer.zero_grad()
        render_img = GS_Renderer(track_camera, obj_gaussians, gaussian_PipeP, gaussian_BG)['render'] * mask
        
        # Debug
        # render_img_np = (render_img.detach().cpu().numpy() * 255).astype(np.uint8).transpose(1, 2, 0)
        
        loss = 0
        
        rgb_loss = (L1Loss(render_img, frame) * mask).mean()
        loss += rgb_loss
    
        ssim_loss = ((1 - SSIM_METRIC(render_img[None], frame[None])) * mask).mean()
        loss += ssim_loss
        
        loss.backward()
        with torch.no_grad():
            total_grad_norm = 0
            for param in optimizer.param_groups[0]['params']:
                total_grad_norm += param.grad.norm().item()**2
            norms.append(total_grad_norm**0.5)
        optimizer.step()
        lr_scheduler.step()
        
        losses.append(loss.item())
        if iter_step >= CFG.EARLY_STOP_MIN_STEPS:
            loss_grads = (torch.as_tensor(losses)[1:] - torch.as_tensor(losses)[:-1]).abs()
            loss_grad = loss_grads[-CFG.EARLY_STOP_MIN_STEPS:].mean() 
            if loss_grad < CFG.EARLY_STOP_LOSS_GRAD_NORM:
                break
            
            
    
    gs3d_delta_RT = obj_gaussians.get_delta_pose.squeeze(0).detach().cpu().numpy()
    curr_pose = best_proposal @ gs3d_delta_RT
        
    return{
        'track_pose': curr_pose,
        'render_img': render_img,
        'iter_step': iter_step,
        'loss': losses,
        'grad_norm': norms,
        'time_added': time_added,
        'best_proposal': best_proposal_index,
    }


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
