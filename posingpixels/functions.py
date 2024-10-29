import os
import sys
from typing import Optional

import numpy as np

from posingpixels.alignment import get_safe_query_points

PROJ_ROOT = os.getcwd()
sys.path.append(PROJ_ROOT)

import argparse
from dataset.demo_dataset import OnePoseCap_Dataset
from config import inference_cfg as CFG

from posingpixels.utils.gs_pose import (
    create_or_load_gaussian_splat_from_images,
    load_model_net,
    perform_pose_estimation,
)

def _parse_args(object_name: Optional[str], object_directory: Optional[str] = None, model_net=None):
    if not model_net:
        model_net = load_model_net(os.path.join(PROJ_ROOT, "checkpoints/model_weights.pth"))
    if (object_name is None) == (object_directory is None):
        raise ValueError("Either object_name or object_directory must be provided (but not both)")
    elif object_directory:
        object_name = object_directory.split("/")[-1]
    elif object_name:
        object_directory = f"{PROJ_ROOT}/data/objects/{object_name}"
    return object_name, object_directory, model_net

def get_or_create_object(object_name: Optional[str] = None, object_directory: Optional[str] = None, model_net=None):
    object_name, object_directory, model_net = _parse_args(object_name, object_directory, model_net)

    object_video_directory = f"{object_directory}/{object_name}-annotate"
    object_database_directory = f"{object_directory}/{object_name}-database"
    dataset = OnePoseCap_Dataset(
        obj_data_dir=object_video_directory,
        obj_database_dir=object_database_directory,
        use_binarized_mask=CFG.BINARIZE_MASK,
    )
    reference_database = create_or_load_gaussian_splat_from_images(
        object_database_directory,
        object_name,
        model_net,
        obj_refer_dataset=dataset,
    )

    return reference_database

def estimate_first_pose_and_save_query_points(video, camKs, queries_path, object_name: Optional[str] = None, object_directory: Optional[str] = None, model_net=None):
    object_name, object_directory, model_net = _parse_args(object_name, object_directory, model_net)

    reference_database = get_or_create_object(object_name, object_directory, model_net)
    
    initial_pose = perform_pose_estimation(
        model_net, reference_database, video[:1], camKs[:1]
    )[0][0]["track_pose"]
    R, T = initial_pose[:3, :3], initial_pose[:3, 3]
    H, W = video[0].shape[:2]
    
    query_points = get_safe_query_points(
        reference_database["obj_gaussians"],
        R,
        T,
        camKs[0],
        H,
        W,
    )
    
    np.save(queries_path, query_points)

    return R, T

