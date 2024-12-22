import os  # noqa
import sys  # noqa

file_dir = os.path.dirname(os.path.realpath(__file__))
proj_root = os.path.abspath(os.path.join(file_dir, os.pardir))
sys.path.append(proj_root)


VIS_CONF_THRESHOLD = 0.9

import matplotlib.pyplot as plt
import torch
from posingpixels.datasets import YCBinEOATDataset, load_video_images
from posingpixels.utils.cotracker import visualize_results
from posingpixels.utils.evaluation import get_gt_tracks
from posingpixels.pnp import GradientPnP

import pickle

from posingpixels.utils.cotracker import unscale_by_crop

from posingpixels.utils.evaluation import compute_add_metrics


import time

from posingpixels.pnp import OpenCVePnP
import mediapy
from posingpixels.utils.geometry import (
    apply_pose_to_points_batch,
    render_points_in_2d_batch,
)
from posingpixels.visualization import overlay_bounding_box_on_video

import random
import numpy as np
from posingpixels.alignment import CanonicalPointSampler
from posingpixels.cotracker import CropCoPoseTracker
from posingpixels.pointselector import SelectMostConfidentView


torch.manual_seed(42)
random.seed(0)
np.random.seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

for video_name, object_name in YCBinEOATDataset.videoname_to_object.items():
    # Check if the results already exist
    if os.path.exists(os.path.join(proj_root, "data", "inputs", video_name, "results.pkl")):
        print(f"Skipping {video_name} {object_name} as results already exist")
        continue
    
    print(video_name, object_name)
    video_dir = os.path.join(proj_root, "data", "inputs", video_name)
    tracker_result_video = os.path.join(video_dir)
    obj_dir = os.path.join(proj_root, "data", "objects", object_name)
    
    dataset = YCBinEOATDataset(video_dir, obj_dir)
    
    pnp_solver = OpenCVePnP(min_inliers=20, ransac_inliner_threshold=2.0)
    point_sampler = CanonicalPointSampler()
    tracker = CropCoPoseTracker(
        canonical_point_sampler=point_sampler,
        # pnp_solver=pnp_solver,
        pose_interpolation_steps=1,
    )
    
    dataset.reset_frame_range()
    with torch.no_grad():
        pred_tracks, pred_visibility, pred_confidence, pred_tracks_original, tracker_input = (
            tracker(dataset)
        )
    Q = tracker_input.num_query_points
    N = len(tracker_input)
    
    
    dataset.reset_frame_range()
    N = len(dataset)
    video = load_video_images(tracker_input.video_dir)[:, -N:]
    # init_video = load_video_images(tracker_input.prepend_dir, limit=N, file_type="jpg")
    video_original = load_video_images(dataset.video_rgb_dir)[:, -N:]
    
    

    true_indexes = torch.tensor(tracker_input.query_to_point_indexes, device=device)
    query_lengths = torch.tensor(tracker_input.query_lengths, device=device)

    # tensor_query_to_point = torch.tensor(tracker.query_to_point, device=device)
    # true_indexes = torch.nonzero(tensor_query_to_point)
    # query_lengths = torch.tensor(tracker.queries_sizes, device=device)

    # point_selector = SelectMostConfidentPoint(
    #     tracker_input.num_canonical_points, true_indexes, query_lengths
    # )
    point_selector = SelectMostConfidentView(
        tracker_input.num_canonical_points, true_indexes, query_lengths
    )

    best_coords, best_vis, best_conf, best_indices = point_selector.query_to_point(
        pred_tracks[0],
        pred_visibility[0],
        pred_confidence[0],
        # pred_visibility[0] * pred_confidence[0],
    )
    best_coords = best_coords.unsqueeze(0)
    best_vis = best_vis.unsqueeze(0)
    best_conf = best_conf.unsqueeze(0)

    best_coords_original = unscale_by_crop(
        best_coords[0],
        torch.tensor(tracker_input.bboxes).to(device),
        torch.tensor(tracker_input.scaling).to(device),
    ).unsqueeze(0)
    
    
    visualize_results(
    video,
    pred_tracks[:, tracker_input.prepend_length :],
    pred_visibility[:, tracker_input.prepend_length :],
    pred_confidence[:, tracker_input.prepend_length :],
    tracker_result_video,
    num_of_main_queries=Q,
    )

    visualize_results(
        video,
        best_coords[:, tracker_input.prepend_length :],
        best_vis[:, tracker_input.prepend_length :],
        best_conf[:, tracker_input.prepend_length :],
        tracker_result_video,
        filename="selected_video",
    )

    gt_tracks, gt_visibility = get_gt_tracks(tracker_input)
    visualize_results(
        video,
        torch.tensor(gt_tracks).to(device).unsqueeze(0).float()[:, tracker_input.prepend_length :],
        torch.tensor(gt_visibility).to(device).unsqueeze(0).float()[:, tracker_input.prepend_length :],
        torch.ones_like(torch.tensor(gt_visibility).to(device)).unsqueeze(0).float()[:, tracker_input.prepend_length :],
        tracker_result_video,
        num_of_main_queries=Q,
        filename="gt_video",
        threshold=VIS_CONF_THRESHOLD,
    )
    
    
    # ==========
    # Input
    # ==========
    tracker_input.dataset.reset_frame_range()

    K = tracker_input.dataset.K
    x = (
        torch.tensor(tracker_input.canonical_points, dtype=torch.float32)
        .to(device)
        .unsqueeze(0)
        .repeat(N, 1, 1)
    )
    gt_poses = torch.tensor(tracker_input.gt_poses[tracker_input.prepend_length: tracker_input.prepend_length + N]).float().to(device)
    gt_posed_x = apply_pose_to_points_batch(x, gt_poses[:, :3, :3], gt_poses[:, :3, 3])
    y_gt = render_points_in_2d_batch(gt_posed_x, torch.tensor(K[:3, :3]).float().to(device))

    y = best_coords_original.detach().clone().squeeze(0)[tracker_input.prepend_length: tracker_input.prepend_length + N]


    weights = (best_vis * best_conf).float()
    weights[best_vis * best_conf < VIS_CONF_THRESHOLD] = 0
    weights = weights.squeeze(0)[tracker_input.prepend_length: tracker_input.prepend_length + N]

    camKs = torch.tensor(K[np.newaxis, :], device=device).float()

    # ==========
    # ePnP
    # ==========
    # Start time
    start_time = time.time()
    epnp_cv_solver = OpenCVePnP(
        X=x[0],
        K=camKs,
        ransac_iterations=5000,
        ransac_inliner_threshold=2.0,
    )
    epnp_cv_R, epnp_cv_T, _ = epnp_cv_solver(
        y,
        # X=x,
        K=torch.tensor(K).to(device).float(), weights=weights,
    )
    epnp_cv_poses = torch.eye(4).to(device).unsqueeze(0).repeat(N, 1, 1)
    epnp_cv_poses[:, :3, :3] = epnp_cv_R
    epnp_cv_poses[:, :3, 3] = epnp_cv_T
    # End time
    end_time = time.time()
    print(f"Time to run OpenCV ePnP: {end_time - start_time}")

    # ==========
    # Our model
    # ==========

    gradient_pnp = GradientPnP(
        max_lr=0.02,
        temporal_consistency_weight=1,
        X=x[0],
        K=camKs,
    )

    rotations, translations, all_results = gradient_pnp(
        y,
        weights=weights,
        R=epnp_cv_poses[:, :3, :3].clone(),
        T=epnp_cv_poses[:, :3, 3].clone(),
    )

    gradient_poses = torch.eye(4).to(device).unsqueeze(0).repeat(N, 1, 1)
    gradient_poses[:, :3, :3] = rotations
    gradient_poses[:, :3, 3] = translations


    # ==========
    # Visualize
    # ==========

    my_predicted_poses = gradient_poses
    video_permuted = video_original[0].permute(0, 2, 3, 1)
    bbox_video = overlay_bounding_box_on_video(
        video_permuted[:N].detach().cpu().numpy(),
        dataset.bbox.float(),
        camKs.repeat(N, 1, 1).cpu(),
        gt_poses.detach().cpu().numpy(),
    )
    bbox_video = overlay_bounding_box_on_video(
        bbox_video,
        dataset.bbox.float(),
        camKs.repeat(N, 1, 1).cpu(),
        my_predicted_poses.detach().cpu().numpy(),
        color=(255, 0, 0),
    )

    # Save video
    mediapy.write_video(os.path.join(video_dir, "pose_overlay.mp4"), bbox_video[:N], fps=15)
    
    def compute_and_plot_add_metrics(
        model_3D_pts,
        diameter,
        predicted_poses: np.ndarray,
        gt_poses: np.ndarray,
        percentage=0.1,
        vert_lines=[],
        save_path=None,
    ):
        add_metrics = []
        for i in range(predicted_poses.shape[0]):
            add_metrics.append(
                compute_add_metrics(
                    model_3D_pts,
                    diameter,
                    predicted_poses[i],
                    gt_poses[i],
                    percentage=percentage,
                    return_error=True,
                )
            )
        threshold = diameter * percentage
        score = np.mean(np.array(add_metrics) < threshold, axis=0)
        mean_err = np.mean(add_metrics)
        print(
            f"Percentage of ADD error less than {threshold}: {score}"
        )
        print(f"Mean ADD error: {mean_err}")
        plt.plot(add_metrics)
        plt.axhline(threshold, color="r", linestyle="--")
        for vert_line in vert_lines:
            plt.axvline(vert_line, color="g", linestyle="--")
        plt.title(f"ADD Error over time ({score * 100:.2f}%, {mean_err:.4f})")
        plt.xlabel("Frame")
        plt.ylabel("ADD Error")
        if save_path is not None:
            plt.savefig(save_path)
        plt.show()
        
        
        return add_metrics




    print("RANSAC CV ePnP")
    add_ransac = compute_and_plot_add_metrics(
        np.array(dataset.get_mesh().vertices),
        dataset.obj_diameter,
        epnp_cv_poses.detach().cpu().numpy()[:],
        gt_poses.detach().cpu().numpy(),
        percentage=0.1,
        save_path=os.path.join(video_dir, "add_ransac.png"),
    )
    print("Adam Optimizer")
    add_adam = compute_and_plot_add_metrics(
        np.array(dataset.get_mesh().vertices),
        dataset.obj_diameter,
        gradient_poses.detach().cpu().numpy()[:],
        gt_poses.detach().cpu().numpy(),
        percentage=0.1,
        save_path=os.path.join(video_dir, "add_adam.png"),
    )
    
    
    # Pickle the results
    with open(os.path.join(video_dir, "results.pkl"), "wb") as f:
        pickle.dump(
            {
                "pred_tracks_original": pred_tracks_original.detach().cpu().numpy(),
                "pred_visibility": pred_visibility.detach().cpu().numpy(),
                "pred_confidence": pred_confidence.detach().cpu().numpy(),
                "epnp_cv_poses": epnp_cv_poses.detach().cpu().numpy(),
                "add_ransac": add_ransac,
                "gradient_poses": gradient_poses.detach().cpu().numpy(),
                "add_adam": add_adam,
            },
            f,
        )