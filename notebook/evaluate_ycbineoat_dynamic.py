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
    
    def choose_best(
        tracker_input: CoTrackerInput, pred_tracks, pred_visibility, pred_confidence, view=False
    ):
        true_indexes = torch.tensor(tracker_input.query_to_point_indexes, device=device)
        query_lengths = torch.tensor(tracker_input.query_lengths, device=device)

        if not view:
            point_selector = SelectMostConfidentPoint(
                tracker_input.num_canonical_points, true_indexes, query_lengths
            )
        else:
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

        return best_coords, best_vis, best_conf, best_coords_original, best_indices


    def estimate_poses(
        tracker_input: CoTrackerInput, best_coords_original, best_vis, best_conf
    ):
        N = len(tracker_input)
        K = tracker_input.dataset.K
        x = (
            torch.tensor(tracker_input.canonical_points, dtype=torch.float32)
            .to(device)
            .unsqueeze(0)
            .repeat(N, 1, 1)
        )
        y = best_coords_original.detach().clone().squeeze(0)[:N]

        weights = (best_vis * best_conf).float()[:N]
        weights[best_vis * best_conf < VIS_CONF_THRESHOLD] = 0
        weights = weights.squeeze(0)

        camKs = torch.tensor(K[np.newaxis, :], device=device).float()

        epnp_cv_solver = OpenCVePnP(
            X=x[0],
            K=camKs,
        )
        epnp_cv_R, epnp_cv_T, err = epnp_cv_solver(
            y, X=x, K=torch.tensor(K).to(device).float(), weights=weights
        )

        epnp_cv_poses = torch.eye(4).to(device).unsqueeze(0).repeat(N, 1, 1)
        epnp_cv_poses[:, :3, :3] = epnp_cv_R
        epnp_cv_poses[:, :3, 3] = epnp_cv_T
        return epnp_cv_poses, err

    # def improve_poses(
    #     x, y, K, poses, weights
    # ) -> torch.Tensor:
    #     # Initialize the optimizer to the current pose
        
    #     # Try every pose against every frame, take the best (using reprojection error w/ Huber loss)
        


    with torch.no_grad():
        (
            pred_tracks_batch,
            pred_confidence_batch,
            pred_visibility_batch,
            pred_poses_batch,
            pred_poses_err,
            best_indices_batch
        ) = [], [], [], [], [], []
        dataset.reset_frame_range()
        start_pose = dataset.get_gt_pose(0)
        step = 32
        overlap = 0
        tracks = vis = conf = track_input = best_coords = best_conf = best_vis = None
        for i in range(0, dataset.max_frames, step - overlap):
            dataset.start_frame = i
            dataset.end_frame = min(i + step, dataset.max_frames)
            rgb = dataset.get_rgb(0)
            # plt.imshow(rgb)
            # plt.show()
            print(f"Processing frames {dataset.start_frame} to {dataset.end_frame}")
            # TODO: Force pose to always be in view, and if it is too far gone, do not include it
            # start_pose[:3, 3] = np.array([0, 0, dataset._get_safe_distance()]) # TODO: Doesn't give right perspective, but ensures it's always in view
            rgb, depth, _ = dataset.render_mesh_at_pose(start_pose)
            # TODO: It's good to initialize every point (maybe with a confidence penalty for the ones in the dynamic template)
            # Initialzie dynamic ones from the ones in the best_coords for that point (same with conf and vis)
            if tracks is not None and overlap > 0:
                assert (
                    vis is not None
                    and conf is not None
                    and track_input is not None
                    and best_coords is not None
                    and best_conf is not None
                    and best_vis is not None
                )
                last_specific_length = track_input.query_lengths[-1] if len(track_input.query_lengths) > 4 else -tracks.shape[2]
                forced_coords = tracks[:, -overlap:, :-last_specific_length]
                forced_vis = vis[:, -overlap:, :-last_specific_length]
                forced_vis = torch.logit(forced_vis).clamp(-tracker.init_value, tracker.init_value)
                forced_conf = conf[:, -overlap:, :-last_specific_length]
                forced_conf = torch.logit(forced_conf).clamp(-tracker.init_value, tracker.init_value)
            else:
                forced_coords = forced_vis = forced_conf = None
            print(start_pose)
            tracks, vis, conf, tracks_original, track_input = tracker(
                dataset,
                start_pose=start_pose,
                query_poses=start_pose[np.newaxis],
                forced_coords=forced_coords,
                forced_vis=forced_vis,
                forced_conf=forced_conf,
            )

            Q = track_input.num_query_points
            N = len(track_input)

            best_coords, best_vis, best_conf, best_coords_original, best_indices = (
                choose_best(track_input, tracks, vis, conf, view=True)
            )
            best_indices_batch.append(best_indices[track_input.prepend_length :])
            

            poses, err = estimate_poses(track_input, best_coords_original, best_vis, best_conf)
            poses = poses[
                track_input.prepend_length :
            ]
            if err is not None:
                err = err[
                    track_input.prepend_length :
                ]
            start_pose = poses[-1].detach().cpu().numpy()
            
            video = load_video_images(track_input.video_dir, limit=N)
            visualize_results(
                video,
                tracks,
                vis,
                conf,
                os.path.join(tracker_result_video, "dynamic"),
                # tracker_result_video + f"_{i}",
                num_of_main_queries=track_input.num_query_points,
                filename=f"dynamic_{i}",
            )
            
            visualize_results(
                video,
                best_coords,
                best_vis,
                best_conf,
                os.path.join(tracker_result_video, "dynamic"),
                # tracker_result_video + f"_{i}_best",
                num_of_main_queries=track_input.num_canonical_points,
                filename=f"dynamic_{i}_best",
            )
            
            best_coords, best_vis, best_conf, best_coords_original, best_indices = (
                choose_best(track_input, tracks, vis, conf, view=False)
            )


            print(tracks.shape, vis.shape, conf.shape)
            tracks = tracks[:, track_input.prepend_length :]
            vis = vis[:, track_input.prepend_length :]
            conf = conf[:, track_input.prepend_length :]
            print(tracks.shape, vis.shape, conf.shape)

            pred_tracks_batch.append(tracks.cpu().numpy())
            pred_visibility_batch.append(vis.cpu().numpy())
            pred_confidence_batch.append(conf.cpu().numpy())
            pred_poses_batch.append(poses.cpu().numpy())
            pred_poses_err.append(err)

    
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