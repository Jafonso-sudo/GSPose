# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

from typing import Iterable, Mapping, Optional, Tuple, Union

import torch

from posingpixels.cotracker import CoTrackerInput
from posingpixels.utils.cotracker import get_ground_truths, scale_by_crop
from posingpixels.utils.geometry import apply_pose_to_points

def compute_add_metrics(
    model_3D_pts: np.ndarray,
    diameter: float,
    pose_pred: np.ndarray,
    pose_target: np.ndarray,
    percentage: float = 0.1,
    return_error: bool = False,
    syn: bool = False,
) -> Union[bool, float]:
    """Computes the ADD metric.
    Args:
        model_3D_pts: A numpy array of shape [N, 3] representing the 3D points of
            the model.
        diameter: The diameter of the model.
        pose_pred: A numpy array of shape [4, 4] representing the predicted pose.
        pose_target: A numpy array of shape [4, 4] representing the target pose.
        percentage: The percentage of the diameter to use as the threshold.
        return_error: If True, returns the error instead of a boolean.
        syn: If True, uses a cKDTree to compute the mean distance.
        model_unit: The unit of the model.
    Returns:
        The ADD metric if return_error is False, otherwise the error.
    """
    from scipy import spatial

    if pose_pred.shape[0] == 4:
        pose_pred = pose_pred[:3]
    if pose_target.shape[0] == 4:
        pose_target = pose_target[:3]

    diameter_thres = diameter * percentage
    model_pred = apply_pose_to_points(model_3D_pts, pose_pred[:3, :3], pose_pred[:3, 3])
    model_target = apply_pose_to_points(model_3D_pts, pose_target[:3, :3], pose_target[:3, 3])

    if syn:
        mean_dist_index = spatial.cKDTree(model_pred)
        mean_dist, _ = mean_dist_index.query(model_target, k=1)
        mean_dist = np.mean(mean_dist)
    else:
        mean_dist = np.mean(np.linalg.norm(model_pred - model_target, axis=-1))

    if return_error:
        return mean_dist
    elif mean_dist < diameter_thres:
        return True
    else:
        return False


def compute_tapvid_metrics(
    query_points: np.ndarray,
    gt_occluded: np.ndarray,
    gt_tracks: np.ndarray,
    pred_occluded: np.ndarray,
    pred_tracks: np.ndarray,
    query_mode: str,
    evaluation_points: Optional[np.ndarray] = None,
) -> Mapping[str, np.ndarray]:
    """Computes TAP-Vid metrics (Jaccard, Pts. Within Thresh, Occ. Acc.)
    See the TAP-Vid paper for details on the metric computation.  All inputs are
    given in raster coordinates.  The first three arguments should be the direct
    outputs of the reader: the 'query_points', 'occluded', and 'target_points'.
    The paper metrics assume these are scaled relative to 256x256 images.
    pred_occluded and pred_tracks are your algorithm's predictions.
    This function takes a batch of inputs, and computes metrics separately for
    each video.  The metrics for the full benchmark are a simple mean of the
    metrics across the full set of videos.  These numbers are between 0 and 1,
    but the paper multiplies them by 100 to ease reading.
    Args:
       query_points: The query points, an in the format [t, y, x].  Its size is
         [b, n, 3], where b is the batch size and n is the number of queries
       gt_occluded: A boolean array of shape [b, n, t], where t is the number
         of frames.  True indicates that the point is occluded.
       gt_tracks: The target points, of shape [b, n, t, 2].  Each point is
         in the format [x, y]
       pred_occluded: A boolean array of predicted occlusions, in the same
         format as gt_occluded.
       pred_tracks: An array of track predictions from your algorithm, in the
         same format as gt_tracks.
       query_mode: Either 'first' or 'strided', depending on how queries are
         sampled.  If 'first', we assume the prior knowledge that all points
         before the query point are occluded, and these are removed from the
         evaluation.
        evaluation_points: A boolean array of shape [b, n, t], where t is the number
            of frames.  True indicates that the point should be evaluated at that frame.
    Returns:
        A dict with the following keys:
        occlusion_accuracy: Accuracy at predicting occlusion.
        pts_within_{x} for x in [1, 2, 4, 8, 16]: Fraction of points
          predicted to be within the given pixel threshold, ignoring occlusion
          prediction.
        jaccard_{x} for x in [1, 2, 4, 8, 16]: Jaccard metric for the given
          threshold
        average_pts_within_thresh: average across pts_within_{x}
        average_jaccard: average across jaccard_{x}
    """

    metrics = {}
    # Fixed bug is described in:
    # https://github.com/facebookresearch/co-tracker/issues/20
    eye = np.eye(gt_tracks.shape[2], dtype=np.int32)

    if query_mode == "first":
        # evaluate frames after the query frame
        query_frame_to_eval_frames = np.cumsum(eye, axis=1) - eye
    elif query_mode == "strided":
        # evaluate all frames except the query frame
        query_frame_to_eval_frames = 1 - eye
    else:
        raise ValueError("Unknown query mode " + query_mode)

    query_frame = query_points[..., 0]
    query_frame = np.round(query_frame).astype(np.int32)
    valid_evaluation = query_frame_to_eval_frames[query_frame] > 0 # B x N x T
    if evaluation_points is None:
        evaluation_points = valid_evaluation
    else:
        evaluation_points = evaluation_points & valid_evaluation

    # Occlusion accuracy is simply how often the predicted occlusion equals the
    # ground truth.
    
    occ_acc_ = np.sum(np.equal(pred_occluded, gt_occluded) & evaluation_points, axis=1) / np.sum(evaluation_points, axis=1)
    occ_acc = np.sum(
        np.equal(pred_occluded, gt_occluded) & evaluation_points,
        axis=(1, 2),
    ) / np.sum(evaluation_points)
    metrics["occlusion_accuracy_over_time"] = occ_acc_
    metrics["occlusion_accuracy"] = occ_acc

    # Next, convert the predictions and ground truth positions into pixel
    # coordinates.
    visible = np.logical_not(gt_occluded)
    pred_visible = np.logical_not(pred_occluded)
    all_frac_within = []
    all_frac_within_time_ = []
    all_frac_within_point_ = []
    all_jaccard = []
    all_jaccard_time_ = []
    all_jaccard_point_ = []
    for thresh in [1, 2, 4, 8, 16]:
        # True positives are points that are within the threshold and where both
        # the prediction and the ground truth are listed as visible.
        within_dist = np.sum(
            np.square(pred_tracks - gt_tracks),
            axis=-1,
        ) < np.square(thresh)
        is_correct = np.logical_and(within_dist, visible)

        # Compute the frac_within_threshold, which is the fraction of points
        # within the threshold among points that are visible in the ground truth,
        # ignoring whether they're predicted to be visible.
        count_correct_ = is_correct & evaluation_points
        count_correct = np.sum(
            is_correct & evaluation_points,
            axis=(1, 2),
        )
        count_visible_points_ = visible & evaluation_points
        count_visible_points = np.sum(visible & evaluation_points, axis=(1, 2))
        frac_correct = count_correct / count_visible_points
        frac_correct_time_ = np.sum(count_correct_, axis=1) / np.sum(count_visible_points_, axis=1)
        frac_correct_per_point_ = np.sum(count_correct_, axis=2) / np.sum(count_visible_points_, axis=2)
        metrics["time_pts_within_" + str(thresh)] = frac_correct_time_
        metrics["per_point_pts_within_" + str(thresh)] = frac_correct_per_point_
        metrics["pts_within_" + str(thresh)] = frac_correct
        all_frac_within.append(frac_correct)
        all_frac_within_time_.append(frac_correct_time_)
        all_frac_within_point_.append(frac_correct_per_point_)

        true_positives_ = is_correct & pred_visible & evaluation_points
        true_positives = np.sum(
            true_positives_, axis=(1, 2)
        )

        # The denominator of the jaccard metric is the true positives plus
        # false positives plus false negatives.  However, note that true positives
        # plus false negatives is simply the number of points in the ground truth
        # which is easier to compute than trying to compute all three quantities.
        # Thus we just add the number of points in the ground truth to the number
        # of false positives.
        #
        # False positives are simply points that are predicted to be visible,
        # but the ground truth is not visible or too far from the prediction.
        gt_positives_ = visible & evaluation_points
        gt_positives = np.sum(visible & evaluation_points, axis=(1, 2))
        false_positives = (~visible) & pred_visible
        false_positives = false_positives | ((~within_dist) & pred_visible)
        false_positives_ = false_positives & evaluation_points
        false_positives = np.sum(false_positives & evaluation_points, axis=(1, 2))
        jaccard_time_ = np.sum(true_positives_, axis=1) / (np.sum(gt_positives_ + false_positives_, axis=1))
        jaccard_per_point_ = np.sum(true_positives_, axis=2) / (np.sum(gt_positives_ + false_positives_, axis=2))
        jaccard = true_positives / (gt_positives + false_positives)
        metrics["time_jaccard_" + str(thresh)] = jaccard_time_
        metrics["per_point_jaccard_" + str(thresh)] = jaccard_per_point_
        metrics["jaccard_" + str(thresh)] = jaccard
        all_jaccard.append(jaccard)
        all_jaccard_time_.append(jaccard_time_)
        all_jaccard_point_.append(jaccard_per_point_)
        

    
    metrics["average_jaccard"] = np.mean(
        np.stack(all_jaccard, axis=1),
        axis=1,
    )
    metrics["average_jaccard_time"] = np.mean(
        np.stack(all_jaccard_time_, axis=1),
        axis=1,
    )
    metrics["average_jaccard_point"] = np.mean(
        np.stack(all_jaccard_point_, axis=1),
        axis=1,
    )
    metrics["average_pts_within_thresh"] = np.mean(
        np.stack(all_frac_within, axis=1),
        axis=1,
    )
    metrics["average_pts_within_time"] = np.mean(
        np.stack(all_frac_within_time_, axis=1),
        axis=1,
    )
    metrics["average_pts_within_point"] = np.mean(
        np.stack(all_frac_within_point_, axis=1),
        axis=1,
    )
    return metrics

# def get_gt_tracks(tracker: CoMeshTracker, crop: bool = True, device: Optional[torch.device] = None):
#     if not device:
#         device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     gt_tracks = []
#     gt_visibility = []
#     gt_pose = None
#     for i in range(tracker.limit):
#         if (pose_i := tracker.get_gt_pose(i)) is not None:
#             gt_pose = pose_i
#         assert gt_pose is not None
#         gt_depth = tracker.get_gt_depth(i)

#         coords_i, vis_i = get_ground_truths(
#             gt_pose, tracker.K, tracker.unposed_3d_points, tracker.get_mask(i), gt_depth
#         )
#         gt_tracks.append(coords_i)
#         gt_visibility.append(vis_i)
#     tracks, visibilities = np.array(gt_tracks), np.array(gt_visibility)
#     if crop:
#         tracks = (
#             scale_by_crop(
#                 torch.tensor(tracks).float().to(device),
#                 torch.tensor(tracker.bboxes).to(device),
#                 torch.tensor(tracker.scaling).to(device),
#             )
#             .cpu()
#             .numpy()
#         )
#     return tracks, visibilities

def get_gt_tracks(tracker_input: CoTrackerInput, crop: bool = True, device: Optional[torch.device] = None):
    if not device:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gt_tracks = []
    gt_visibility = []
    gt_pose = None
    for i in range(len(tracker_input)):
        if (pose_i := tracker_input.get_gt_pose(i)) is not None:
            gt_pose = pose_i
        assert gt_pose is not None
        gt_depth = tracker_input.get_gt_depth(i)

        coords_i, vis_i = get_ground_truths(
            gt_pose, tracker_input.dataset.K, tracker_input.canonical_points, tracker_input.get_mask(i), gt_depth
        )
        gt_tracks.append(coords_i)
        gt_visibility.append(vis_i)
    tracks, visibilities = np.array(gt_tracks), np.array(gt_visibility)
    if crop:
        tracks = (
            scale_by_crop(
                torch.tensor(tracks).float().to(device),
                torch.tensor(tracker_input.bboxes).to(device),
                torch.tensor(tracker_input.scaling).to(device),
            )
            .cpu()
            .numpy()
        )
    return tracks, visibilities