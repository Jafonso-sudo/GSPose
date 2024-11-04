import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import torch
from cotracker.predictor import CoTrackerOnlinePredictor
from posingpixels.datasets import ImageBatchIterator, load_video_images


def get_offline_cotracker_predictions(
    video_img_directory,
    grid_size=10,
    queries: np.ndarray = None,
    downcast=False,
    limit=None,
    device=torch.device("cuda"),
):
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=downcast):
        queries_torch = (
            torch.tensor(queries).unsqueeze(0).to(device).float()
            if queries is not None
            else None
        )
        cotracker_offline = CoTrackerOnlinePredictor(offline=True, window_len=60).to(
            device
        )
        pred_tracks, pred_visibility, confidence = cotracker_offline(
            load_video_images(
                video_img_directory, limit=limit, device=device, dtype=torch.float32
            ),
            grid_size=grid_size,
            queries=queries_torch,
            is_first_step=True,
        )  # B T N 2,  B T N 1, B T N 1
    return pred_tracks, pred_visibility, confidence


def get_online_cotracker_predictions(
    video_img_directory,
    grid_size=10,
    queries: np.ndarray = None,
    step=8,
    downcast=False,
    device=torch.device("cuda"),
):
    offline = step != 8
    if queries is not None and grid_size:
        raise ValueError(
            "Cannot provide queries and grid_size at the same time. Provide support_grid_size instead."
        )
    queries_torch = (
        torch.tensor(queries).unsqueeze(0).to(device).float()
        if queries is not None
        else None
    )
    batch_iterator = ImageBatchIterator(
        video_img_directory, batch_size=step * 2, overlap=step, device=device
    )
    is_first_step = True
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=downcast):
        cotracker_online = CoTrackerOnlinePredictor(
            window_len=step * 2, offline=offline
        ).to(device)
        for batch in tqdm(batch_iterator, desc="Processing batches"):
            first_img = batch[0][0]
            pred_tracks, pred_visibility, confidence = cotracker_online(
                video_chunk=batch,
                is_first_step=is_first_step,
                grid_size=grid_size,
                queries=queries_torch,
            )
            is_first_step = False
    return pred_tracks, pred_visibility, confidence
