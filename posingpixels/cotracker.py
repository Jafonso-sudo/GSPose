import numpy as np
import mediapy
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import mediapy
from tqdm import tqdm
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch
from torchvision import transforms
from PIL import Image
import os
from typing import List, Iterator, Tuple
import numpy as np
from cotracker.predictor import CoTrackerPredictor, CoTrackerOnlinePredictor
from posingpixels.datasets import ImageBatchIterator
from cotracker.utils.visualizer import Visualizer


def get_offline_cotracker_predictions(
    video_img_directory,
    grid_size=10,
    mask_path=None,
    downcast=False,
    device=torch.device("cuda"),
):
    len_video = len(os.listdir(video_img_directory))
    if mask_path:
        mask = torch.tensor(np.array(Image.open(mask_path))).to(device)[None, None]
    else:
        mask = None
    video = next(
        ImageBatchIterator(video_img_directory, batch_size=len_video, device=device)
    )
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=downcast):
        cotracker_offline = CoTrackerPredictor().to(device)
        pred_tracks, pred_visibility = cotracker_offline(
            video, grid_size=grid_size, segm_mask=mask
        )  # B T N 2,  B T N 1
    return pred_tracks, pred_visibility


def get_online_cotracker_predictions(
    video_img_directory,
    grid_size=10,
    queries: np.ndarray = None,
    step=8,
    downcast=False,
    device=torch.device("cuda"),
):
    if queries is not None and grid_size:
        raise ValueError("Cannot provide queries and grid_size at the same time. Provide support_grid_size instead.")
    queries_torch = torch.tensor(queries).unsqueeze(0).to(device).float() if queries is not None else None
    batch_iterator = ImageBatchIterator(
        video_img_directory, batch_size=step * 2, overlap=step, device=device
    )
    is_first_step = True
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=downcast):
        cotracker_online = CoTrackerOnlinePredictor(window_len=step * 2).to(device)
        for batch in tqdm(batch_iterator, desc="Processing batches"):
            pred_tracks, pred_visibility, confidence = cotracker_online(
                video_chunk=batch, is_first_step=is_first_step, grid_size=grid_size, queries=queries_torch
            )
            is_first_step = False
    return pred_tracks, pred_visibility, confidence
