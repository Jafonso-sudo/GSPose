import os
import sys

PROJ_ROOT = os.getcwd()
sys.path.append(PROJ_ROOT)

import argparse
from dataset.demo_dataset import OnePoseCap_Dataset
from config import inference_cfg as CFG

from posingpixels.utils.gs_pose import (
    create_or_load_gaussian_splat_from_images,
    load_model_net,
)

# Set up argument parser
parser = argparse.ArgumentParser(
    description="Register an object using OnePoseCap_Dataset."
)
parser.add_argument("test_name", type=str, help="Name of the test")
parser.add_argument("object_name", type=str, help="Name of the object")

args = parser.parse_args()


# Replace TEST_NAME and OBJECT_NAME with arguments from argparse
TEST_NAME = args.test_name
OBJECT_NAME = args.object_name

video_directory = f"{PROJ_ROOT}/data/inputs/{TEST_NAME}"
object_directory = f"{PROJ_ROOT}/data/objects/{OBJECT_NAME}"

object_video_directory = f"{object_directory}/{OBJECT_NAME}-annotate"
object_database_directory = f"{object_directory}/{OBJECT_NAME}-database"

model_net = load_model_net(os.path.join(PROJ_ROOT, "checkpoints/model_weights.pth"))
dataset = OnePoseCap_Dataset(
    obj_data_dir=object_video_directory,
    obj_database_dir=object_database_directory,
    use_binarized_mask=CFG.BINARIZE_MASK,
)
reference_database = create_or_load_gaussian_splat_from_images(
    object_database_directory,
    OBJECT_NAME,
    model_net,
    obj_refer_dataset=dataset,
)
