import subprocess
import os
import sys

file_dir = os.path.dirname(os.path.realpath(__file__))
proj_root = os.path.abspath(os.path.join(file_dir, os.pardir))
print(f"Project root: {proj_root}")
sys.path.append(proj_root)

# from posingpixels.functions import get_or_create_object

OBJECT_NAME = "og-lion"
INPUT_NAME = "og-lion"
PROMPTS = {
    "lion": [(1271, 903), (894, 745), (614, 619)],
    "lion-occlusion": [(420, 317), (505, 365), (570, 438), (508, 293)]
}
SPATRACKER_DOWNSAMPLE = 0.4
SPATRACKER_GRID_SIZE = 20


sam2_dir = "../segment-anything-2"
spatracker_dir = "../SpaTracker"

object_dir = os.path.join(proj_root, f"data/objects/{OBJECT_NAME}")

object_video_dir = os.path.join(object_dir, f"{OBJECT_NAME}-annotate")
object_video_path = f"{object_video_dir}/Frames.m4v"

object_database_dir = os.path.join(proj_root, object_dir, f"{OBJECT_NAME}-database")

input_video_dir = os.path.join(proj_root, f"data/inputs/{INPUT_NAME}")
input_video_path = f"{input_video_dir}/Frames.m4v"

# Step 0: Check that the necessary directories exist

if not os.path.exists(sam2_dir):
    raise Exception(f"segment-anything-2 directory not found at {sam2_dir}")
if not os.path.exists(spatracker_dir):
    raise Exception(f"SpaTracker directory not found at {spatracker_dir}")
if not os.path.exists(object_video_dir):
    raise Exception(f"Object video directory not found at {object_video_dir}")
if not os.path.exists(input_video_dir):
    raise Exception(f"Input video directory not found at {input_video_dir}")

# Step 1: Run SAM2 on object registration

# Check if video has been transformed into images
if not os.path.exists(f"{object_video_dir}/img"):
    print("Creating img directory for the object video")
    os.makedirs(f"{object_video_dir}/img")
if len(os.listdir(f"{object_video_dir}/img")) == 0:
    print("Converting object video to images")
    subprocess.run(["ffmpeg", "-i", object_video_path, "-q:v", "2", "-start_number", "0", f"{object_video_dir}/img/%05d.jpg"])
# Run SAM2 on object video if it hasn't been run yet
if not os.path.exists(f"{object_video_dir}/masks"):
    print("Creating masks directory for the object video")
    os.makedirs(f"{object_video_dir}/masks")
if len(os.listdir(f"{object_video_dir}/masks")) == 0:
    video_prompts = PROMPTS.get(OBJECT_NAME)
    prompts_str = ""
    if video_prompts:
        prompts_str = "--prompts '" + str(video_prompts) + "'"
    print("Running SAM2 on object video")
    command = f"""
source $(conda info --base)/etc/profile.d/conda.sh && \
conda activate sam2 && \
python {sam2_dir}/notebooks/posing_pixels.py {object_video_dir}/img {object_video_dir}/masks {prompts_str} && \
conda deactivate
""".strip()
    subprocess.run(command, shell=True, check=True, executable='/bin/bash')
# Step 2: Construct the gaussian object
# get_or_create_object(object_directory=os.path.join(proj_root, f"data/objects/{OBJECT_NAME}"))
# Step 3: Estimate the pose for the first frame of the input video (to get position for SAM2 mask)
# TODO: Implement this, for now gonna just set the position to the center of the frame
# video, camKs = load_test_data(input_video_dir)
# estimate_first_pose_and_save_query_points(video, camKs, f"{input_video_dir}/queries.npy", object_directory=os.path.join(proj_root, f"data/objects/{OBJECT_NAME}"))

# Step 3: Run SAM2 on the input video

# Check if video has been transformed into images
if not os.path.exists(f"{input_video_dir}/img"):
    print("Creating img directory for the input video")
    os.makedirs(f"{input_video_dir}/img")
if len(os.listdir(f"{input_video_dir}/img")) == 0:
    print("Converting input video to images")
    subprocess.run(["ffmpeg", "-i", input_video_path, "-q:v", "2", "-start_number", "0", f"{input_video_dir}/img/%05d.jpg"])
# Run SAM2 on object video if it hasn't been run yet
if not os.path.exists(f"{input_video_dir}/masks"):
    print("Creating masks directory for the input video")
    os.makedirs(f"{input_video_dir}/masks")
if len(os.listdir(f"{input_video_dir}/masks")) == 0:
    video_prompts = PROMPTS.get(INPUT_NAME)
    prompts_str = ""
    if video_prompts:
        prompts_str = "--prompts '" + str(video_prompts) + "'"
    print("Running SAM2 on input video")
    command = f"""
source $(conda info --base)/etc/profile.d/conda.sh && \
conda activate sam2 && \
python {sam2_dir}/notebooks/posing_pixels.py {input_video_dir}/img {input_video_dir}/masks {prompts_str} && \
conda deactivate
""".strip()
    subprocess.run(command, shell=True, check=True, executable='/bin/bash')

# Step 4: Run SpaTracker with mask generated from Step 3

if not os.path.exists(f"{input_video_dir}/tracks"):
    print("Creating spatracker directory for the input video")
    os.makedirs(f"{input_video_dir}/tracks")
if len(os.listdir(f"{input_video_dir}/tracks")) < 4:
    print("Running SpaTracker on input video")
    # python posingpixels.py --downsample 0.5 --grid_size 20 --vid_path <video_path> --mask_path <mask_path> --outdir <output_dir>
    query_points = f"{input_video_dir}/queries.npy" if os.path.exists(f"--query_points {input_video_dir}/queries.npy") else ""
    command = f"""
source $(conda info --base)/etc/profile.d/conda.sh && \
conda activate SpaTrack && \
cd {spatracker_dir} && \
python posingpixels.py --downsample {SPATRACKER_DOWNSAMPLE} --grid_size {SPATRACKER_GRID_SIZE} --vid_path {input_video_path} --mask_path {input_video_dir}/masks/0.png {query_points} --outdir {input_video_dir}/tracks && \
cd {proj_root} && \
conda deactivate
""".strip()
    subprocess.run(command, shell=True, check=True, executable='/bin/bash')

# Step 5: Run aligner & pose optimizer
