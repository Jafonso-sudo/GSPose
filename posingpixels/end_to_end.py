import subprocess
import os


OBJECT_NAME = "lion"
INPUT_NAME = "demo_lion"
SPATRACKER_DOWNSAMPLE = 0.5
SPATRACKER_GRID_SIZE = 0

file_dir = os.path.dirname(os.path.realpath(__file__))
proj_root = os.path.abspath(os.path.join(file_dir, os.pardir))
print(f"Project root: {proj_root}")

sam2_dir = "../segment-anything-2"
spatracker_dir = "../SpaTracker"

object_video_dir = os.path.join(proj_root, f"data/objects/{OBJECT_NAME}/{OBJECT_NAME}-annotate")
object_video_path = f"{object_video_dir}/Frames.m4v"

object_database_dir = os.path.join(proj_root, f"data/objects/{OBJECT_NAME}/{OBJECT_NAME}-database")

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
    print("Running SAM2 on object video")
    command = f"""
source $(conda info --base)/etc/profile.d/conda.sh && \
conda activate sam2 && \
python {sam2_dir}/notebooks/posing_pixels.py {object_video_dir}/img {object_video_dir}/masks && \
conda deactivate
""".strip()
    subprocess.run(command, shell=True, check=True, executable='/bin/bash')
    
# Step 2: Construct the gaussian object
if not os.path.exists(object_database_dir):
    print(f"Creating object database directory at {object_database_dir}")
    command = f"""
source $(conda info --base)/etc/profile.d/conda.sh && \
conda activate gspose && \
python posingpixels/create_object.py {INPUT_NAME} {OBJECT_NAME} && \
conda deactivate
""".strip()
    subprocess.run(command, shell=True, check=True, executable='/bin/bash')

# Step 3: Estimate the pose for the first frame of the input video (to get position for SAM2 mask)
# TODO: Implement this, for now gonna just set the position to the center of the frame

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
    print("Running SAM2 on input video")
    command = f"""
source $(conda info --base)/etc/profile.d/conda.sh && \
conda activate sam2 && \
python {sam2_dir}/notebooks/posing_pixels.py {input_video_dir}/img {input_video_dir}/masks && \
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
    command = f"""
source $(conda info --base)/etc/profile.d/conda.sh && \
conda activate SpaTrack && \
cd {spatracker_dir} && \
python posingpixels.py --downsample {SPATRACKER_DOWNSAMPLE} --grid_size {SPATRACKER_GRID_SIZE} --vid_path {input_video_path} --mask_path {input_video_dir}/masks/0.png --query_points {input_video_dir}/queries.npy --outdir {input_video_dir}/tracks && \
cd {proj_root} && \
conda deactivate
""".strip()
    subprocess.run(command, shell=True, check=True, executable='/bin/bash')

# Step 5: Run aligner & pose optimizer
