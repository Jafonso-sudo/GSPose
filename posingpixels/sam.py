import subprocess
import os
import sys

file_dir = os.path.dirname(os.path.realpath(__file__))
proj_root = os.path.abspath(os.path.join(file_dir, os.pardir))
sys.path.append(proj_root)

sam2_dir = os.path.join(proj_root, os.pardir, "segment-anything-2")


def segment(input_image_dir, output_mask_dir, prompts=None):
    # Check existence of valid input image directory
    if (
        not os.path.exists(input_image_dir)
        or not os.path.isdir(input_image_dir)
        or len(os.listdir(input_image_dir)) == 0
    ):
        raise Exception(f"Input image directory not found at {input_image_dir}")
    # Create or clear output mask directory
    if not os.path.exists(output_mask_dir):
        os.makedirs(output_mask_dir)
    for file in os.listdir(output_mask_dir):
        os.remove(os.path.join(output_mask_dir, file))
    prompts_str = f"--prompts '{prompts}'" if prompts else ""
    print(f"Running SAM2 on input image directory {input_image_dir}")
    command = f"""
source $(conda info --base)/etc/profile.d/conda.sh && \
conda activate sam2 && \
python {sam2_dir}/notebooks/posing_pixels.py {input_image_dir} {output_mask_dir} {prompts_str} && \
conda deactivate
""".strip()
    subprocess.run(command, shell=True, check=True, executable="/bin/bash")
    print(f"Finished running SAM2, masks saved to {output_mask_dir}")