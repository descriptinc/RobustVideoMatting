import torch
import argparse
from inference import convert_video

parser = argparse.ArgumentParser(description='enter the input video file')
parser.add_argument('input_file', help='enter the path to the input video file')
args = parser.parse_args()

model = torch.hub.load("PeterL1n/RobustVideoMatting", "mobilenetv3").cuda() # or "resnet50"
#convert_video = torch.hub.load("PeterL1n/RobustVideoMatting", "converter")

input_video_path=args.input_file
output_composition_video_path = input_video_path.split(".mp4")[0] + "_com.mp4"
output_alpha_video_path = input_video_path.split(".mp4")[0] + "_alpha.mp4"
output_foreground_video_path = input_video_path.split(".mp4")[0] + "_fgr.mp4"

convert_video(
    model,                           # The loaded model, can be on any device (cpu or cuda).
    input_source=input_video_path,        # A video file or an image sequence directory.
    downsample_ratio=None,           # [Optional] If None, make downsampled max size be 512px.
    output_type='video',             # Choose "video" or "png_sequence"
    output_composition=output_composition_video_path,    # File path if video; directory path if png sequence.
    output_alpha=output_foreground_video_path,          # [Optional] Output the raw alpha prediction.
    output_foreground=output_foreground_video_path,     # [Optional] Output the raw foreground prediction.
    output_video_mbps=4,             # Output video mbps. Not needed for png sequence.
    seq_chunk=12,                    # Process n frames at once for better parallelism.
    num_workers=1,                   # Only for image sequence input. Reader threads.
    progress=True                    # Print conversion progress.
)
