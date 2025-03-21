import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
import argparse
import numpy as np

from PIL import Image
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from cotracker.predictor import CoTrackerPredictor



DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

# 传入参数

# 创建一个 ArgumentParser 实例
parser = argparse.ArgumentParser(description="Process some arguments.")

# 添加 --name 参数
parser.add_argument(
    "--name",
    default="cat",
    help="Name of the video (default: cat)",
)

# 添加其他参数
parser.add_argument(
    "--video_path",
    help="path to a video",
)
parser.add_argument(
    "--mask_path",
    help="path to a segmentation mask",
)
parser.add_argument(
    "--checkpoint",
    default="./checkpoints/scaled_offline.pth",
    help="CoTracker model parameters",
)
parser.add_argument("--grid_size", type=int, default=50, help="Regular grid size")
parser.add_argument(
    "--grid_query_frame",
    type=int,
    default=0,
    help="Compute dense and grid tracks starting from this frame",
)
parser.add_argument(
    "--backward_tracking",
    action="store_true",
    help="Compute tracks in both directions, not only forward",
)
parser.add_argument(
    "--use_v2_model",
    action="store_true",
    help="Pass it if you wish to use CoTracker2, CoTracker++ is the default now",
)
parser.add_argument(
    "--offline",
    #action="store_true",
    default=True,
    help="Pass it if you would like to use the offline model, in case of online don't pass it",
)

args = parser.parse_args()

# 如果用户没有指定 video_path 和 mask_path，则使用默认值
if args.video_path is None:
    deFrames_path = f"data/{args.name}/{args.name}_deFrames.mp4"
    default_path = f"data/videos/{args.name}.mp4"
    args.video_path = deFrames_path if os.path.exists(deFrames_path) else default_path
if args.mask_path is None:
    args.mask_path = f"data/{args.name}/{args.name}_mask.png"



# load the input video frame by frame
video = read_video_from_path(args.video_path)
video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
segm_mask = np.array(Image.open(os.path.join(args.mask_path)))
segm_mask = torch.from_numpy(segm_mask)[None, None]

if args.checkpoint is not None:
    if args.use_v2_model:
        model = CoTrackerPredictor(checkpoint=args.checkpoint, v2=args.use_v2_model)
    else:
        if args.offline:
            window_len = 60
        else:
            window_len = 16
        model = CoTrackerPredictor(
            checkpoint=args.checkpoint,
            v2=args.use_v2_model,
            offline=args.offline,
            window_len=window_len,
        )
else:
    model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")

model = model.to(DEFAULT_DEVICE)
video = video.to(DEFAULT_DEVICE)

pred_tracks, pred_visibility = model(
    video,
    grid_size=args.grid_size,
    grid_query_frame=args.grid_query_frame,
    backward_tracking=args.backward_tracking,
    segm_mask=segm_mask
)
print(f"Start tracking object masked in \"data/{args.name}/{args.name}_mask.png\"")

# save a video with predicted tracks
seq_name = args.video_path.split("/")[-1]
vis = Visualizer(save_dir=f"data/{args.name}", pad_value=120, linewidth=3, tracks_leave_trace=-1)
vis.visualize(
    video,
    pred_tracks,
    pred_visibility,
    query_frame=0 if args.backward_tracking else args.grid_query_frame,
    filename=f"{args.name}_tracked"
)

print("End of tracking")