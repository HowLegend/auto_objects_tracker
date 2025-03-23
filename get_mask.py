# 导包
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import numpy as np
import torch
import argparse
import matplotlib.pyplot as plt
from PIL import Image
import cv2

###################### 配置信息 ##########################

parser = argparse.ArgumentParser()

parser.add_argument(
    "--name",
)
parser.add_argument(
    "--input_box",
    default=None,
)
parser.add_argument(
    "--input_point",
    default=None,
)
parser.add_argument(
    "--input_label",
    default=None,
)
args = parser.parse_args()

def parse_array(array_str):
    # 使用 eval 将字符串解析为 Python 列表
    array_list = eval(array_str)
    # 将列表转换为 NumPy 数组
    return np.array(array_list)

name=args.name
if args.input_box is not None:
    input_box=parse_array(args.input_box)
else:
    input_box=None
if args.input_point is not None:
    input_point=parse_array(args.input_point)
    input_label=parse_array(args.input_label)
else:
    input_point=None
    input_label=None

###################### 生成mask的过程 ######################

# Set-up

# select the device for computation
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
print(f"using device: {device}")

if device.type == "cuda":
    # use bfloat16 for the entire notebook
    torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    if torch.cuda.get_device_properties(0).major >= 8:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
elif device.type == "mps":
    print(
        "\nSupport for MPS devices is preliminary. SAM 2 is trained with CUDA and might "
        "give numerically different outputs and sometimes degraded performance on MPS. "
        "See e.g. https://github.com/pytorch/pytorch/issues/84936 for a discussion."
    )


np.random.seed(3)

def show_mask(mask, ax, random_color=False, borders = True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask = mask.astype(np.uint8)
    mask_image =  mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    if borders:
        import cv2
        contours, _ = cv2.findContours(mask,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        # Try to smooth contours
        contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]
        mask_image = cv2.drawContours(mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2) 
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))    

def show_masks(image, masks, scores, point_coords=None, box_coords=None, input_labels=None, borders=True):
    for i, (mask, score) in enumerate(zip(masks, scores)):
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        show_mask(mask, plt.gca(), borders=borders)
        if point_coords is not None:
            assert input_labels is not None
            show_points(point_coords, input_labels, plt.gca())
        if box_coords is not None:
            # boxes
            show_box(box_coords, plt.gca())
        if len(scores) > 1:
            plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)
        plt.axis('off')
        plt.show()

def extract_first_frame(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}.")
        return
    
    print(f"Extracting the first frame from the video which is from {video_path}")
    
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"First frame saved as {output_path}")
    else:
        print(f"Error: Could not read the first frame from {video_path}.")
    
    cap.release()

# 提取视频第一帧
print("Start getting mask")
video_path = f"data/videos/{name}.mp4"
firstFrame_path = f"data/{name}/{name}_firstFrame.png"

folder_path = f"data/{name}" # 新建一个目录
os.makedirs(folder_path, exist_ok=True)  # 如果目录已存在，不会抛出错误

extract_first_frame(video_path, firstFrame_path)

# 载入图片
image = Image.open(f"data/{name}/{name}_firstFrame.png")

# # 获取分辨率
# width, height = image.size
# print('图片分辨率：', width, 'x', height)

image = np.array(image.convert("RGB"))

# 使用SAM2来选择物体
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

sam2_checkpoint = "checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"

sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)

predictor = SAM2ImagePredictor(sam2_model)

predictor.set_image(image)

# # 展示 选取的对象坐标
# print("show the selected points")
# print(f"\ninput_point:\n{input_point}")
# print(f"input_label:\n{input_label}\n")

# Predict with `SAM2ImagePredictor.predict`
masks, scores, logits = predictor.predict(
    point_coords=input_point if input_point is not None else None,
    point_labels=input_label if input_label is not None else None,
    box=input_box if input_box is not None else None,
    multimask_output=False,
)
sorted_ind = np.argsort(scores)[::-1]
masks = masks[sorted_ind]
scores = scores[sorted_ind]
logits = logits[sorted_ind]

masks.shape  # (number_of_masks) x H x W

# 保存图片
# 确保 masks[0] 是 uint8 类型
mask_0 = (masks[0] > 0.5).astype(np.uint8) * 255  # 将掩码转换为二值图像（0 和 255）
output_path = f"data/{name}/{name}_mask.png"
# 使用 OpenCV 保存
cv2.imwrite(output_path, mask_0)
print(f"First frame saved as {output_path}")

print("End of getting mask")



