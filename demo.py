import numpy as np
import argparse
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import subprocess
import torch
####################################################
# 配置信息
# box 样例1：
name = "rubbish" # 视频名，后缀默认 .mp4
# 选取边界框坐标(单个)
input_box = "[460, 335, 560, 400]"                     # 不使用的话，请设置为None

# 选取(框内的)坐标点(可多个)
input_point = None                                     # 不使用的话，请设置为None
input_label = None                                     # 不使用的话，请设置为None

# points 样例2：
# # 输入视频名称（存于 ./data/videos/）
# name = "paragliding" # 视频名，后缀默认 .mp4
# 选取边界框坐标(单个)
# input_box = None                                     # 不使用的话，请设置为None
# # 选取(框内的)坐标点(可多个)
# input_point = "[[640, 150], [630, 370]]"             # 不使用的话，请设置为None
# input_label = "[1, 1]"  # 其中 0表示背景、1表示目标物体  # 不使用的话，请设置为None

# “抽帧”选项
decrease_frames = False   # 改为True，则会对视频进行抽帧
# 修改抽帧的情况
frame_interval = 2  # 每隔 n 帧提取一次


####################################################

def run_script(script_name, *args):
    try:
        # 构造命令列表，包括脚本名称和参数
        command = ['python', script_name] + list(args)
        # 使用 subprocess.run 运行脚本
        result = subprocess.run(command, check=True, text=True, capture_output=True)
        print(f"Output from {script_name}:\n{result.stdout}")
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}:\n{e.stderr}")

if __name__ == "__main__":
    # 运行抽帧程序
    if decrease_frames is True:
        run_script("decrease_frames.py", "--name", f"{name}", "--frame_interval",f"{frame_interval}")

    # 生成 firstFrame.png、mask.png 文件，生成的文件存于 “./data/name/”中
    if input_box is None:
        run_script("get_mask.py", "--name", f"{name}", "--input_point", f"{input_point}", "--input_label", f"{input_label}")
    elif input_point is None:
        run_script("get_mask.py", "--name", f"{name}", "--input_box", f"{input_box}")
    else:
        run_script("get_mask.py", "--name", f"{name}", "--input_box", f"{input_box}", "--input_point", f"{input_point}", "--input_label", f"{input_label}")

    # 由于 Intel 的数学核心库（Intel MKL）与 GNU 的线程库（libgomp.so.1）之间存在不兼容问题，所以加这个
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    
    # 生成 .mp4 文件，生成的文件存于 “./data/name/name.mp4”
    run_script("tracker.py", "--name", f"{name}")


