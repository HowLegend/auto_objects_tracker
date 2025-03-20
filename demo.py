import numpy as np
import argparse
import os
import subprocess
####################################################
# 配置信息
# 样例1：
# 输入视频名称（存于 ./data/videos/）
name = "paragliding" # 视频名，后缀默认 .mp4
# 用坐标，标出你要提取的物体
input_point = "[[640, 150], [630, 370]]"
input_label = "[1, 1]"  # 其中 0表示背景、1表示目标物体

# 样例2：
# name = "cat" # 视频名，后缀默认 .mp4
# input_point = "[[190, 240]]"
# input_label = "[1]"
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
    # 生成 firstFrame.png、mask.png 文件，生成的文件存于 “./data/name/”中
    run_script("get_mask.py", "--name", f"{name}", "--input_point", f"{input_point}", "--input_label", f"{input_label}")

    # 由于 Intel 的数学核心库（Intel MKL）与 GNU 的线程库（libgomp.so.1）之间存在不兼容问题，所以加这个
    os.environ['MKL_THREADING_LAYER'] = 'GNU'
    
    # 生成 .mp4 文件，生成的文件存于 “./data/name/name.mp4”
    run_script("tracker.py", "--name", f"{name}")


