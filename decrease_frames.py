import cv2
import os

def extract_frames_and_create_video(input_video_path, output_video_path, frame_interval=1):
    """
    从原始视频中抽帧，并生成一个新的视频文件。
    
    参数:
        input_video_path (str): 原始视频文件路径。
        output_video_path (str): 输出视频文件路径。
        frame_interval (int): 每隔多少帧提取一次。默认为1（提取每一帧）。
    """
    # 打开原始视频
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Could not open input video {input_video_path}.")
        return

    # 获取原始视频的属性
    fps = cap.get(cv2.CAP_PROP_FPS) / frame_interval  # 新视频的帧率
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 创建 VideoWriter 对象，用于写入新的视频
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 编码格式
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    frame_count = 0  # 当前帧编号

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # 读取到视频末尾，退出循环

        if frame_count % frame_interval == 0:
            # 将当前帧写入新的视频
            out.write(frame)

        frame_count += 1

    print(f"Processed {frame_count} frames. Output video saved to {output_video_path}.")
    cap.release()
    out.release()

# 输入你的视频名称
name = "paragliding"

folder_path = f"data/{name}" # 新建一个目录
os.makedirs(folder_path, exist_ok=True)  # 如果目录已存在，不会抛出错误

input_video_path = f"data/videos/{name}.mp4"  # 替换为你的视频文件路径
output_video_path = f"data/{name}/{name}_deFrames.mp4"  # 替换为保存帧的文件夹路径


# 修改抽帧的情况
frame_interval = 2  # 每隔 n 帧提取一次

extract_frames_and_create_video(input_video_path, output_video_path, frame_interval)