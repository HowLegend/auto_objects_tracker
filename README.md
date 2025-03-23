# auto_objects_tracker: 连接 SAM 2 和 co-tracker3，实现自动只对视频中的选定物体提取光流

## 开源项目地址
SAM2: https://github.com/facebookresearch/sam2.git  
co-tracker3: https://github.com/facebookresearch/co-tracker

## 克隆仓库
```
git clone https://github.com/HowLegend/auto_objects_tracker.git
cd auto_objects_tracker
```

## 环境配置
### 关于cuda和pytorch
我使用的是 cuda11.8.0 和 pytorch2.1.0 ，具体配置教程参考的是 https://blog.csdn.net/qq_46699596/article/details/134552021 

使用其他版本应该也可以。  

### 下载checkpoints
```
mkdir -p checkpoints
cd checkpoints

# checkpoint of co-tracker 
wget https://huggingface.co/facebook/cotracker3/resolve/main/scaled_offline.pth 

# checkpoint of SAM 2 
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt

cd ..
```

### 安装各种依赖（Set-up）
```
cd setup/cotracker
pip install -e .
pip install matplotlib flow_vis tqdm tensorboard
cd ../sam2
pip install -e .
cd ..
cd ..
pip install opencv-python
pip install imageio
pip install imageio[ffmpeg]
```
## 运行
为了优化 PyTorch 的 CUDA 内存分配策略，避免显存碎片化，请在根据系统类别，在终端运行
```
# Linux
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Windows
set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
后来，我已经把这行代码嵌入到运行文件中，应该无需手动设置了。
### 运行示例-使用示例的视频
在终端上运行命令
```
python demo.py
```
会在 data 目录下新建一个以视频名name为名称的文件夹 data/{name}/{name}.mp4  

PS：若显示错误如下
```
torch.OutOfMemoryError: CUDA out of memory.
```
则说明“显存”不足，可以通过降低视频帧数来减小显卡内存开销。这里准备了“抽帧”脚本 [decrease_frames.py](./decrease_frame.py)。
目前已将该“抽帧”函数集成进demo文件中，在demo文件的“配置信息”代码区可以看见
```
# “抽帧”选项
decrease_frames = False   # 改为True，则会对视频进行抽帧

# 修改抽帧的情况
frame_interval = n  # 每隔 n 帧提取一次, 例如 15s 的视频，常常 n = 2 开始
```
将“抽帧”选项打开，并设置好抽帧情况，即
```
decrease_frames = True  # “抽帧”选项
frame_interval = 2   # 修改抽帧的情况/抽帧数量
```
再在终端重新运行
```
python demo.py
```
抽帧后的视频会存于 data/{name}/{name}_deFrames.mp4  

若仍然显示“显存不足”，那么加大抽帧数量，直到显存足够而不报错  

⚠注意：当检测到抽帧后的文件（"data/{name}/{name}_deFrames.mp4"）存在时，默认使用的就是抽帧后的文件，而不是原视频文件（"data/videos/{name}.mp4"）。若想使用回原视频文件，请把抽帧后的文件（"data/{name}/{name}_deFrames.mp4"）删掉/更名/转移至其他位置，让程序检测不到"data/{name}/{name}_deFrames.mp4"的存在。

### 追踪 自己的视频中的自定义物体
如果需要上传自己的视频，那么请把 .mp4视频 放在 data/videos/your_video_name.mp4 。 并且在demo中的“配置信息”处的变量name修改成
```
name = "your_video_name"
```
然后输入坐标，通过坐标来选出你想要追踪的物体。规则：  
input_point 表示选取的目标位置坐标，选中目标本身部分的一点，会自动提取该目标的全身。可以只是用一个坐标，也可以插入多个坐标来选取多个物体  
input_label 中的每个 label 应与 input_point 每个 point(x,y) 对应，表示坐标点的特征，0 表示“这里是背景（非选）”，1 表示“这里是物体（选中）”  。
  
可以借助[points2mask可视化工具](./check_mask_with_xy.ipynb)来获得“选取目标点”的坐标(x,y)，还可以通过mask图像来确定“目标物体”是否被完美选中
```
input_point = "[[x_1, y_1], [x_2, y_2]]"
input_label = "[label_1, label_2]" 
```
# auto_objects_tracker
