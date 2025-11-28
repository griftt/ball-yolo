# 🏀 Basketball YOLO - 智能篮球进球检测与剪辑系统

基于 YOLOv11 的智能篮球视频分析系统，能够自动检测篮球进球并生成精彩集锦视频。支持 Apple Silicon (MPS) 和 CoreML 加速，特别优化了 Mac M3 Pro 性能。

## ✨ 核心功能

- 🎯 **自动进球检测**：智能识别篮球和篮筐，准确判断进球时刻
- 🎬 **智能剪辑**：自动生成进球前后的精彩片段（可配置时长）
- 🎨 **轨迹可视化**：平滑绘制篮球飞行轨迹，生成完美弧线特效
- ⚡️ **性能优化**：支持跳帧检测、批量处理、多线程剪辑
- 🧠 **智能校准**：自动锁定篮筐位置，适应不同场景
- 🔥 **硬件加速**：完整支持 Apple MPS 和 CoreML Neural Engine

## 🎥 演示效果

系统能够：
- 实时检测篮球和篮筐位置
- 准确判断投篮轨迹和进球时刻
- 自动剪辑生成精彩集锦
- 叠加平滑的篮球轨迹特效

## 📋 系统要求

> 💡 **想了解完整的工作流程？** 查看 [📖 WORKFLOW.md](WORKFLOW.md) - 从数据采集到生成集锦的详细指南

### 硬件要求
- **推荐**：Apple Silicon (M1/M2/M3 系列)
- **最低**：支持 MPS 的 Mac 或 CUDA GPU

### 软件依赖
```bash
Python >= 3.8
ultralytics >= 8.0.0
opencv-python >= 4.8.0
torch >= 2.0.0
numpy >= 1.24.0
tqdm
psutil
ffmpeg  # 视频剪辑必需
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 克隆仓库
git clone https://github.com/griftt/ball-yolo.git
cd ball-yolo

# 安装 Python 依赖
pip install -r requirements.txt

# 安装 FFmpeg
# macOS:
brew install ffmpeg

# Ubuntu/Debian:
sudo apt-get install ffmpeg

# Windows:
# 从 https://ffmpeg.org/download.html 下载并添加到 PATH
```

### 2. 准备模型

项目支持两种模型格式：

**方式一：使用 PyTorch 模型 (.pt)**
```bash
# 下载预训练模型或使用自己训练的模型
# 放置在项目根目录
```

**方式二：使用 CoreML 模型 (.mlpackage)** （推荐 Apple Silicon）
```bash
# 使用 exportml.py 导出为 CoreML 格式
python exportml.py
```

### 3. 运行检测

#### 单视频处理（带轨迹特效）
```bash
# 编辑 ball_track.py 配置
python ball_track.py
```

关键配置：
```python
# 视频路径
VIDEO_PATH = "/path/to/your/video.mov"

# 模型路径
MODEL_PATH = "./runs/train/yolo11_finetune_new_court/weights/best.mlpackage"

# 输出目录
OUTPUT_DIR = "./outputs/auto_mps_clips"

# 剪辑参数
CLIP_PRE_TIME = 4.0   # 进球前 4 秒
CLIP_POST_TIME = 2.0  # 进球后 2 秒
```

#### 批量处理多个视频
```bash
# 编辑 best6.0.py 配置
python best6.0.py
```

批量配置示例：
```python
VIDEO_TASKS = [
    {"path": "/path/to/video1.mp4", "start": 0.5},
    {"path": "/path/to/video2.mp4", "start": 10.25},
]
```

## 📂 项目结构

```
ball-yolo/
├── README.md                   # 📘 项目说明文档
├── WORKFLOW.md                 # 📖 完整工作流程指南（推荐阅读）
├── requirements.txt            # Python 依赖列表
├── .gitignore                  # Git 忽略配置
│
├── 🎯 核心检测脚本
│   ├── ball_track.py           # 单视频处理（完整版，带轨迹特效）
│   ├── best6.0.py              # 批量处理多视频（极速版，推荐）
│   ├── best5.0.py              # 批量处理（v5.0）
│   ├── best4.0.py              # 批量处理（v4.0）
│   ├── best3.0.py              # 批量处理（v3.0）
│   ├── best2.0.py              # 批量处理（v2.0）
│   └── best.py                 # 基础版本
│
├── 🧠 模型训练脚本
│   ├── best_train_yolo11nv2.py # 训练脚本 v2（推荐，优化版）
│   └── best_train_yolo11l.py   # 训练脚本 v1
│
├── 🛠️ 工具脚本
│   ├── datacut.py              # 数据采集与标注工具
│   ├── exportml.py             # 模型导出为 CoreML 格式
│   ├── merge_instant.py        # 视频合并工具（推荐）
│   └── check.py                # 模型检查工具
│
├── 📦 预训练模型（可选）
│   ├── yolo11n.pt              # YOLO11 Nano 模型
│   ├── yolo11s.pt              # YOLO11 Small 模型
│   └── yolo11l.pt              # YOLO11 Large 模型
│
├── outputs/                    # 输出目录（剪辑视频）
└── runs/                       # 训练结果目录
    └── train/
        ├── yolo11_finetune_new_court/
        ├── yolo11sbest/
        └── yolov11s_hd_train/
```

## 🎯 核心脚本说明

### 🏆 推荐使用

#### 1、ball_track.py - 单视频完整版（带轨迵特效）
**适用场景**：需要高质量轨迹特效的单视频处理

**特点**：
- ✅ 智能轨迹清洗算法（去除运球、反弹等干扰）
- ✅ 平滑轨迹渲染（完美抛物线）
- ✅ 自动篮框校准
- ✅ 区域判定逻辑（高位区、触框区、进球区）

**配置参数**：
```python
START_FROM_MINUTES = 0.0        # 从第几分钟开始
MAX_PROCESS_MINUTES = None      # 处理时长 (None=全部)
DRAW_TRAJECTORY = True          # 是否绘制轨迹
TRAJECTORY_COLOR = (0, 140, 255) # 轨迹颜色 (BGR)
CONF_THRES_BALL = 0.15          # 篮球检测置信度
INFERENCE_SIZE = 1024           # 推理分辨率
```

#### 2、best6.0.py - 批量处理器（极速版，推荐）
**适用场景**：批量处理多个视频，追求速度

**特点**：
- ✅ 跳帧检测（FRAME_SKIP 可配置）
- ✅ 多视频队列处理
- ✅ 内存监控
- ✅ 散热保护机制（可选）

**配置参数**：
```python
FRAME_SKIP = 3              # 每3帧检测一次
MAX_PROCESS_MINUTES = 30    # 每个视频最多处理时长
ROTATE_VIDEO_180 = False    # 是否旋转180度

VIDEO_TASKS = [
    {"path": "/path/to/video1.mp4", "start": 25.25},
    {"path": "/path/to/video2.mp4", "start": 27.97},
]
```

### 📚 历史版本（了解即可）

- **best5.0.py**：批量处理 v5，添加了散热保护
- **best4.0.py**：批量处理 v4，优化了内存管理
- **best3.0.py**：批量处理 v3，添加了跳帧功能
- **best2.0.py**：批量处理 v2，多线程剪辑
- **best.py**：基础版本，单视频检测

### 🛠️ 工具脚本

#### datacut.py - 数据采集与标注工具
交互式的数据标注工具，用于从视频中截取训练数据。

**操作说明**：
- `空格`：暂停/播放
- `S`：保存当前帧
- `A/D`：逐帧前进/后退
- `拖动滑块`：快速定位
- `Q`：退出程序

#### exportml.py - CoreML 模型导出
将训练好的 PyTorch 模型导出为 CoreML 格式，利用 Apple Neural Engine 加速。

```python
model = YOLO("./runs/train/yolo11n_640_train/weights/best.pt")
model.export(format="coreml", imgsz=1024, nms=True, half=True)
```

#### merge_instant.py - 视频合并工具
将所有剪辑片段合并为一个完整的集锦视频。

**配置示例**：
```python
INPUT_FOLDERS = [
    "./outputs/auto_mps_clips_batch_final",
]
OUTPUT_FILE = "./outputs/highlight_collection.mp4"
```

#### check.py - 模型检查工具
验证模型文件完整性和基本信息。

## 🧪 模型训练

### 1. 数据采集与标注

```bash
# 使用数据标注工具
python datacut.py
```

**操作说明**：
- `空格`：暂停/播放
- `S`：保存当前帧
- `A/D`：逐帧前进/后退
- 拖动滑块：快速定位

### 2. 训练模型

#### 使用 best_train_yolo11nv2.py（推荐）

优化的训练脚本，专门适配 Mac M3 Pro 硬件。

```bash
python best_train_yolo11nv2.py
```

**配置参数**：
```python
DATASET_DIR = "/Users/grifftwu/Desktop/历史篮球/1126/"
MODEL_NAME = "yolo11n.pt"  # Nano 模型（速度快）
# MODEL_NAME = "yolo11s.pt"  # Small 模型（推荐，性价比高）

epochs = 50          # 训练轮数
imgsz = 640          # 图片分辨率
batch = 8            # 批次大小
device = "mps"       # 使用 Apple MPS 加速
```

**核心优化**：
- `mosaic=1.0`：马赛克增强（对小目标检测重要）
- `close_mosaic=10`：最后 10 轮关闭马赛克（精细化微调）
- `workers=4`：M3 Pro 可用 4 个线程加载数据
- `cache=True`：18GB 内存可开启，速度更快

#### 使用 best_train_yolo11l.py

基础训练脚本，适用于 Large 模型。

```bash
python best_train_yolo11l.py
```

### 3. 导出 CoreML 模型

```bash
python exportml.py
```

导出后的 `.mlpackage` 模型可利用 Neural Engine，推理速度提升 2-3 倍。

## ⚙️ 进球检测逻辑

系统采用三阶段区域判定：

```
1. 高位区判定 (High Zone)
   └─> 球在篮筐上方 150px 范围内

2. 触框判定 (Rim Touch)
   └─> 球与篮筐框重叠

3. 进球区判定 (Goal Zone)
   └─> 球在篮筐下方 150px 范围内

判定逻辑：
- 如果球先经过「高位区」或「触框」
- 然后在 2.5 秒内进入「进球区」
- 则触发进球事件
```

## 🎨 轨迹特效技术

**智能轨迹清洗算法** (`TrajectoryProcessor`):

1. **时间窗口提取**：提取进球前后的球位置历史
2. **最高点检测**：找到抛物线顶点（Apex）
3. **起点智能回溯**：从最高点往前找，遇到下降趋势则截断
4. **终点智能截断**：球落到篮筐下方一定距离后停止
5. **平滑处理**：使用移动平均滤波生成完美弧线

**效果**：
- ✅ 自动去除运球、反弹等干扰轨迹
- ✅ 只保留投篮弧线
- ✅ 渲染平滑无抖动

## 🛠️ 工具脚本

### mergevideo.py - 视频合并
合并多个剪辑片段为一个集锦视频

### merge_instant.py - 即时合并
实时合并输出目录中的所有视频

### check.py - 模型检查
验证模型文件完整性

## 📊 性能优化建议

### Apple Silicon (M1/M2/M3)
```python
device = 'mps'              # 使用 Metal Performance Shaders
INFERENCE_SIZE = 1024       # 推荐分辨率
FRAME_SKIP = 3              # 跳帧检测提速 3 倍
```

### CoreML 模型优化
```python
MODEL_PATH = "best.mlpackage"
# Neural Engine 自动加速
# 推理延迟降低 50-70%
```

### 内存管理
- 关闭缓存：`cache=False`
- 降低 workers：`workers=0`
- 监控内存：集成 `psutil` 实时监控

## ⚠️ 常见问题

### Q1: MPS out of memory
**解决方案**：
```python
batch = 2  # 降低批次大小
imgsz = 640  # 降低推理分辨率
```

### Q2: 检测不到进球
**检查项**：
1. 降低置信度阈值 `CONF_THRES_BALL = 0.1`
2. 检查篮筐是否正确校准（查看日志）
3. 调整区域偏移参数 `GOAL_ZONE_OFFSET`

### Q3: 轨迹断断续续
**解决方案**：
- 降低跳帧：`FRAME_SKIP = 1`
- 提高检测频率
- 检查模型精度

### Q4: 剪辑视频无音频
**检查**：确保 FFmpeg 已安装且源视频有音轨

## 🤝 贡献指南

欢迎提交 Issue 和 Pull Request！

建议贡献方向：
- 🎯 更精确的进球判定算法
- 🎨 更多轨迹特效样式
- 📊 训练数据集共享
- 🌐 支持更多硬件平台

## 📄 许可证

MIT License

## 🙏 致谢

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) - 提供强大的目标检测框架
- Apple Silicon - 提供高效的 MPS 和 Neural Engine 加速

## 📮 联系方式

有问题或建议？欢迎提交 Issue 或联系作者。

---

**⭐️ 如果这个项目对你有帮助，请给个 Star！**
