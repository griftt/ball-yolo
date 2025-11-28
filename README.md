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
├── README.md              # 项目说明文档
├── requirements.txt       # Python 依赖列表
├── .gitignore            # Git 忽略配置
├── ball_track.py          # 单视频处理（完整版，带轨迹特效）
├── best6.0.py             # 批量处理多视频（极速版）
├── sync_cut.py            # 实时剪辑版本
├── exportml.py            # 模型导出为 CoreML 格式
├── datacut.py             # 数据标注工具
├── best_train_yolo11l.py  # 模型训练脚本
├── mergevideo.py          # 视频合并工具
├── outputs/               # 输出目录
└── runs/                  # 训练结果
    └── train/
        ├── yolo11_finetune_new_court/
        ├── yolo11sbest/
        └── yolov11s_hd_train/
```

## 🎯 核心脚本说明

### ball_track.py - 完整版检测器
**适用场景**：需要高质量轨迹特效的单视频处理

**特点**：
- ✅ 智能轨迹清洗算法（去除运球、反弹等干扰）
- ✅ 平滑轨迹渲染（完美抛物线）
- ✅ 自动篮筐校准
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

### best6.0.py - 批量处理器
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
```

### sync_cut.py - 实时剪辑版
**适用场景**：已知篮筐坐标，直接处理

**特点**：
- ✅ 使用固定篮筐坐标（跳过校准）
- ✅ 状态机进球判定
- ✅ 后台异步剪辑

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

```bash
# 编辑 best_train_yolo11l.py 配置数据集路径
python best_train_yolo11l.py
```

**训练配置**：
```python
DATASET_DIR = "/path/to/your/dataset"
MODEL_NAME = "yolo11s.pt"  # 基础模型
epochs = 50
imgsz = 1024               # 高清训练
batch = 4                  # 批次大小
device = "mps"             # 使用 MPS 加速
```

**优化技巧**：
- 关闭 mosaic 增强（`mosaic=0.0`）避免小球被过度变形
- 降低 scale 参数（`scale=0.1`）保持球的尺寸稳定
- 使用 `workers=0` 避免 Mac 多线程日志卡死

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
