# ball-yolo

`ball-yolo` 是一个本地 Python 工作区，用来做篮球检测相关实验。

它主要负责这些事情：

- 采集和校验训练数据
- 训练 `basketball` / `rim` 检测模型
- 将训练好的模型导出为 CoreML
- 在录制视频上运行投篮 / 进球检测
- 使用 `ffmpeg` 自动裁剪精彩片段

它不是主应用。主应用在 `HighlightMoment/`，这里更像模型侧和实验侧工作台。

## 这个目录里有什么

### 检测与剪辑脚本

- `best7.0.py`
  当前最像主入口的批量处理脚本。负责动态检测篮筐和篮球、执行投篮判定、导出自动剪辑。
- `best.py` 到 `best6.0.py`
  同一条工作流的历史迭代版本，主要用于对比和回退。
- `check.py`、`check_person.py`、`check_shoot.py`、`debug_detection.py`
  用来检查模型输出、排查投篮逻辑。
- `goal_detection_debug.py`、`goal_detection_diagnostic.py`
  更聚焦于进球判定本身的调试。

### 训练与模型导出

- `best_train_yolo11l.py`
- `best_train_yolo11nv2.py`
- `best_train_yolo26s.py`
- `best_train_yolo26s_person.py`

这些脚本用于训练不同版本的 YOLO 模型。

- `exportml.py`
  把训练好的 `.pt` 模型导出成 CoreML。
- `convert_model.py`、`mlcheck.py`
  模型转换和检查辅助脚本。

### 数据准备

- `datacut.py`
  交互式抽帧工具，用来制作训练数据。
- `datacut_multi.py`
  多视频版本的数据提取脚本。
- `validate_labels.py`
  标签检查。
- `auto_annotate_person.py`
  和人物标注相关的辅助脚本。
- `video_split.py`
  拆分原始大视频。

### 工具脚本

- `merge_instant.py`
  快速合并生成的剪辑。
- `video_player_with_delete.py`
  用于回看和人工筛选素材或输出。
- `gemini_check.py`、`checkosnet.py`
  实验期留下的辅助工具。

### 模型与输出

- `yolo11n.pt`、`yolo11s.pt`、`yolo26s.pt`
  放在目录根部的基础模型或训练模型。
- `runs/`
  训练和推理输出目录，也是这个目录里体积最大的部分。

## 当前最实用的工作流

如果只走最短路径，基本就是：

1. 用 `datacut.py` 或 `datacut_multi.py` 采集训练帧
2. 标注两类目标：`basketball` 和 `rim`
3. 用 `best_train_yolo26s.py` 或其他训练脚本训练模型
4. 如果应用侧需要，用 `exportml.py` 导出 CoreML
5. 用 `best7.0.py` 做批量检测和自动剪辑
6. 如有需要，再用 `merge_instant.py` 合并片段

更详细的流程见 [WORKFLOW.md](/Users/grifftwu/IdeaProjects/HighlightMoment/ball-yolo/WORKFLOW.md)。

## 环境要求

### Python 依赖

直接安装 `requirements.txt`：

```bash
pip install -r requirements.txt
```

核心依赖包括：

- `ultralytics`
- `opencv-python`
- `torch`
- `torchvision`
- `numpy`
- `tqdm`
- `psutil`

### 外部依赖

裁剪视频依赖 `ffmpeg`。

macOS：

```bash
brew install ffmpeg
```

## 推荐入口

### 1. 训练模型

示例：

```bash
python best_train_yolo26s.py
```

这个脚本当前会：

- 自动写出 YOLO 数据集 yaml
- 从 `runs/yolo26s/best.pt` 加载模型
- 训练 50 轮
- 使用 Apple `mps`

运行前先检查脚本顶部的硬编码数据集路径。

### 2. 导出 CoreML

```bash
python exportml.py
```

这个脚本当前默认：

- 源模型：`runs/detect/runs/train/yolo26s_640_train_hd_person/weights/best.pt`
- 导出格式：CoreML
- 输入尺寸：`640`

如果你的权重不在这个路径，先改脚本。

### 3. 批量处理视频

```bash
python best7.0.py
```

这个脚本顶部最重要的配置有：

- `VIDEO_TASKS`
- `OUTPUT_DIR`
- `MODEL_PATH`
- `FRAME_SKIP`
- `CONF_THRES_RIM`
- `CONF_THRES_BALL`
- `CLIP_PRE_TIME`
- `CLIP_POST_TIME`

`best7.0.py` 当前做的事情：

- 加载一个 YOLO 模型
- 逐个扫描 `VIDEO_TASKS` 中的视频
- 逐帧检测篮球和篮筐
- 应用投篮 / 进球判定逻辑
- 通过 `ffmpeg` 导出剪辑

## 数据集假设

训练脚本默认使用标准 YOLO 数据集布局：

```text
dataset/
  images/
    train/
    val/
  labels/
    train/
    val/
  basketball_hd_dataset.yaml
```

当前工作区使用的类别约定：

- `0 = basketball`
- `1 = rim`

## 这个目录的现实情况

- 很多脚本里仍然写着 `/Users/grifftwu/Desktop/...` 这样的绝对路径
  这说明它更像个人工作站脚本，而不是对外封装好的通用工具
- 脚本命名按版本迭代，不是稳定 API
  当前最接近主入口的是 `best7.0.py`
- `runs/` 很大，而且主要是生成物
  它应该被看作输出目录，不是核心源码

## 建议先看哪些文件

- [WORKFLOW.md](/Users/grifftwu/IdeaProjects/HighlightMoment/ball-yolo/WORKFLOW.md)
- [best7.0.py](/Users/grifftwu/IdeaProjects/HighlightMoment/ball-yolo/best7.0.py)
- [best_train_yolo26s.py](/Users/grifftwu/IdeaProjects/HighlightMoment/ball-yolo/best_train_yolo26s.py)
- [exportml.py](/Users/grifftwu/IdeaProjects/HighlightMoment/ball-yolo/exportml.py)

## 一句话总结

`ball-yolo` 是篮球检测模型的实验台。
训练、导出、检测实验都在这里做，验证稳定之后再和 `HighlightMoment` 主应用联动。
