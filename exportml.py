from ultralytics import YOLO

# 1. 加载你训练好的 .pt 模型
model = YOLO("./runs/train/yolov11s_hd_train/weights/best.pt")

# 2. 导出为 CoreML
# format='coreml': 目标格式
# imgsz=1024: 必须固定分辨率，这很重要！
# nms=True: 关键！让模型直接输出最终框，减少 Python 计算量
# half=True: 使用 FP16 半精度（速度快，精度几乎无损，M芯片原生支持）
model.export(format="coreml", imgsz=1024, nms=True, half=True)