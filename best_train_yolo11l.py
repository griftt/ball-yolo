# -*- coding: utf-8 -*-
"""
优化版 YOLOv11 训练脚本 (带详细日志版)
硬件适配：Mac M3 Pro (18GB)
"""

import os
import time
import torch
from ultralytics import YOLO

# ---------------- 1. 路径配置 ----------------
DATASET_DIR = "/Users/grifftwu/Desktop/历史篮球/1126/"
TRAIN_IMAGES = os.path.join(DATASET_DIR, "images/train")
VAL_IMAGES = os.path.join(DATASET_DIR, "images/val")
DATA_YAML = os.path.join(DATASET_DIR, "basketball_hd_dataset.yaml")

# ---------------- 2. 自定义回调函数 (增加日志) ----------------
# 这些函数会在训练的不同阶段自动触发，告诉你进度

def on_train_start(trainer):
    print("\n" + "="*50)
    print("🚀 【训练启动】 正在初始化显存和优化器...")
    print(f"📊 总轮数 (Epochs): {trainer.epochs}")
    print(f"💾 保存路径: {trainer.save_dir}")
    print("="*50 + "\n")

def on_train_epoch_start(trainer):
    # 每一轮开始时打印
    current_epoch = trainer.epoch + 1
    total_epoch = trainer.epochs
    print(f"\n🟢 [进度] 第 {current_epoch}/{total_epoch} 轮开始...")

def on_train_epoch_end(trainer):
    # 每一轮结束时打印
    current_epoch = trainer.epoch + 1
    print(f"🔴 [进度] 第 {current_epoch} 轮训练结束，正在进行验证和保存...")

def on_fit_epoch_end(trainer):
    # 验证完成后的打印
    metrics = trainer.metrics
    if metrics:
        # 尝试获取 mAP50，如果刚开始可能为0
        map50 = metrics.get("metrics/mAP50(B)", 0)
        print(f"📈 [性能] 当前 mAP50: {map50:.4f}")

# ---------------- 3. 生成配置 ----------------
print("Step 1/4: 正在生成数据集配置文件...")
yaml_content = f"""
path: {os.path.abspath(DATASET_DIR)}
train: {os.path.abspath(TRAIN_IMAGES)}
val: {os.path.abspath(VAL_IMAGES)}
nc: 2
names: ['basketball', 'rim']
"""
os.makedirs(DATASET_DIR, exist_ok=True)
with open(DATA_YAML, "w") as f:
    f.write(yaml_content)

# ---------------- 4. 加载模型 ----------------
# 建议先用 Medium 模型，Large 模型在 18GB 内存上加载高清图风险较大
MODEL_NAME = "yolo11s.pt" 
# 这样模型不用从零学起，只需要“适应新环境”
# MODEL_NAME = "./runs/train/yolov11n_hd_optimized/weights/best.pt" 
print(f"Step 2/4: 正在加载预训练模型 {MODEL_NAME} (首次运行会自动下载)...")
model = YOLO(MODEL_NAME)

# 注册我们的自定义日志回调
model.add_callback("on_train_start", on_train_start)
model.add_callback("on_train_epoch_start", on_train_epoch_start)
model.add_callback("on_train_epoch_end", on_train_epoch_end)
model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

# ---------------- 5. 开始训练 ----------------
print("\nStep 3/4: 准备开始训练...")
print("⚠️ 注意：Mac 上启动 MPS 加速可能需要 1-2 分钟预热，期间看起来像卡住，请耐心等待！")
print("⚠️ 注意：正在使用 imgsz=1280 高清模式，速度会比平时慢，但精度更高。")

try:
    results = model.train(
        data=DATA_YAML,
        epochs=50,
        
        # --- 核心优化 ---
        imgsz=1024,        # 高清训练
        batch=4,           # 显存安全值
        
        # --- 增强配置 ---
        mosaic=0.0,        # 关闭马赛克增强（关键）
        mixup=0.0,
        degrees=2.0,
        translate=0.1,# 保持默认或设小一点。平移增强。
        fliplr=0.5,
        scale=0.1,     # 原来是 0.5。改成 0.1，意味着图片大小只会在 90%-110% 之间波动，不会缩得特别小。
        
        # --- 光照增强 ---
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        
        # --- 系统配置 ---
        device="mps",
        workers=0,         # 🔴 改为0！Mac上设为多线程容易在打印日志时卡死，0最稳
        
        project="./runs/train",
        name="yolov11s_hd_train",
        exist_ok=True,
        patience=30,
        save_period=5,
        verbose=True,      # 🔴 开启官方详细日志
        plots=True,
        cache=False        # 🔴 关闭缓存，防止“扫描图片”时卡住
    )
    print("\nStep 4/4: 🎉 训练全部完成！")

except Exception as e:
    print(f"\n❌ 发生错误: {e}")
    print("💡 提示：如果是 'MPS out of memory'，请将 batch 改为 2")