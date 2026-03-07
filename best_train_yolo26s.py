# -*- coding: utf-8 -*-
"""
优化版 YOLOv26s 训练脚本 (带详细日志版)
硬件适配：Mac M3 Pro (18GB)
"""

import os
import time
import torch
from ultralytics import YOLO

# ---------------- 1. 路径配置 ----------------
DATASET_DIR = "/Users/grifftwu/Desktop/历史篮球/20260111/"
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

# 这样模型不用从零学起，只需要“适应新环境”
MODEL_NAME = "runs/yolo26s/best.pt" 

# MODEL_NAME = "yolo26s.pt"
try:
    # 建议换成 Small 模型，性价比更高
    # 如果没下载，会自动下载
    print(f"Step 2/4: 正在加载预训练模型 {MODEL_NAME} (首次运行会自动下载)...")
    model = YOLO(MODEL_NAME)
    
    # 注册回调
    model.add_callback("on_train_start", on_train_start)
    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)
# ---------------- 5. 开始训练 (优化版) ----------------
    print("\nStep 3/4: 准备开始训练...")
    print("⚠️ 硬件提示: M3 Pro 18GB 性能强劲，已调高 Batch Size 以提升稳定性。")

    results = model.train(
        data=DATA_YAML,
        epochs=50,
        
        # --- 🔥 核心优化区 ---
        imgsz=640,        # 保持高清
        batch=8,          # 🚀 [提升] 16是安全值。如果报错OOM，改成 8。千万别用2。
        
        # --- 🎨 数据增强优化 ---
        mosaic=1.0,        # ✅ [开启] 对小目标检测非常重要
        close_mosaic=10,   # ✅ [新增] 最后10轮关闭马赛克，进行精细化微调
        
        mixup=0.1,         # 稍微给一点 mixup，有助于防止过拟合
        degrees=0.0,       # 篮球场通常是水平的，旋转不要太大，或者直接为0
        translate=0.1,     
        fliplr=0.5,        # 左右翻转没问题
        scale=0.5,         # 恢复默认的 0.5，让模型适应远近不同的球
        
        # --- 🌈 光照增强 (稍微调低一点，太强会破坏球的颜色特征) ---
        hsv_h=0.015,
        hsv_s=0.4,         # 原来0.7太高了，可能把橙色球变成红色
        hsv_v=0.4,
        
        # --- ⚙️ 系统配置 ---
        device="mps",
        workers=4,         # 🚀 [提升] 尝试用4个线程加载数据。如果报错改成0。
        
        project="./runs/train",
        name="yolo11n_640_train_hd", # 名字改一下
        exist_ok=True,
        patience=30,
        save_period=5,     # 每5轮保存一次
        verbose=True,      # 🔴 开启官方详细日志
        plots=True,
        cache=True        # 内存18G如果数据集不大（<5000张），可以改成 True (RAM缓存)，速度更快
    )
    print("\nStep 4/4: 🎉 训练全部完成！")

except Exception as e:
    print(f"\n❌ 发生错误: {e}")
    print("💡 排错指南：")
    print("1. 如果报错 'MPS out of memory' -> 把 batch 改成 8 或 4")
    print("2. 如果报错 'Broken pipe' 或卡住不动 -> 把 workers 改回 0")    