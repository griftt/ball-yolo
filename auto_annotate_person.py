# -*- coding: utf-8 -*-
"""
自动标注人员到训练数据集
遍历训练图片，使用 YOLO 模型检测人物，并将检测结果添加到标注文件
"""

import os
import cv2
import torch
from pathlib import Path
from ultralytics import YOLO

# ==================== 配置区域 ====================

# 数据集路径
DATASET_DIR = "/Users/grifftwu/Desktop/历史篮球/20260111bak2"
IMAGES_DIR = os.path.join(DATASET_DIR, "images/train")
LABELS_DIR = os.path.join(DATASET_DIR, "labels/train")

# 模型路径 (使用预训练的 YOLO11 模型，这个模型包含人员检测)
MODEL_PATH = "yolo26s.pt"

# 人员检测阈值
CONF_THRES_PERSON = 0.25

# 设备
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

# 是否跳过已包含 person 标注的文件 (True=跳过, False=覆盖添加)
SKIP_EXISTING = True

# =============================================================


def add_person_to_label(label_path, detections, img_width, img_height):
    """
    将检测到的人员添加到标注文件

    Args:
        label_path: 标注文件路径
        detections: 人员检测结果 [(x1, y1, x2, y2, conf), ...]
        img_width: 图片宽度
        img_height: 图片高度
    """

    if detections:
        with open(label_path, "a") as f:
            for x1, y1, x2, y2, conf in detections:
                # 转换为 YOLO 格式 (归一化坐标)
                x_center = ((x1 + x2) / 2) / img_width
                y_center = ((y1 + y2) / 2) / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                # 类别索引: 0=basketball, 1=rim, 2=person
                f.write(f"2 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n")


def has_person_annotation(label_path):
    """检查标注文件中是否已包含 person 标注"""
    if not os.path.exists(label_path):
        return False

    with open(label_path, "r") as f:
        for line in f:
            if line.startswith("2 "):
                return True
    return False


def main():
    print("=" * 60)
    print("🚀 开始自动标注人员到训练数据集")
    print("=" * 60)

    # 检查目录
    if not os.path.exists(IMAGES_DIR):
        print(f"❌ 错误: 图片目录不存在 {IMAGES_DIR}")
        return

    os.makedirs(LABELS_DIR, exist_ok=True)

    # 加载模型
    print(f"\n📦 加载模型: {MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    # 获取所有图片文件
    image_files = sorted(
        list(Path(IMAGES_DIR).glob("*.jpg"))
        + list(Path(IMAGES_DIR).glob("*.png"))
        + list(Path(IMAGES_DIR).glob("*.jpeg"))
    )

    total_images = len(image_files)
    print(f"📊 找到 {total_images} 张图片")

    if total_images == 0:
        print("❌ 没有找到图片文件")
        return

    # 统计变量
    processed = 0
    skipped = 0
    added_person_count = 0
    total_persons = 0

    print("\n" + "=" * 60)
    print("🔍 开始处理图片...")
    print("=" * 60 + "\n")

    for idx, img_path in enumerate(image_files):
        # 获取对应的标注文件路径
        label_filename = img_path.stem + ".txt"
        label_path = os.path.join(LABELS_DIR, label_filename)

        # 检查是否跳过
        if SKIP_EXISTING and os.path.exists(label_path):
            if has_person_annotation(label_path):
                skipped += 1
                if (idx + 1) % 50 == 0:
                    print(f"⏭️  进度: {idx + 1}/{total_images} (已跳过 {skipped} 张)")
                continue

        # 读取图片
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"⚠️  警告: 无法读取图片 {img_path}")
            continue

        img_height, img_width = img.shape[:2]

        # 检测人员
        results = model.predict(
            img, conf=CONF_THRES_PERSON, device=DEVICE, verbose=False, classes=[0]
        )

        # 提取人员检测结果 (COCO 数据集中 person 的 class_id 是 0)
        person_detections = []
        if results[0].boxes is not None:
            boxes = results[0].boxes
            coords = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()

            for i, conf in enumerate(confs):
                if int(clss[i]) == 0:  # person class in COCO
                    x1, y1, x2, y2 = coords[i]
                    person_detections.append((x1, y1, x2, y2, conf))

        # 添加到标注文件
        if person_detections:
            add_person_to_label(label_path, person_detections, img_width, img_height)
            added_person_count += len(person_detections)
            total_persons += len(person_detections)
            processed += 1

            if processed % 10 == 0:
                print(
                    f"✅ [{idx + 1:3d}/{total_images}] {img_path.name} -> 添加 {len(person_detections)} 个人员"
                )
        else:
            if (idx + 1) % 50 == 0:
                print(
                    f"📷 [{idx + 1:3d}/{total_images}] {img_path.name} -> 未检测到人员"
                )

    # 输出统计
    print("\n" + "=" * 60)
    print("🎉 处理完成！")
    print("=" * 60)
    print(f"📊 总图片数: {total_images}")
    print(f"✅ 已处理: {processed}")
    print(f"⏭️  已跳过: {skipped}")
    print(f"👤 总添加人员数: {added_person_count}")
    print(
        f"📈 平均每张图片添加: {added_person_count / processed:.2f} 个人员"
        if processed > 0
        else ""
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
