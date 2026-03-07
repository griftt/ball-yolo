# -*- coding: utf-8 -*-
"""
验证标注文件的可视化工具
遍历训练图片，读取标注文件，在图片上绘制标注框供人工检查
"""

import os
import cv2
import numpy as np
from pathlib import Path

# ==================== 配置区域 ====================

# 数据集路径
DATASET_DIR = "/Users/grifftwu/Desktop/历史篮球/20260111bak2"
IMAGES_DIR = os.path.join(DATASET_DIR, "images/train")
LABELS_DIR = os.path.join(DATASET_DIR, "labels/train")

# 类别定义（顺序要和训练数据一致）
CLASS_NAMES = ["basketball", "rim", "person"]

# 类别颜色 (BGR 格式)
CLASS_COLORS = {
    0: (0, 140, 255),  # 篮球 - 橙色
    1: (0, 255, 0),  # 篮筐 - 绿色
    2: (255, 0, 255),  # 人员 - 紫色
}

# 显示高度
DISPLAY_HEIGHT = 800

# =============================================================


def parse_yolo_label(label_path, img_width, img_height):
    """
    解析 YOLO 格式的标注文件

    Args:
        label_path: 标注文件路径
        img_width: 图片宽度
        img_height: 图片高度

    Returns:
        list: [(class_id, x1, y1, x2, y2), ...]
    """
    if not os.path.exists(label_path):
        return []

    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # 转换为像素坐标
                x1 = int((x_center - width / 2) * img_width)
                y1 = int((y_center - height / 2) * img_height)
                x2 = int((x_center + width / 2) * img_width)
                y2 = int((y_center + height / 2) * img_height)

                boxes.append((class_id, x1, y1, x2, y2))

    return boxes


def draw_labels(img, boxes):
    """
    在图片上绘制标注框

    Args:
        img: 图片 (numpy array)
        boxes: 标注框列表 [(class_id, x1, y1, x2, y2), ...]

    Returns:
        绘制后的图片
    """
    result = img.copy()

    for class_id, x1, y1, x2, y2 in boxes:
        color = CLASS_COLORS.get(class_id, (255, 255, 255))
        class_name = (
            CLASS_NAMES[class_id]
            if class_id < len(CLASS_NAMES)
            else f"class_{class_id}"
        )

        # 绘制矩形框
        cv2.rectangle(result, (x1, y1), (x2, y2), color, 2)

        # 绘制标签
        label = class_name
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)

        # 确保标签不会超出图片边界
        label_x = max(x1, 0)
        label_y = max(y1 - 10, label_h)

        # 绘制标签背景
        cv2.rectangle(
            result,
            (label_x, label_y - label_h),
            (label_x + label_w, label_y + 5),
            color,
            -1,
        )

        # 绘制标签文字
        cv2.putText(
            result,
            label,
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

    return result


def main():
    print("=" * 60)
    print("🔍 标注验证工具")
    print("=" * 60)

    # 检查目录
    if not os.path.exists(IMAGES_DIR):
        print(f"❌ 错误: 图片目录不存在 {IMAGES_DIR}")
        return

    # 获取所有图片文件
    image_files = sorted(
        list(Path(IMAGES_DIR).glob("*.jpg"))
        + list(Path(IMAGES_DIR).glob("*.png"))
        + list(Path(IMAGES_DIR).glob("*.jpeg"))
    )

    total_images = len(image_files)
    print(f"📊 找到 {total_images} 张图片\n")

    if total_images == 0:
        print("❌ 没有找到图片文件")
        return

    # 获取有标注的图片
    annotated_images = []
    for img_path in image_files:
        label_path = os.path.join(LABELS_DIR, img_path.stem + ".txt")
        if os.path.exists(label_path):
            annotated_images.append((img_path, label_path))

    print(f"✅ 有标注的图片: {len(annotated_images)} 张\n")

    if len(annotated_images) == 0:
        print("❌ 没有找到标注文件")
        return

    # 快捷键说明
    print("=" * 60)
    print("⌨️  快捷键说明:")
    print("   [空格]  暂停/继续自动播放")
    print("   [N]     下一张")
    print("   [P]     上一张")
    print("   [J]     跳转到指定编号")
    print("   [S]     保存当前标注的图片")
    print("   [D]     删除当前标注文件 (⚠️ 谨慎使用)")
    print("   [Q]     退出")
    print("=" * 60 + "\n")

    # 开始循环
    idx = 0
    auto_play = True
    delay = 500  # 自动播放延迟（毫秒）

    while True:
        img_path, label_path = annotated_images[idx]
        img = cv2.imread(str(img_path))

        if img is None:
            print(f"⚠️  无法读取图片: {img_path.name}")
            idx = (idx + 1) % len(annotated_images)
            continue

        img_height, img_width = img.shape[:2]

        # 解析标注
        boxes = parse_yolo_label(label_path, img_width, img_height)

        # 绘制标注
        annotated_img = draw_labels(img, boxes)

        # 缩放显示
        scale = DISPLAY_HEIGHT / img_height
        display_w = int(img_width * scale)
        display_h = int(img_height * scale)
        display_img = cv2.resize(annotated_img, (display_w, display_h))

        # 统计信息
        class_counts = {}
        for class_id, _, _, _, _ in boxes:
            class_counts[class_id] = class_counts.get(class_id, 0) + 1

        # 叠加信息
        info_y = 40
        cv2.putText(
            display_img,
            f"[{idx + 1}/{len(annotated_images)}] {img_path.name}",
            (20, info_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        info_y += 30
        for class_id in sorted(class_counts.keys()):
            class_name = (
                CLASS_NAMES[class_id]
                if class_id < len(CLASS_NAMES)
                else f"class_{class_id}"
            )
            count = class_counts[class_id]
            color = CLASS_COLORS.get(class_id, (255, 255, 255))
            cv2.putText(
                display_img,
                f"{class_name}: {count}",
                (20, info_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                color,
                2,
            )
            info_y += 25

        # 显示图片
        cv2.imshow("Label Validator", display_img)

        # 等待按键
        key = cv2.waitKey(delay if auto_play else 0) & 0xFF

        if key == ord("q"):  # 退出
            break
        elif key == 32:  # 空格 - 暂停/继续
            auto_play = not auto_play
            status = "▶️  自动播放" if auto_play else "⏸️  已暂停"
            print(f"   {status}")
        elif key == ord("n"):  # 下一张
            idx = (idx + 1) % len(annotated_images)
            auto_play = False
        elif key == ord("p"):  # 上一张
            idx = (idx - 1) % len(annotated_images)
            auto_play = False
        elif key == ord("j"):  # 跳转
            try:
                jump_idx = int(input(f"\n跳转到 (1-{len(annotated_images)}): ")) - 1
                if 0 <= jump_idx < len(annotated_images):
                    idx = jump_idx
                    auto_play = False
                    print(f"   ✅ 跳转到第 {idx + 1} 张")
                else:
                    print(f"   ❌ 无效的编号")
            except ValueError:
                print(f"   ❌ 请输入有效的数字")
        elif key == ord("s"):  # 保存
            save_dir = os.path.join(DATASET_DIR, "validated_labels")
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, img_path.name)
            cv2.imwrite(save_path, display_img)
            print(f"   💾 已保存到: {save_path}")
        elif key == ord("d"):  # 删除标注
            response = input(f"\n⚠️  确定要删除标注文件吗? {label_path} (y/N): ")
            if response.lower() == "y":
                os.remove(label_path)
                print(f"   🗑️  已删除: {label_path}")
                annotated_images.pop(idx)
                if idx >= len(annotated_images):
                    idx = 0
                auto_play = False

    cv2.destroyAllWindows()
    print("\n" + "=" * 60)
    print("👋 退出验证工具")
    print("=" * 60)


if __name__ == "__main__":
    main()
