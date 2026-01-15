# -*- coding: utf-8 -*-
"""调试脚本：可视化检测结果，帮助诊断问题"""
import cv2
from ultralytics import YOLO
import os

# 配置
MODEL_PATH = "runs/detect/runs/train/yolo11n_640_train_hd/weights/best.pt"
VIDEO_PATH = "/Users/grifftwu/ball/test2.mp4"
OUTPUT_VIDEO = "./debug_output.mp4"
CONF_THRES_BALL = 0.15
CONF_THRES_RIM = 0.40

# 加载模型
print("📦 加载模型...")
model = YOLO(MODEL_PATH)

# 打开视频
cap = cv2.VideoCapture(VIDEO_PATH)
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"📹 视频信息: {width}x{height} @ {fps}fps, 总帧数: {total_frames}")

# 创建输出视频
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (width, height))

# 统计信息
ball_count = 0
rim_count = 0
frame_idx = 0
check_frames = 300  # 只检查前300帧（约10秒）

print(f"🔍 开始检测前 {check_frames} 帧...")

while frame_idx < check_frames:
    ret, frame = cap.read()
    if not ret:
        break
    
    # 运行检测
    results = model.predict(frame, conf=0.1, verbose=False, imgsz=640)
    
    if results[0].boxes is not None:
        boxes = results[0].boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 绘制检测框
            if cls == 0:  # 篮球
                if conf > CONF_THRES_BALL:
                    ball_count += 1
                    color = (0, 255, 0)  # 绿色
                    label = f"Ball {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            elif cls == 1:  # 篮筐
                if conf > CONF_THRES_RIM:
                    rim_count += 1
                    color = (255, 0, 0)  # 蓝色
                    label = f"Rim {conf:.2f}"
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    # 绘制三个区域（基于检测到的篮筐）
                    # 高位区线
                    high_line = y1 - 150
                    cv2.line(frame, (0, high_line), (width, high_line), (0, 255, 255), 2)
                    cv2.putText(frame, "High Zone", (10, high_line-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
                    # 触框区
                    cv2.rectangle(frame, (x1-10, y1-10), (x2+10, y2+10), (255, 255, 0), 2)
                    
                    # 进球区
                    goal_y2 = min(y2 + 150, height)
                    cv2.rectangle(frame, (x1-30, y1+10), (x2+30, goal_y2), (255, 0, 255), 2)
                    cv2.putText(frame, "Goal Zone", (x1, goal_y2-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
    
    # 添加帧信息
    cv2.putText(frame, f"Frame: {frame_idx}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Ball: {ball_count} | Rim: {rim_count}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    out.write(frame)
    frame_idx += 1
    
    if frame_idx % 30 == 0:
        print(f"  处理到第 {frame_idx} 帧 | 篮球检测: {ball_count} 次 | 篮筐检测: {rim_count} 次")

cap.release()
out.release()

print(f"\n✅ 完成！调试视频已保存: {OUTPUT_VIDEO}")
print(f"📊 统计结果:")
print(f"  - 篮球检测次数: {ball_count}")
print(f"  - 篮筐检测次数: {rim_count}")
print(f"\n🔍 诊断:")
if ball_count == 0:
    print("  ❌ 没有检测到篮球！可能原因：")
    print("     1. 模型对篮球检测效果不好")
    print("     2. 置信度阈值太高 (当前 0.15)")
    print("     3. 视频中篮球太小或模糊")
elif rim_count == 0:
    print("  ❌ 没有检测到篮筐！可能原因：")
    print("     1. 模型对篮筐检测效果不好")
    print("     2. 置信度阈值太高 (当前 0.40)")
    print("     3. 篮筐不在画面中或被遮挡")
else:
    print("  ✅ 篮球和篮筐都能检测到")
    print("  💡 如果仍然检测不到进球，可能是:")
    print("     1. 区域参数需要调整 (HIGH_ZONE_OFFSET, GOAL_ZONE_OFFSET)")
    print("     2. 时间窗口太短 (SHOT_WINDOW 当前 2.5秒)")
    print("     3. 球的运动轨迹不符合判定逻辑")
