# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ==================== ⚙️ 调试配置 (修改这里) ====================
# VIDEO_PATH = "/Users/grifftwu/Desktop/历史篮球/1122/check.mp4"
VIDEO_PATH = "/Users/grifftwu/Desktop/历史篮球/20260304/111.MP4"
# MODEL_PATH = "./runs/train/yolo11n_640_train/weights/best.pt"
# MODEL_PATH = "runs/detect/runs/train/yolo11n_640_train_hd/weights/best.pt"
# MODEL_PATH = "runs/detect/runs/train/yolo26s_640_train_hd_person/weights/best.pt"
# MODEL_PATH = "runs/detect/runs/train/yolo26s_640_train_hd_person/weights/best.pt"
MODEL_PATH ="runs/yolo26s/best.pt"
# VIDEO_PATH = "/Users/grifftwu/ball/test1.mp4"
OUTPUT_DIR = "./outputs/auto_clips_202600304check"

# ⏱️ [这里修改] 从第几分钟开始看？
START_MIN = 0

# 🔍 阈值设置 (保持和你主程序一致)
CONF_THRES_BALL = 0.15   
CONF_THRES_RIM = 0.10    

# 📏 区域参数 (画出来给你看)
HIGH_ZONE_OFFSET = 150   # 蓝线 (高空线)
GOAL_ZONE_OFFSET = 150   # 红框 (得分区深度)
# =============================================================

def run_debug():
    print(f"📦 加载模型: {MODEL_PATH}")
    device = 'mps' if torch.backends.mps.is_available() and not MODEL_PATH.endswith(".mlpackage") else 'cpu'
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 🟢 1. 跳转到指定时间
    start_frame = int(START_MIN * 60 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print(f"\n🚀以此开始播放: {START_MIN}分 ({start_frame}/{total_frames}帧)")
    print("------------------------------------------------")
    print("⌨️  快捷键说明:")
    print("   [空格]  暂停/继续")
    print("   [F]     下一帧 (暂停时用)")
    print("   [D]     快进 5秒 ⏩")
    print("   [A]     快退 5秒 ⏪")
    print("   [Q]     退出")
    print("------------------------------------------------")

    paused = False
    frame_count = 0
    rim_detected_frames = 0
    ball_detected_frames = 0
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret: 
                print("\n视频播放结束")
                print(f"📊 检测统计: 总帧数={frame_count}, 篮筐检测帧={rim_detected_frames}, 篮球检测帧={ball_detected_frames}")
                break
            frame_count += 1
        else:
            # 暂停时重复显示当前帧(为了保持窗口响应)
            pass

        # 复制画面用于绘图
        debug_frame = frame.copy()
        
        # --- YOLO 推理 ---
        results = model.predict(debug_frame, conf=0.01, device=device, verbose=False, imgsz=1024)
        
        rim_box = None
        has_ball_this_frame = False
        has_rim_this_frame = False
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            coords = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            
            for i, conf in enumerate(confs):
                cls_id = int(clss[i])
                x1, y1, x2, y2 = map(int, coords[i])
                
                # 🏀 篮球
                if cls_id == 0:
                    if conf > CONF_THRES_BALL:
                        has_ball_this_frame = True
                        color = (0, 140, 255) # 橙色
                        label = f"Ball {conf:.2f}"
                        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(debug_frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        curr_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
                        print(f"🥅 [帧{frame_count:05d}] {curr_time:.2f}s - 检测到篮球! 置信度={conf:.3f}, 位置=[{x1},{y1},{x2},{y2}]")
                    else:
                        color = (200, 200, 200) # 灰色(被过滤)
                        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 1)
                        cv2.putText(debug_frame, f"{conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
                        curr_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
                        print(f"🥅 [帧{frame_count:05d}] {curr_time:.2f}s - 篮球置信度过低! 置信度={conf:.3f}, 位置=[{x1},{y1},{x2},{y2}]")
                        
                # 🥅 篮筐
                elif cls_id == 1:
                    if conf > CONF_THRES_RIM:
                        has_rim_this_frame = True
                        color = (0, 255, 0) # 绿色
                        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                        if rim_box is None: 
                            rim_box = [x1, y1, x2, y2]
                            curr_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
                            print(f"🥅 [帧{frame_count:05d}] {curr_time:.2f}s - 检测到篮筐! 置信度={conf:.3f}, 位置=[{x1},{y1},{x2},{y2}]")
                    else:
                        # 打印低置信度的篮筐
                        if conf > 0.05:  # 只打印置信度>5%的
                            curr_time = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
                            print(f"⚠️  [帧{frame_count:05d}] {curr_time:.2f}s - 篮筐置信度过低: {conf:.3f} < {CONF_THRES_RIM}")
        
        # 统计检测成功的帧
        if has_rim_this_frame:
            rim_detected_frames += 1
        if has_ball_this_frame:
            ball_detected_frames += 1
        
        # --- 🎨 画区域 ---
        if rim_box is not None:
            rx1, ry1, rx2, ry2 = rim_box
            # 蓝线 (高空线)
            cv2.line(debug_frame, (0, ry1), (debug_frame.shape[1], ry1), (255, 0, 0), 2)
            # 黄框 (接触区)
            cv2.rectangle(debug_frame, (rx1-10, ry1-10), (rx2+10, ry2+10), (0, 255, 255), 1)
            # 红框 (得分区)
            gx1, gy1, gx2, gy2 = rx1 - 30, ry1 + 10, rx2 + 30, ry2 + GOAL_ZONE_OFFSET
            cv2.rectangle(debug_frame, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)

        # 缩放显示
        display_h = 800
        scale = display_h / debug_frame.shape[0]
        display_w = int(debug_frame.shape[1] * scale)
        small_frame = cv2.resize(debug_frame, (display_w, display_h))
        
        # 叠加时间和统计信息
        curr_sec = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
        cv2.putText(small_frame, f"Time: {curr_sec/60:.2f} min", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(small_frame, f"Rim: {rim_detected_frames}/{frame_count}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(small_frame, f"Ball: {ball_detected_frames}/{frame_count}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 255), 2)
        
        cv2.imshow('YOLO Inspector', small_frame)
        
        # --- 🕹️ 键盘控制 ---
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        
        if key == ord('q'): # 退出
            break
        elif key == 32: # 空格暂停
            paused = not paused
        elif key == ord('f'): # F键下一帧
            paused = True 
            ret, frame = cap.read()
        elif key == ord('d'): # D键 快进 5秒
            curr_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_pos + (5 * fps))
            print("⏩ 快进 5秒")
        elif key == ord('a'): # A键 快退 5秒
            curr_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, curr_pos - (5 * fps)))
            print("⏪ 快退 5秒")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_debug()