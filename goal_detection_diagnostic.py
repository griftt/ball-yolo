# -*- coding: utf-8 -*-
"""
进球检测诊断工具
用于排查为什么检测不到进球
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque

# ==================== ⚙️ 配置 ====================
VIDEO_PATH = "/Users/grifftwu/Desktop/历史篮球/20260304/333.MP4"
# MODEL_PATH = "runs/yolo26s/best.pt"
MODEL_PATH = "runs/detect/runs/train/yolo26s_640_train_hd_person/weights/best.pt"
START_MIN = 0

# ==================== 🎯 检测参数 ====================
class DetectionConfig:
    CONF_THRESHOLD_BALL = 0.15
    CONF_THRESHOLD_RIM = 0.15
    
    CALIBRATION_FRAMES = 30
    
    TARGET_ZONE_HEIGHT = 0.06
    TARGET_ZONE_BELOW_HEIGHT = 0.08
    TARGET_ZONE_H_EXPANSION = 0.01
    INTERACTION_DISTANCE = 0.20
    EXPANSION_FACTOR = 0.10
    
    EVENT_WINDOW = 2.5
    EVENT_COOLDOWN = 3.0
    
    REQUIRE_BELOW_ZONE = True  # 严格模式

config = DetectionConfig()

# ==================== 📊 诊断状态 ====================
class DiagnosticState:
    def __init__(self):
        self.is_calibrated = False
        self.calibration_buffer = []
        self.rim_box = None
        self.target_zone = None
        self.below_target_zone = None
        
        self.last_interaction_time = -10.0
        self.last_event_time = -10.0
        
        self.frame_count = 0
        self.rim_detected_frames = 0
        self.ball_detected_frames = 0
        
        # 诊断信息
        self.ball_in_below_zone_frames = 0
        self.ball_in_upper_zone_frames = 0
        self.interaction_frames = 0
        self.failed_attempts = []  # 记录失败的尝试

state = DiagnosticState()

# ==================== 🔧 辅助函数 ====================

def normalize_box(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    return {
        'x': x1 / frame_width,
        'y': y1 / frame_height,
        'width': (x2 - x1) / frame_width,
        'height': (y2 - y1) / frame_height,
        'centerX': ((x1 + x2) / 2) / frame_width,
        'centerY': ((y1 + y2) / 2) / frame_height,
        'minX': x1 / frame_width,
        'minY': y1 / frame_height,
        'maxX': x2 / frame_width,
        'maxY': y2 / frame_height,
        'midX': ((x1 + x2) / 2) / frame_width,
        'midY': ((y1 + y2) / 2) / frame_height,
    }

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def process_calibration(rim_box_norm):
    state.calibration_buffer.append(rim_box_norm)
    
    if len(state.calibration_buffer) < config.CALIBRATION_FRAMES:
        return f"校准中 {len(state.calibration_buffer)}/{config.CALIBRATION_FRAMES}"
    
    avg_x = np.mean([b['x'] for b in state.calibration_buffer])
    avg_y = np.mean([b['y'] for b in state.calibration_buffer])
    avg_w = np.mean([b['width'] for b in state.calibration_buffer])
    avg_h = np.mean([b['height'] for b in state.calibration_buffer])
    
    state.rim_box = {
        'x': avg_x, 'y': avg_y, 'width': avg_w, 'height': avg_h,
        'minX': avg_x, 'minY': avg_y,
        'maxX': avg_x + avg_w, 'maxY': avg_y + avg_h,
        'midX': avg_x + avg_w / 2, 'midY': avg_y + avg_h / 2,
    }
    
    state.target_zone = {
        'x': state.rim_box['minX'] - config.TARGET_ZONE_H_EXPANSION,
        'y': state.rim_box['minY'] - config.TARGET_ZONE_HEIGHT,
        'width': state.rim_box['width'] + (config.TARGET_ZONE_H_EXPANSION * 2),
        'height': config.TARGET_ZONE_HEIGHT
    }
    
    state.below_target_zone = {
        'x': state.rim_box['minX'] - config.TARGET_ZONE_H_EXPANSION,
        'y': state.rim_box['maxY'],
        'width': state.rim_box['width'] + (config.TARGET_ZONE_H_EXPANSION * 2),
        'height': config.TARGET_ZONE_BELOW_HEIGHT
    }
    
    state.is_calibrated = True
    state.calibration_buffer.clear()
    
    print(f"\n✅ 校准完成")
    print(f"   篮筐位置: ({state.rim_box['midX']:.3f}, {state.rim_box['midY']:.3f})")
    print(f"   上方区域: y={state.target_zone['y']:.3f} ~ {state.target_zone['y'] + state.target_zone['height']:.3f}")
    print(f"   下方区域: y={state.below_target_zone['y']:.3f} ~ {state.below_target_zone['y'] + state.below_target_zone['height']:.3f}\n")
    
    return "✅ 校准完成"

def diagnose_detection(ball_box_norm, timestamp):
    """诊断为什么没有检测到进球"""
    
    # 冷却时间检查
    if timestamp - state.last_event_time < config.EVENT_COOLDOWN:
        return None, "冷却中", {}
    
    ball_center = (ball_box_norm['centerX'], ball_box_norm['centerY'])
    
    # 检测交互
    has_interaction = False
    interaction_type = []
    
    rim_center = (state.rim_box['midX'], state.rim_box['midY'])
    distance = calculate_distance(ball_center, rim_center)
    
    if distance < config.INTERACTION_DISTANCE:
        state.last_interaction_time = timestamp
        has_interaction = True
        interaction_type.append(f"距离={distance:.3f}")
        state.interaction_frames += 1
    
    expanded_rect = {
        'x': state.rim_box['minX'] - config.EXPANSION_FACTOR,
        'y': state.rim_box['minY'] - config.EXPANSION_FACTOR,
        'width': state.rim_box['width'] + (config.EXPANSION_FACTOR * 2),
        'height': state.rim_box['height'] + (config.EXPANSION_FACTOR * 2)
    }
    
    if (expanded_rect['x'] <= ball_center[0] <= expanded_rect['x'] + expanded_rect['width'] and
        expanded_rect['y'] <= ball_center[1] <= expanded_rect['y'] + expanded_rect['height']):
        state.last_interaction_time = timestamp
        has_interaction = True
        interaction_type.append("扩展区域")
        state.interaction_frames += 1
    
    # 检测区域
    in_upper_zone = False
    if (state.target_zone['x'] <= ball_center[0] <= state.target_zone['x'] + state.target_zone['width'] and
        state.target_zone['y'] <= ball_center[1] <= state.target_zone['y'] + state.target_zone['height']):
        in_upper_zone = True
        state.ball_in_upper_zone_frames += 1
    
    in_below_zone = False
    if (state.below_target_zone['x'] <= ball_center[0] <= state.below_target_zone['x'] + state.below_target_zone['width'] and
        state.below_target_zone['y'] <= ball_center[1] <= state.below_target_zone['y'] + state.below_target_zone['height']):
        in_below_zone = True
        state.ball_in_below_zone_frames += 1
    
    # 诊断信息
    diagnostic = {
        'has_interaction': has_interaction,
        'interaction_type': interaction_type,
        'in_upper_zone': in_upper_zone,
        'in_below_zone': in_below_zone,
        'distance': distance,
        'ball_y': ball_center[1],
        'rim_y': state.rim_box['midY'],
        'below_zone_y_range': (state.below_target_zone['y'], 
                               state.below_target_zone['y'] + state.below_target_zone['height'])
    }
    
    # 判定
    in_valid_zone = in_below_zone if config.REQUIRE_BELOW_ZONE else (in_below_zone or in_upper_zone)
    
    if in_valid_zone and has_interaction:
        time_diff = abs(timestamp - state.last_interaction_time)
        
        if time_diff <= config.EVENT_WINDOW:
            state.last_event_time = timestamp
            zone_type = "下方区域" if in_below_zone else "上方区域"
            return True, f"🎉 进球! {zone_type}", diagnostic
        else:
            # 记录失败尝试
            state.failed_attempts.append({
                'timestamp': timestamp,
                'reason': f"时间窗口超时: {time_diff:.2f}s > {config.EVENT_WINDOW}s",
                'diagnostic': diagnostic
            })
            return None, f"时间窗口超时: {time_diff:.2f}s", diagnostic
    else:
        # 记录失败原因
        reasons = []
        if not in_valid_zone:
            if config.REQUIRE_BELOW_ZONE:
                reasons.append(f"❌ 不在下方区域 (球Y={ball_center[1]:.3f}, 下方区域Y={state.below_target_zone['y']:.3f}~{state.below_target_zone['y'] + state.below_target_zone['height']:.3f})")
            else:
                reasons.append("❌ 不在有效区域")
        if not has_interaction:
            reasons.append(f"❌ 无交互 (距离={distance:.3f} > {config.INTERACTION_DISTANCE})")
        
        if reasons:
            state.failed_attempts.append({
                'timestamp': timestamp,
                'reason': ', '.join(reasons),
                'diagnostic': diagnostic
            })
        
        status = f"交互={'✅' if has_interaction else '❌'} 上方={'✅' if in_upper_zone else '❌'} 下方={'✅' if in_below_zone else '❌'}"
        return None, status, diagnostic

def denormalize_rect(rect, frame_width, frame_height):
    x1 = int(rect['x'] * frame_width)
    y1 = int(rect['y'] * frame_height)
    x2 = int((rect['x'] + rect['width']) * frame_width)
    y2 = int((rect['y'] + rect['height']) * frame_height)
    return (x1, y1, x2, y2)

def draw_zones(frame, frame_width, frame_height):
    if not state.is_calibrated:
        return
    
    # 篮筐
    rim_pixel = denormalize_rect(state.rim_box, frame_width, frame_height)
    cv2.rectangle(frame, (rim_pixel[0], rim_pixel[1]), (rim_pixel[2], rim_pixel[3]), (0, 255, 0), 3)
    
    # 上方区域
    upper_pixel = denormalize_rect(state.target_zone, frame_width, frame_height)
    cv2.rectangle(frame, (upper_pixel[0], upper_pixel[1]), (upper_pixel[2], upper_pixel[3]), (255, 0, 0), 2)
    
    # 下方区域（加粗显示）
    below_pixel = denormalize_rect(state.below_target_zone, frame_width, frame_height)
    cv2.rectangle(frame, (below_pixel[0], below_pixel[1]), (below_pixel[2], below_pixel[3]), (0, 0, 255), 4)
    cv2.putText(frame, "BELOW ZONE (REQUIRED)", (below_pixel[0], below_pixel[1]+30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

def draw_diagnostic_panel(frame, timestamp, diagnostic):
    h, w = frame.shape[:2]
    
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (1200, 700), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    y = 50
    line_h = 40
    
    cv2.putText(frame, f"DIAGNOSTIC MODE - Time: {timestamp:.1f}s", 
                (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    y += line_h + 10
    
    # 校准状态
    if state.is_calibrated:
        cv2.putText(frame, "Status: CALIBRATED", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    else:
        cv2.putText(frame, f"Status: Calibrating {len(state.calibration_buffer)}/{config.CALIBRATION_FRAMES}", 
                    (30, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
    y += line_h
    
    # 统计
    cv2.putText(frame, f"Frames: {state.frame_count} | Rim: {state.rim_detected_frames} | Ball: {state.ball_detected_frames}", 
                (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    y += line_h
    
    # 关键统计
    cv2.putText(frame, f"Ball in Below Zone: {state.ball_in_below_zone_frames} frames", 
                (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
    y += line_h
    
    cv2.putText(frame, f"Ball in Upper Zone: {state.ball_in_upper_zone_frames} frames", 
                (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
    y += line_h
    
    cv2.putText(frame, f"Interaction Frames: {state.interaction_frames}", 
                (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 255), 2)
    y += line_h + 10
    
    # 当前帧诊断
    if diagnostic:
        cv2.putText(frame, "Current Frame:", (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2)
        y += line_h
        
        color = (0, 255, 0) if diagnostic['has_interaction'] else (0, 0, 255)
        cv2.putText(frame, f"  Interaction: {'YES' if diagnostic['has_interaction'] else 'NO'} {diagnostic['interaction_type']}", 
                    (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y += 35
        
        color = (0, 0, 255) if diagnostic['in_below_zone'] else (128, 128, 128)
        cv2.putText(frame, f"  Below Zone: {'YES' if diagnostic['in_below_zone'] else 'NO'}", 
                    (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y += 35
        
        color = (255, 0, 0) if diagnostic['in_upper_zone'] else (128, 128, 128)
        cv2.putText(frame, f"  Upper Zone: {'YES' if diagnostic['in_upper_zone'] else 'NO'}", 
                    (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        y += 35
        
        cv2.putText(frame, f"  Ball Y: {diagnostic['ball_y']:.3f}", 
                    (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y += 35
        
        cv2.putText(frame, f"  Below Zone Y: {diagnostic['below_zone_y_range'][0]:.3f} ~ {diagnostic['below_zone_y_range'][1]:.3f}", 
                    (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        y += 35
        
        cv2.putText(frame, f"  Distance: {diagnostic['distance']:.3f} (threshold: {config.INTERACTION_DISTANCE})", 
                    (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # 配置
    y = h - 150
    cv2.putText(frame, f"Config: REQUIRE_BELOW_ZONE={config.REQUIRE_BELOW_ZONE}", 
                (30, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

def run_diagnostic():
    print(f"📦 加载模型: {MODEL_PATH}")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    start_frame = int(START_MIN * 60 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print(f"\n🔍 诊断模式启动")
    print(f"   严格模式: {config.REQUIRE_BELOW_ZONE}")
    print(f"   下方区域高度: {config.TARGET_ZONE_BELOW_HEIGHT}")
    print(f"   交互距离: {config.INTERACTION_DISTANCE}")
    print("=" * 60)
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✅ 诊断完成")
                print(f"\n📊 统计:")
                print(f"   总帧数: {state.frame_count}")
                print(f"   球在下方区域帧数: {state.ball_in_below_zone_frames}")
                print(f"   球在上方区域帧数: {state.ball_in_upper_zone_frames}")
                print(f"   交互帧数: {state.interaction_frames}")
                print(f"\n❌ 失败尝试: {len(state.failed_attempts)}")
                if state.failed_attempts:
                    print("\n最近的失败尝试:")
                    for attempt in state.failed_attempts[-5:]:
                        print(f"   时间={attempt['timestamp']:.2f}s: {attempt['reason']}")
                break
            
            state.frame_count += 1
        
        debug_frame = frame.copy()
        timestamp = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
        
        results = model.predict(debug_frame, conf=0.01, device=device, verbose=False, imgsz=1024)
        
        diagnostic = None
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            coords = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            
            for i, conf in enumerate(confs):
                cls_id = int(clss[i])
                x1, y1, x2, y2 = map(int, coords[i])
                
                if cls_id == 1 and conf > config.CONF_THRESHOLD_RIM:
                    state.rim_detected_frames += 1
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    if not state.is_calibrated:
                        rim_box_norm = normalize_box([x1, y1, x2, y2], frame_width, frame_height)
                        process_calibration(rim_box_norm)
                
                elif cls_id == 0 and conf > config.CONF_THRESHOLD_BALL:
                    state.ball_detected_frames += 1
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 140, 255), 3)
                    
                    if state.is_calibrated:
                        ball_box_norm = normalize_box([x1, y1, x2, y2], frame_width, frame_height)
                        goal, msg, diagnostic = diagnose_detection(ball_box_norm, timestamp)
                        
                        if goal:
                            print(f"\n🎉 [Frame {state.frame_count}] 检测到进球! 时间={timestamp:.2f}s")
        
        draw_zones(debug_frame, frame_width, frame_height)
        draw_diagnostic_panel(debug_frame, timestamp, diagnostic)
        
        cv2.imshow('Goal Detection Diagnostic', debug_frame)
        
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:
            paused = not paused
        elif key == ord('f'):
            paused = True
            ret, frame = cap.read()
        elif key == ord('d'):
            curr_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_pos + (5 * fps))
        elif key == ord('a'):
            curr_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, curr_pos - (5 * fps)))
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_diagnostic()
