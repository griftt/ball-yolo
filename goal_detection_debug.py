# -*- coding: utf-8 -*-
"""
进球检测逻辑可视化调试工具
基于 VideoAnalysisSDK 的检测逻辑
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque
import time

# ==================== ⚙️ 调试配置 ====================
VIDEO_PATH = "/Users/grifftwu/Desktop/历史篮球/20260304/111.MP4"
MODEL_PATH = "runs/yolo26s/best.pt"
OUTPUT_DIR = "./outputs/goal_detection_debug"
START_MIN = 10  # 从第几分钟开始

# ==================== 🎯 检测参数 (对应 VideoAnalysisConfig) ====================
class DetectionConfig:
    # 推理配置
    CONF_THRESHOLD_BALL = 0.15      # confidenceThreshold
    CONF_THRESHOLD_RIM = 0.15       # confidenceThreshold
    NMS_THRESHOLD = 0.45            # nmsThreshold
    
    # 校准参数
    CALIBRATION_FRAMES = 30         # calibrationFrames
    
    # 空间参数 (归一化坐标 0.0-1.0)
    TARGET_ZONE_HEIGHT = 0.06       # targetZoneHeight - 上方区域高度
    TARGET_ZONE_BELOW_HEIGHT = 0.08 # targetZoneBelowHeight - 下方区域高度
    TARGET_ZONE_H_EXPANSION = 0.01  # targetZoneHorizontalExpansion
    INTERACTION_DISTANCE = 0.20     # interactionDistanceThreshold
    EXPANSION_FACTOR = 0.10         # expansionFactor
    
    # 时间参数
    EVENT_WINDOW = 2.5              # eventWindow (秒)
    EVENT_COOLDOWN = 3.0            # eventCooldown (秒)
    
    # 标签配置
    RIM_LABELS = {1, "rim", "hoop", "basket", "class_1"}
    BALL_LABELS = {0, "ball", "basketball", "class_0"}

config = DetectionConfig()

# ==================== 📊 检测状态 ====================
class DetectionState:
    def __init__(self):
        # 校准状态
        self.is_calibrated = False
        self.calibration_buffer = []
        self.rim_box = None  # 锁定的篮筐位置 (归一化坐标)
        self.target_zone = None  # 上方检测区域
        self.below_target_zone = None  # 下方检测区域
        
        # 事件检测状态
        self.last_interaction_time = -10.0
        self.last_event_time = -10.0
        
        # 统计信息
        self.frame_count = 0
        self.rim_detected_frames = 0
        self.ball_detected_frames = 0
        self.goal_events = []
        
        # 可视化
        self.trajectory = deque(maxlen=30)  # 球的轨迹

state = DetectionState()

# ==================== 🔧 辅助函数 ====================

def normalize_box(box, frame_width, frame_height):
    """将像素坐标转换为归一化坐标 (0.0-1.0)"""
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

def denormalize_rect(rect, frame_width, frame_height):
    """将归一化矩形转换为像素坐标"""
    x1 = int(rect['x'] * frame_width)
    y1 = int(rect['y'] * frame_height)
    x2 = int((rect['x'] + rect['width']) * frame_width)
    y2 = int((rect['y'] + rect['height']) * frame_height)
    return (x1, y1, x2, y2)

def calculate_distance(p1, p2):
    """计算两点之间的欧氏距离"""
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

# ==================== 🎯 校准逻辑 ====================

def process_calibration(rim_box_norm):
    """处理校准逻辑"""
    state.calibration_buffer.append(rim_box_norm)
    
    if len(state.calibration_buffer) < config.CALIBRATION_FRAMES:
        return f"校准中 {len(state.calibration_buffer)}/{config.CALIBRATION_FRAMES}"
    
    # 计算平均位置
    avg_x = np.mean([b['x'] for b in state.calibration_buffer])
    avg_y = np.mean([b['y'] for b in state.calibration_buffer])
    avg_w = np.mean([b['width'] for b in state.calibration_buffer])
    avg_h = np.mean([b['height'] for b in state.calibration_buffer])
    
    # 锁定篮筐位置
    state.rim_box = {
        'x': avg_x,
        'y': avg_y,
        'width': avg_w,
        'height': avg_h,
        'minX': avg_x,
        'minY': avg_y,
        'maxX': avg_x + avg_w,
        'maxY': avg_y + avg_h,
        'midX': avg_x + avg_w / 2,
        'midY': avg_y + avg_h / 2,
    }
    
    # 计算上方目标区域
    state.target_zone = {
        'x': state.rim_box['minX'] - config.TARGET_ZONE_H_EXPANSION,
        'y': state.rim_box['minY'] - config.TARGET_ZONE_HEIGHT,
        'width': state.rim_box['width'] + (config.TARGET_ZONE_H_EXPANSION * 2),
        'height': config.TARGET_ZONE_HEIGHT
    }
    
    # 计算下方目标区域
    state.below_target_zone = {
        'x': state.rim_box['minX'] - config.TARGET_ZONE_H_EXPANSION,
        'y': state.rim_box['maxY'],  # 从篮筐底部开始
        'width': state.rim_box['width'] + (config.TARGET_ZONE_H_EXPANSION * 2),
        'height': config.TARGET_ZONE_BELOW_HEIGHT
    }
    
    state.is_calibrated = True
    state.calibration_buffer.clear()
    
    return "✅ 校准完成"

# ==================== 🎯 事件检测逻辑 ====================

def process_event_detection(ball_box_norm, timestamp):
    """处理事件检测逻辑"""
    # 冷却时间检查
    if timestamp - state.last_event_time < config.EVENT_COOLDOWN:
        return None, "冷却中"
    
    ball_center = (ball_box_norm['centerX'], ball_box_norm['centerY'])
    
    # 检测交互
    has_interaction = False
    interaction_details = []
    
    # 1. 距离检测
    rim_center = (state.rim_box['midX'], state.rim_box['midY'])
    distance = calculate_distance(ball_center, rim_center)
    
    if distance < config.INTERACTION_DISTANCE:
        state.last_interaction_time = timestamp
        has_interaction = True
        interaction_details.append(f"距离={distance:.3f}")
    
    # 2. 扩展区域检测
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
        interaction_details.append("扩展区域")
    
    # 3. 检测上方区域
    object_in_target_zone = False
    if (state.target_zone['x'] <= ball_center[0] <= state.target_zone['x'] + state.target_zone['width'] and
        state.target_zone['y'] <= ball_center[1] <= state.target_zone['y'] + state.target_zone['height']):
        object_in_target_zone = True
    
    # 4. 检测下方区域（关键）
    object_in_below_zone = False
    if (state.below_target_zone['x'] <= ball_center[0] <= state.below_target_zone['x'] + state.below_target_zone['width'] and
        state.below_target_zone['y'] <= ball_center[1] <= state.below_target_zone['y'] + state.below_target_zone['height']):
        object_in_below_zone = True
    
    # 状态信息
    status = f"交互={'✅' if has_interaction else '❌'} 上方={'✅' if object_in_target_zone else '❌'} 下方={'✅' if object_in_below_zone else '❌'}"
    
    # 进球判定
    in_valid_zone = object_in_below_zone or object_in_target_zone
    
    if in_valid_zone and has_interaction:
        time_diff = abs(timestamp - state.last_interaction_time)
        
        if time_diff <= config.EVENT_WINDOW:
            state.last_event_time = timestamp
            zone_type = "下方区域(高置信)" if object_in_below_zone else "上方区域(可能)"
            state.goal_events.append({
                'timestamp': timestamp,
                'frame': state.frame_count,
                'zone_type': zone_type,
                'interaction': ', '.join(interaction_details)
            })
            return True, f"🎉 进球! {zone_type}"
        else:
            return None, f"时间窗口不满足: {time_diff:.2f}s > {config.EVENT_WINDOW}s"
    
    return None, status

# ==================== 🎨 可视化函数 ====================

def draw_detection_zones(frame, frame_width, frame_height):
    """绘制检测区域"""
    if not state.is_calibrated:
        return
    
    # 绘制篮筐
    rim_pixel = denormalize_rect(state.rim_box, frame_width, frame_height)
    cv2.rectangle(frame, (rim_pixel[0], rim_pixel[1]), (rim_pixel[2], rim_pixel[3]), 
                  (0, 255, 0), 3)
    cv2.putText(frame, "RIM", (rim_pixel[0], rim_pixel[1]-10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # 绘制上方区域（蓝色）
    upper_pixel = denormalize_rect(state.target_zone, frame_width, frame_height)
    cv2.rectangle(frame, (upper_pixel[0], upper_pixel[1]), (upper_pixel[2], upper_pixel[3]), 
                  (255, 0, 0), 2)
    cv2.putText(frame, "UPPER ZONE", (upper_pixel[0], upper_pixel[1]-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    # 绘制下方区域（红色）- 关键区域
    below_pixel = denormalize_rect(state.below_target_zone, frame_width, frame_height)
    cv2.rectangle(frame, (below_pixel[0], below_pixel[1]), (below_pixel[2], below_pixel[3]), 
                  (0, 0, 255), 3)
    cv2.putText(frame, "BELOW ZONE (KEY)", (below_pixel[0], below_pixel[1]+20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
    # 绘制扩展区域（黄色虚线）
    expanded = {
        'x': state.rim_box['minX'] - config.EXPANSION_FACTOR,
        'y': state.rim_box['minY'] - config.EXPANSION_FACTOR,
        'width': state.rim_box['width'] + (config.EXPANSION_FACTOR * 2),
        'height': state.rim_box['height'] + (config.EXPANSION_FACTOR * 2)
    }
    exp_pixel = denormalize_rect(expanded, frame_width, frame_height)
    cv2.rectangle(frame, (exp_pixel[0], exp_pixel[1]), (exp_pixel[2], exp_pixel[3]), 
                  (0, 255, 255), 1, cv2.LINE_AA)
    
    # 绘制交互距离圆（紫色）
    rim_center_x = int(state.rim_box['midX'] * frame_width)
    rim_center_y = int(state.rim_box['midY'] * frame_height)
    radius = int(config.INTERACTION_DISTANCE * max(frame_width, frame_height))
    cv2.circle(frame, (rim_center_x, rim_center_y), radius, (255, 0, 255), 1)

def draw_trajectory(frame, frame_width, frame_height):
    """绘制球的轨迹（已禁用）"""
    # 轨迹绘制已禁用，减少视觉干扰
    pass

def draw_info_panel(frame, timestamp, status_msg, rim_info=None, ball_info=None):
    """绘制信息面板（放大2倍，字体更大）"""
    h, w = frame.shape[:2]
    
    # 更大的半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (1000, 600), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    y_offset = 50
    line_height = 45
    
    # 时间信息（更大字体）
    cv2.putText(frame, f"Time: {timestamp/60:.2f} min ({timestamp:.1f}s)", 
                (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
    y_offset += line_height
    
    # 校准状态
    calib_status = "Calibrated" if state.is_calibrated else f"Calibrating {len(state.calibration_buffer)}/{config.CALIBRATION_FRAMES}"
    color = (0, 255, 0) if state.is_calibrated else (0, 165, 255)
    cv2.putText(frame, f"Status: {calib_status}", 
                (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
    y_offset += line_height
    
    # 统计信息
    cv2.putText(frame, f"Frame: {state.frame_count}", 
                (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    y_offset += line_height
    
    cv2.putText(frame, f"Rim: {state.rim_detected_frames} | Ball: {state.ball_detected_frames} | Goals: {len(state.goal_events)}", 
                (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    y_offset += line_height + 10
    
    # 篮筐信息
    if rim_info:
        cv2.putText(frame, f"RIM: conf={rim_info['conf']:.3f} pos=({rim_info['x']:.3f},{rim_info['y']:.3f})", 
                    (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "RIM: NOT DETECTED", 
                    (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    y_offset += line_height
    
    # 篮球信息
    if ball_info:
        cv2.putText(frame, f"BALL: conf={ball_info['conf']:.3f} pos=({ball_info['x']:.3f},{ball_info['y']:.3f})", 
                    (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
    else:
        cv2.putText(frame, "BALL: NOT DETECTED", 
                    (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (128, 128, 128), 2)
    y_offset += line_height + 20
    
    # 检测状态
    cv2.putText(frame, f"Status: {status_msg}", 
                (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
    y_offset += line_height + 20
    
    # 配置参数
    cv2.putText(frame, "Config:", 
                (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
    y_offset += 35
    
    cv2.putText(frame, f"  Upper Zone: {config.TARGET_ZONE_HEIGHT}", 
                (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    y_offset += 35
    
    cv2.putText(frame, f"  Below Zone: {config.TARGET_ZONE_BELOW_HEIGHT}", 
                (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    y_offset += 35
    
    cv2.putText(frame, f"  Interaction: {config.INTERACTION_DISTANCE}", 
                (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    y_offset += 35
    
    cv2.putText(frame, f"  Event Window: {config.EVENT_WINDOW}s", 
                (30, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

# ==================== 🚀 主程序 ====================

def run_debug():
    print(f"📦 加载模型: {MODEL_PATH}")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 跳转到指定时间
    start_frame = int(START_MIN * 60 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print(f"\n🚀 开始播放: {START_MIN}分 ({start_frame}/{total_frames}帧)")
    print(f"📐 分辨率: {frame_width}×{frame_height}")
    print("=" * 60)
    print("⌨️  快捷键:")
    print("   [空格]  暂停/继续")
    print("   [F]     下一帧")
    print("   [D]     快进 5秒")
    print("   [A]     快退 5秒")
    print("   [Q]     退出")
    print("=" * 60)
    
    paused = False
    
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                print("\n✅ 视频播放结束")
                print(f"📊 最终统计:")
                print(f"   总帧数: {state.frame_count}")
                print(f"   篮筐检测帧: {state.rim_detected_frames}")
                print(f"   篮球检测帧: {state.ball_detected_frames}")
                print(f"   进球事件: {len(state.goal_events)}")
                if state.goal_events:
                    print("\n🎯 进球列表:")
                    for i, goal in enumerate(state.goal_events, 1):
                        print(f"   {i}. 时间={goal['timestamp']:.2f}s, 帧={goal['frame']}, 类型={goal['zone_type']}")
                break
            
            state.frame_count += 1
        
        debug_frame = frame.copy()
        timestamp = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
        
        # YOLO 推理
        results = model.predict(debug_frame, conf=0.01, device=device, verbose=False, imgsz=1024)
        
        rim_detected = False
        ball_detected = False
        status_msg = "等待检测..."
        goal_detected = False
        rim_info = None
        ball_info = None
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            coords = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            
            for i, conf in enumerate(confs):
                cls_id = int(clss[i])
                x1, y1, x2, y2 = map(int, coords[i])
                
                # 篮筐检测
                if cls_id == 1 and conf > config.CONF_THRESHOLD_RIM:
                    rim_detected = True
                    rim_box_norm = normalize_box([x1, y1, x2, y2], frame_width, frame_height)
                    rim_info = {
                        'conf': conf,
                        'x': rim_box_norm['centerX'],
                        'y': rim_box_norm['centerY']
                    }
                    
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    cv2.putText(debug_frame, f"Rim {conf:.2f}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # 打印篮筐信息
                    print(f"[Frame {state.frame_count:05d}] RIM: conf={conf:.3f}, pos=({rim_box_norm['centerX']:.3f}, {rim_box_norm['centerY']:.3f})")
                    
                    # 校准逻辑
                    if not state.is_calibrated:
                        status_msg = process_calibration(rim_box_norm)
                
                # 篮球检测
                elif cls_id == 0 and conf > config.CONF_THRESHOLD_BALL:
                    ball_detected = True
                    ball_box_norm = normalize_box([x1, y1, x2, y2], frame_width, frame_height)
                    ball_info = {
                        'conf': conf,
                        'x': ball_box_norm['centerX'],
                        'y': ball_box_norm['centerY']
                    }
                    
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 140, 255), 3)
                    cv2.putText(debug_frame, f"Ball {conf:.2f}", (x1, y1-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 140, 255), 2)
                    
                    # 打印篮球信息
                    print(f"[Frame {state.frame_count:05d}] BALL: conf={conf:.3f}, pos=({ball_box_norm['centerX']:.3f}, {ball_box_norm['centerY']:.3f})")
                    
                    # 事件检测
                    if state.is_calibrated:
                        state.trajectory.append((ball_box_norm['centerX'], ball_box_norm['centerY']))
                        
                        goal, msg = process_event_detection(ball_box_norm, timestamp)
                        status_msg = msg
                        if goal:
                            goal_detected = True
                            print(f"🎉 [Frame {state.frame_count:05d}] GOAL DETECTED! Time={timestamp:.2f}s")
        
        # 更新统计
        if rim_detected:
            state.rim_detected_frames += 1
        if ball_detected:
            state.ball_detected_frames += 1
        
        # 绘制检测区域和信息
        draw_detection_zones(debug_frame, frame_width, frame_height)
        # draw_trajectory(debug_frame, frame_width, frame_height)  # 已禁用
        draw_info_panel(debug_frame, timestamp, status_msg, rim_info, ball_info)
        
        # 进球提示
        if goal_detected:
            cv2.putText(debug_frame, "GOAL!!!", (frame_width//2 - 100, frame_height//2), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 5)
        
        # 缩放显示（不缩放，保持原始大小以便看清细节）
        small_frame = debug_frame
        
        cv2.imshow('Goal Detection Debug', small_frame)
        
        # 键盘控制
        key = cv2.waitKey(1 if not paused else 0) & 0xFF
        if key == ord('q'):
            break
        elif key == 32:  # 空格
            paused = not paused
        elif key == ord('f'):
            paused = True
            ret, frame = cap.read()
        elif key == ord('d'):
            curr_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_pos + (5 * fps))
            print("⏩ 快进 5秒")
        elif key == ord('a'):
            curr_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, curr_pos - (5 * fps)))
            print("⏪ 快退 5秒")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_debug()
