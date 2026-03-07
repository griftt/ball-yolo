# -*- coding: utf-8 -*-
"""
进球检测诊断工具 + 射手追踪回溯系统 + 交互式 UI 控制
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque

# ==================== ⚙️ 配置 ====================
VIDEO_PATH = "outputs/CCC.mp4"
# VIDEO_PATH = "/Users/grifftwu/Desktop/历史篮球/20260304/CCC.MP4"
MODEL_PATH = "runs/detect/runs/train/yolo26s_640_train_hd_person/weights/best.pt"
START_MIN = 0

# 类别 ID 配置
CLASS_BALL = 0
CLASS_RIM = 1
CLASS_PERSON = 2

# ==================== 🎯 检测与射手参数 ====================
class DetectionConfig:
    CONF_THRESHOLD_BALL = 0.15
    CONF_THRESHOLD_RIM = 0.15
    CONF_THRESHOLD_PERSON = 0.30
    CALIBRATION_FRAMES = 30
    TARGET_ZONE_HEIGHT = 0.06
    TARGET_ZONE_BELOW_HEIGHT = 0.08
    TARGET_ZONE_H_EXPANSION = 0.01
    INTERACTION_DISTANCE = 0.20
    EXPANSION_FACTOR = 0.10
    EVENT_WINDOW = 2.5
    EVENT_COOLDOWN = 3.0
    REQUIRE_BELOW_ZONE = True

class ShooterConfig:
    HISTORY_SECONDS = 4.0
    SEARCH_WINDOW_START = 2.5
    SEARCH_WINDOW_END = 0.5
    POCKET_DISTANCE = 0.15
    IOU_THRESHOLD = 0.3

config = DetectionConfig()
shooter_cfg = ShooterConfig()

# ==================== 📊 状态机 ====================
class DiagnosticState:
    def __init__(self):
        self.is_calibrated = False
        self.calibration_buffer =[]
        self.rim_box = None
        self.target_zone = None
        self.below_target_zone = None
        
        self.last_interaction_time = -10.0
        self.last_event_time = -10.0
        
        self.frame_count = 0
        self.rim_detected_frames = 0
        self.ball_detected_frames = 0
        
        self.history_buffer = deque()

state = DiagnosticState()

# ==================== 🖱️ UI 控制与鼠标事件 ====================
class UIState:
    paused = False
    seek_offset = 0  # 需要跳转的帧数
    step_forward = False
    step_backward = False
    
    # 按钮区域定义 (x1, y1, x2, y2)
    buttons = {
        "prev_5s": (0, 0, 0, 0),
        "prev_frame": (0, 0, 0, 0),
        "play_pause": (0, 0, 0, 0),
        "next_frame": (0, 0, 0, 0),
        "next_5s": (0, 0, 0, 0)
    }

ui_state = UIState()

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        fps = param['fps']
        
        if in_box(x, y, ui_state.buttons["play_pause"]):
            ui_state.paused = not ui_state.paused
        elif in_box(x, y, ui_state.buttons["prev_frame"]):
            ui_state.paused = True
            ui_state.step_backward = True
        elif in_box(x, y, ui_state.buttons["next_frame"]):
            ui_state.paused = True
            ui_state.step_forward = True
        elif in_box(x, y, ui_state.buttons["prev_5s"]):
            ui_state.seek_offset = -int(5 * fps)
        elif in_box(x, y, ui_state.buttons["next_5s"]):
            ui_state.seek_offset = int(5 * fps)

def in_box(x, y, box):
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

def draw_ui_controls(frame, frame_width, frame_height):
    bar_h = 80
    y1 = frame_height - bar_h
    y2 = frame_height
    
    # 画底部半透明背景
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, y1), (frame_width, y2), (20, 20, 25), -1)
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
    
    # 按钮配置
    btn_w = 160
    btn_h = 50
    spacing = 20
    total_w = (btn_w * 5) + (spacing * 4)
    start_x = (frame_width - total_w) // 2
    btn_y = y1 + 15
    
    labels =[
        ("<< 5s", "prev_5s"),
        ("< Frame", "prev_frame"),
        ("PLAY" if ui_state.paused else "PAUSE", "play_pause"),
        ("Frame >", "next_frame"),
        ("5s >>", "next_5s")
    ]
    
    for i, (label, key) in enumerate(labels):
        bx1 = start_x + i * (btn_w + spacing)
        bx2 = bx1 + btn_w
        by1, by2 = btn_y, btn_y + btn_h
        
        ui_state.buttons[key] = (bx1, by1, bx2, by2)
        
        # 按钮背景色 (暂停按钮高亮)
        color = (0, 140, 255) if key == "play_pause" and not ui_state.paused else (70, 70, 70)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), color, -1)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (255, 255, 255), 2)
        
        # 文字居中
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        tx = bx1 + (btn_w - text_size[0]) // 2
        ty = by1 + (btn_h + text_size[1]) // 2
        cv2.putText(frame, label, (tx, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# ==================== 🔧 辅助函数 ====================

def normalize_box(box, frame_width, frame_height):
    x1, y1, x2, y2 = box
    return {
        'x': x1 / frame_width, 'y': y1 / frame_height,
        'width': (x2 - x1) / frame_width, 'height': (y2 - y1) / frame_height,
        'centerX': ((x1 + x2) / 2) / frame_width, 'centerY': ((y1 + y2) / 2) / frame_height,
        'minX': x1 / frame_width, 'minY': y1 / frame_height,
        'maxX': x2 / frame_width, 'maxY': y2 / frame_height,
        'midX': ((x1 + x2) / 2) / frame_width, 'midY': ((y1 + y2) / 2) / frame_height,
    }

def calculate_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

def calculate_iou(boxA, boxB):
    xA, yA = max(boxA['minX'], boxB['minX']), max(boxA['minY'], boxB['minY'])
    xB, yB = min(boxA['maxX'], boxB['maxX']), min(boxA['maxY'], boxB['maxY'])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    if interArea == 0: return 0
    boxAArea = boxA['width'] * boxA['height']
    boxBArea = boxB['width'] * boxB['height']
    return interArea / float(boxAArea + boxBArea - interArea)

# ==================== 🏀 核心逻辑与校准 ====================

def process_calibration(rim_box_norm):
    state.calibration_buffer.append(rim_box_norm)
    if len(state.calibration_buffer) < config.CALIBRATION_FRAMES: return
    
    avg_x = np.mean([b['x'] for b in state.calibration_buffer])
    avg_y = np.mean([b['y'] for b in state.calibration_buffer])
    avg_w = np.mean([b['width'] for b in state.calibration_buffer])
    avg_h = np.mean([b['height'] for b in state.calibration_buffer])
    
    state.rim_box = {'x': avg_x, 'y': avg_y, 'width': avg_w, 'height': avg_h,
                     'minX': avg_x, 'minY': avg_y, 'maxX': avg_x + avg_w, 'maxY': avg_y + avg_h,
                     'midX': avg_x + avg_w / 2, 'midY': avg_y + avg_h / 2}
    
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

def diagnose_detection(ball_box_norm, timestamp):
    if timestamp - state.last_event_time < config.EVENT_COOLDOWN: return None
    
    ball_center = (ball_box_norm['centerX'], ball_box_norm['centerY'])
    rim_center = (state.rim_box['midX'], state.rim_box['midY'])
    distance = calculate_distance(ball_center, rim_center)
    
    has_interaction = False
    if distance < config.INTERACTION_DISTANCE:
        state.last_interaction_time = timestamp
        has_interaction = True
    
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
    
    in_upper_zone = (state.target_zone['x'] <= ball_center[0] <= state.target_zone['x'] + state.target_zone['width'] and
                     state.target_zone['y'] <= ball_center[1] <= state.target_zone['y'] + state.target_zone['height'])
    
    in_below_zone = (state.below_target_zone['x'] <= ball_center[0] <= state.below_target_zone['x'] + state.below_target_zone['width'] and
                     state.below_target_zone['y'] <= ball_center[1] <= state.below_target_zone['y'] + state.below_target_zone['height'])
    
    in_valid_zone = in_below_zone if config.REQUIRE_BELOW_ZONE else (in_below_zone or in_upper_zone)
    
    if in_valid_zone and has_interaction:
        time_diff = abs(timestamp - state.last_interaction_time)
        if time_diff <= config.EVENT_WINDOW:
            state.last_event_time = timestamp
            return True
    return None

# ==================== 🕵️‍♂️ 射手回溯 VAR 系统 ====================

class TrackedPerson:
    def __init__(self, track_id, box_norm):
        self.track_id = track_id
        self.box_norm = box_norm
        self.score = 0
        self.color = (np.random.randint(50, 255), np.random.randint(50, 255), np.random.randint(50, 255))
        self.last_seen = 0

def denormalize_rect(rect, frame_width, frame_height):
    return (int(rect['x'] * frame_width), int(rect['y'] * frame_height),
            int((rect['x'] + rect['width']) * frame_width), int((rect['y'] + rect['height']) * frame_height))

def run_shooter_var_replay(goal_timestamp, frame_width, frame_height):
    print(f"\n📺 启动 VAR 射手判定回放 (Goal @ {goal_timestamp:.2f}s)...")
    search_start = goal_timestamp - shooter_cfg.SEARCH_WINDOW_START
    search_end = goal_timestamp - shooter_cfg.SEARCH_WINDOW_END
    replay_frames =[f for f in state.history_buffer if search_start <= f['timestamp'] <= search_end]
    
    if not replay_frames: return print("❌ 历史缓存不足，无法回溯")
        
    active_tracks =[]
    next_track_id = 1
    
    for data in replay_frames:
        timestamp, orig_frame = data['timestamp'], data['image'].copy()
        ball_box, person_boxes = data['ball'], data['persons']
        
        current_frame_tracks =[]
        for p_box in person_boxes:
            best_iou = 0
            best_track = None
            for track in active_tracks:
                iou = calculate_iou(track.box_norm, p_box)
                if iou > best_iou and iou > shooter_cfg.IOU_THRESHOLD:
                    best_iou, best_track = iou, track
            
            if best_track:
                best_track.box_norm = p_box
                best_track.last_seen = timestamp
                current_frame_tracks.append(best_track)
            else:
                new_track = TrackedPerson(next_track_id, p_box)
                new_track.last_seen = timestamp
                active_tracks.append(new_track)
                current_frame_tracks.append(new_track)
                next_track_id += 1
                
        if ball_box:
            ball_center = (ball_box['centerX'], ball_box['centerY'])
            b_px = denormalize_rect(ball_box, frame_width, frame_height)
            cv2.circle(orig_frame, (int((b_px[0]+b_px[2])/2), int((b_px[1]+b_px[3])/2)), 10, (0, 165, 255), -1)
            
            for track in current_frame_tracks:
                pocket_y = track.box_norm['minY'] + track.box_norm['height'] * 0.25
                dist = calculate_distance(ball_center, (track.box_norm['centerX'], pocket_y))
                if dist < shooter_cfg.POCKET_DISTANCE:
                    track.score += 1
                    p_px = denormalize_rect(track.box_norm, frame_width, frame_height)
                    cx, cy = int((p_px[0]+p_px[2])/2), int(p_px[1] + (p_px[3]-p_px[1])*0.25)
                    cv2.line(orig_frame, (cx, cy), (int((b_px[0]+b_px[2])/2), int((b_px[1]+b_px[3])/2)), (0, 255, 255), 2)
        
        cv2.putText(orig_frame, f"VAR REPLAY: T-{goal_timestamp - timestamp:.2f}s", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
        for track in current_frame_tracks:
            p_px = denormalize_rect(track.box_norm, frame_width, frame_height)
            cv2.rectangle(orig_frame, (p_px[0], p_px[1]), (p_px[2], p_px[3]), track.color, 2)
            cv2.putText(orig_frame, f"ID:{track.track_id} Score:{track.score}", (p_px[0], p_px[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255) if track.score > 0 else (255, 255, 255), 2)
        
        cv2.imshow('Goal Detection Diagnostic', orig_frame)
        cv2.waitKey(40)
        
    valid_tracks =[t for t in active_tracks if t.score > 3]
    if valid_tracks:
        winner = max(valid_tracks, key=lambda x: x.score)
        print(f"🎯 射手锁定! Track ID: {winner.track_id}, 最终得分: {winner.score}")
        final_frame = replay_frames[-1]['image'].copy()
        w_px = denormalize_rect(winner.box_norm, frame_width, frame_height)
        cv2.rectangle(final_frame, (w_px[0], w_px[1]), (w_px[2], w_px[3]), (0, 0, 255), 6)
        cv2.putText(final_frame, f"WINNER! SCORE: {winner.score}", (w_px[0]-20, w_px[1]-20), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
        cv2.imshow('Goal Detection Diagnostic', final_frame)
        cv2.waitKey(2000)

# ==================== 🚀 主流程 ====================

def run_diagnostic():
    print(f"📦 加载模型: {MODEL_PATH}")
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    max_history_frames = int(shooter_cfg.HISTORY_SECONDS * fps)
    
    cv2.namedWindow('Goal Detection Diagnostic')
    cv2.setMouseCallback('Goal Detection Diagnostic', mouse_callback, {'fps': fps})
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_MIN * 60 * fps))
    
    last_frame_img = None
    
    while True:
        # 处理 UI 控制导致的位置跳转
        if ui_state.seek_offset != 0:
            curr_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            new_pos = max(0, curr_pos + ui_state.seek_offset)
            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
            state.history_buffer.clear() # 清空缓存防错乱
            ui_state.seek_offset = 0
            ret, last_frame_img = cap.read()
            
        elif ui_state.step_backward:
            curr_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, curr_pos - 2))
            state.history_buffer.clear()
            ui_state.step_backward = False
            ret, last_frame_img = cap.read()
            
        elif ui_state.step_forward:
            ui_state.step_forward = False
            ret, last_frame_img = cap.read()
            
        elif not ui_state.paused:
            ret, last_frame_img = cap.read()
            if not ret: break
            state.frame_count += 1
        
        if last_frame_img is None:
            continue
            
        debug_frame = last_frame_img.copy()
        timestamp = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
        
        # 即使暂停，为了显示效果我们依然每帧推理一下当前的图片
        results = model.predict(debug_frame, conf=0.01, device=device, verbose=False, imgsz=1024)
        
        ball_box = None
        person_boxes =[]
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            coords = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            
            for i, conf in enumerate(confs):
                cls_id = int(clss[i])
                x1, y1, x2, y2 = map(int, coords[i])
                
                if cls_id == CLASS_RIM and conf > config.CONF_THRESHOLD_RIM:
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    if not state.is_calibrated and not ui_state.paused:
                        process_calibration(normalize_box([x1, y1, x2, y2], frame_width, frame_height))
                
                elif cls_id == CLASS_BALL and conf > config.CONF_THRESHOLD_BALL:
                    ball_box = normalize_box([x1, y1, x2, y2], frame_width, frame_height)
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (0, 140, 255), 3)
                
                elif cls_id == CLASS_PERSON and conf > config.CONF_THRESHOLD_PERSON:
                    person_box = normalize_box([x1, y1, x2, y2], frame_width, frame_height)
                    person_boxes.append(person_box)
                    cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
        
        if not ui_state.paused:
            state.history_buffer.append({
                'timestamp': timestamp, 'image': last_frame_img.copy(),
                'ball': ball_box, 'persons': person_boxes
            })
            if len(state.history_buffer) > max_history_frames:
                state.history_buffer.popleft()
        
        if ball_box and state.is_calibrated and not ui_state.paused:
            goal = diagnose_detection(ball_box, timestamp)
            if goal:
                print(f"\n🎉 [Frame] 进球! 时间={timestamp:.2f}s")
                ui_state.paused = True # 进球后自动暂停一下主画面
                run_shooter_var_replay(timestamp, frame_width, frame_height)
        
        cv2.putText(debug_frame, f"LIVE - Time: {timestamp:.2f}s", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        
        # 绘制 UI 按钮
        draw_ui_controls(debug_frame, frame_width, frame_height)
        
        cv2.imshow('Goal Detection Diagnostic', debug_frame)
        
        # 依然保留键盘快捷键作为备用
        key = cv2.waitKey(1 if not ui_state.paused else 30) & 0xFF
        if key == ord('q'): break
        elif key == 32: ui_state.paused = not ui_state.paused
        elif key == ord('f'): 
            ui_state.paused = True
            ui_state.step_forward = True
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_diagnostic()