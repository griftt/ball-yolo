# -*- coding: utf-8 -*-
import os
import cv2
import time
import logging
import threading
import subprocess
import queue
import numpy as np
import torch
from ultralytics import YOLO
from tqdm import tqdm
from collections import deque

# ==================== ⚙️ 核心配置区域 ====================

# 1. ⏱️ 运行时间控制
START_FROM_MINUTES = 1.1     
MAX_PROCESS_MINUTES = 1.0     # None = 跑完为止

# 2. 🎯 自动校准参数
CONF_THRES_RIM_INIT = 0.40   
CALIBRATION_SAMPLES = 30     

# 3. ⚡️ 进球逻辑 (Zone-Based)
# [x1, y1, x2, y2]
HIGH_ZONE_OFFSET = 150       
GOAL_ZONE_OFFSET = 150       
SHOT_WINDOW = 3.0            

# 4. 🎬 剪辑参数
CLIP_PRE_TIME = 4.0          
CLIP_POST_TIME = 2.0         
SHOT_COOLDOWN = 2.0          

# 5. 🎨 轨迹特效配置 (核心优化)
DRAW_TRAJECTORY = True       
TRAJECTORY_COLOR = (0, 140, 255) # 纯正橙色 (BGR)
TRAJECTORY_ALPHA = 0.7         # 透明度 (越高越不透明)
TRAJECTORY_THICKNESS_SCALE = 0.8 # 线条粗细倍率

# 6. 🤖 模型与路径
MODEL_PATH = "runs/train/yolo11mbest/best.mlpackage"
VIDEO_PATH = "/Users/grifftwu/Desktop/历史篮球/1122/check.mp4"
OUTPUT_DIR = "./outputs/auto_mps_clips_1112_perfect_arc"

# 7. 推理配置
CONF_THRES_BALL = 0.15       
INFERENCE_SIZE = 1024        

CLS_BALL = 0
CLS_RIM = 1

# ==================== 系统初始化 ====================
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

class TrajectoryProcessor:
    """专门负责优化轨迹的数学类"""
    @staticmethod
    def smooth_path(points, window_size=5):
        """使用卷积平滑坐标点，生成完美弧线"""
        if len(points) < window_size: return points
        
        # 分离 X, Y, R
        xs = np.array([p[0] for p in points])
        ys = np.array([p[1] for p in points])
        rs = np.array([p[2] for p in points])
        
        # 简单移动平均 (Moving Average)
        box = np.ones(window_size) / window_size
        xs_smooth = np.convolve(xs, box, mode='valid')
        ys_smooth = np.convolve(ys, box, mode='valid')
        rs_smooth = np.convolve(rs, box, mode='valid') # 半径也平滑一下
        
        # 重新组合
        smoothed_points = []
        offset = window_size // 2
        
        # 补齐开头和结尾被切掉的点(用原始数据)
        for i in range(offset):
            smoothed_points.append(points[i])
            
        for i in range(len(xs_smooth)):
            smoothed_points.append((int(xs_smooth[i]), int(ys_smooth[i]), int(rs_smooth[i])))
            
        # 补齐结尾
        for i in range(len(points) - offset, len(points)):
            smoothed_points.append(points[i])
            
        return smoothed_points

    @staticmethod
    def clean_shot_trajectory(raw_history, start_frame, end_frame, rim_y, fps):
        """
        核心算法：截取从'起手(Dip)'到'入网(Splash)'的完美片段
        """
        # 1. 提取时间窗口内的点
        candidates = [p for p in raw_history if start_frame <= p[0] <= end_frame]
        if not candidates: return {}
        
        # candidates = [(frame, x, y, r), ...]
        
        # 2. 找到最高点 (Apex) 的索引
        # Y轴向下为正，所以 min(y) 是最高点
        min_y = float('inf')
        apex_index = 0
        for i, p in enumerate(candidates):
            if p[2] < min_y:
                min_y = p[2]
                apex_index = i
                
        # 3. 【智能起点】从最高点往回找 (寻找起手点)
        # 规则：球应该是一直在上升的 (Y在变小)。
        # 如果发现Y变大(球在下降)，说明这是上一次运球的反弹，立即停止。
        start_index = 0
        if apex_index > 0:
            for i in range(apex_index - 1, -1, -1):
                current_y = candidates[i][2]
                next_y = candidates[i+1][2] # 时间上靠后的点
                
                # 容差：允许微小的抖动 (5px)
                if current_y < next_y - 10: 
                    # 当前点比后面的点高很多 -> 说明球在下降 -> 截断！
                    start_index = i + 1
                    break
                start_index = i # 继续回溯

        # 4. 【智能终点】从最高点往后找 (寻找入网点)
        # 规则：当球落到篮筐高度(rim_y)以下一定程度时，停止
        stop_index = len(candidates) - 1
        for i in range(apex_index, len(candidates)):
            current_y = candidates[i][2]
            # 如果球掉到了篮筐下沿 50px 的位置，认为已经进网了，切断
            if current_y > rim_y + 50:
                stop_index = i
                break
        
        # 截取有效片段
        valid_segment = candidates[start_index : stop_index + 1]
        
        if len(valid_segment) < 3: return {}

        # 5. 平滑处理
        # 提取坐标进行平滑: (x, y, r)
        points_only = [(p[1], p[2], p[3]) for p in valid_segment]
        smoothed = TrajectoryProcessor.smooth_path(points_only)
        
        # 重组回字典: {frame: (x, y, r)}
        cleaned_trace = {}
        for i, original_p in enumerate(valid_segment):
            if i < len(smoothed):
                cleaned_trace[original_p[0]] = smoothed[i]
        
        return cleaned_trace

class ClipWorker(threading.Thread):
    def __init__(self, task_queue, fps, width, height):
        super().__init__()
        self.task_queue = task_queue
        self.fps = fps
        self.frame_size = (int(width), int(height))
        self.daemon = True 
        self.running = True

    def run(self):
        while self.running:
            try:
                task = self.task_queue.get(timeout=1) 
            except queue.Empty:
                continue
            if task is None: break
            
            source, start_sec, duration_sec, out_path, trace_data = task
            
            if DRAW_TRAJECTORY and trace_data:
                self.process_with_trajectory(source, start_sec, duration_sec, out_path, trace_data)
            else:
                self.process_fast_ffmpeg(source, start_sec, duration_sec, out_path)
            
            self.task_queue.task_done()

    def process_fast_ffmpeg(self, source, start, duration, out_path):
        try:
            cmd = [
                "ffmpeg", "-nostdin", "-y",
                "-ss", f"{start:.3f}",
                "-i", source,
                "-t", f"{duration:.3f}",
                "-c", "copy",
                "-avoid_negative_ts", "1",
                "-loglevel", "error",
                out_path
            ]
            subprocess.run(cmd, check=True)
            tqdm.write(f"✅ [纯剪辑] {os.path.basename(out_path)}")
        except Exception as e:
            logger.error(f"❌ FFmpeg: {e}")

    def process_with_trajectory(self, source, start_sec, duration_sec, final_out_path, trace_data):
        temp_video = final_out_path.replace(".mp4", "_temp.mp4")
        
        try:
            cap = cv2.VideoCapture(source)
            start_frame = int(start_sec * self.fps)
            end_frame = start_frame + int(duration_sec * self.fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(temp_video, fourcc, self.fps, self.frame_size)
            
            points_drawn = []
            
            current_f = start_frame
            while current_f < end_frame:
                ret, frame = cap.read()
                if not ret: break
                
                if current_f in trace_data:
                    points_drawn.append(trace_data[current_f])
                
                # 🟢 绘制平滑曲线
                if len(points_drawn) > 1:
                    overlay = np.zeros_like(frame)
                    
                    for i in range(1, len(points_drawn)):
                        pt1 = points_drawn[i-1]
                        pt2 = points_drawn[i]
                        
                        # 动态线宽
                        thickness = int((pt1[2] + pt2[2]) / 2 * TRAJECTORY_THICKNESS_SCALE)
                        thickness = max(2, thickness)
                        
                        cv2.line(overlay, (pt1[0], pt1[1]), (pt2[0], pt2[1]), TRAJECTORY_COLOR, thickness, cv2.LINE_AA)
                        cv2.circle(overlay, (pt2[0], pt2[1]), int(thickness/2), TRAJECTORY_COLOR, -1, cv2.LINE_AA)
                    
                    # Alpha 混合
                    mask = np.any(overlay != [0, 0, 0], axis=-1)
                    frame[mask] = cv2.addWeighted(frame, 1.0, overlay, TRAJECTORY_ALPHA, 0)[mask]
                    
                    # 亮白色球心 (稍微小一点)
                    last = points_drawn[-1]
                    cv2.circle(frame, (last[0], last[1]), 3, (255, 255, 255), -1, cv2.LINE_AA)

                out.write(frame)
                current_f += 1
                
            cap.release()
            out.release()
            
            # 2. 合并音频
            cmd = [
                "ffmpeg", "-y", "-nostdin",
                "-i", temp_video,
                "-ss", f"{start_sec:.3f}",
                "-t", f"{duration_sec:.3f}",
                "-i", source,
                "-map", "0:v", "-map", "1:a",
                "-c:v", "libx264", "-preset", "ultrafast",
                "-c:a", "aac",
                "-shortest", "-loglevel", "error",
                final_out_path
            ]
            subprocess.run(cmd, check=True)
            if os.path.exists(temp_video): os.remove(temp_video)
            tqdm.write(f"🎨 [完美轨迹] {os.path.basename(final_out_path)}")
            
        except Exception as e:
            logger.error(f"❌ 渲染: {e}")
            if os.path.exists(temp_video): os.remove(temp_video)

class AutoMPSDetector:
    def __init__(self, model_path, video_path, start_min, duration_min):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.clip_queue = queue.Queue()
        self.worker = ClipWorker(self.clip_queue, self.fps, width, height)
        self.worker.start()
        
        # 记录 8秒 数据
        self.ball_history = deque(maxlen=int(8 * self.fps))
        
        if MODEL_PATH.endswith(".mlpackage"):
            self.device = 'cpu'
            print(f"⚠️ CoreML (CPU)")
        else:
            self.device = 'mps'
            print(f"⚡️ MPS 加速")

        print(f"📦 模型: {model_path}")
        self.model = YOLO(model_path)
        # 预热
        self.model.predict(np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8), device=self.device, verbose=False, imgsz=INFERENCE_SIZE)
        
        self.start_frame = int(start_min * 60 * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        
        if duration_min is None:
            self.end_frame = self.total_frames
        else:
            self.end_frame = min(self.total_frames, self.start_frame + int(duration_min * 60 * self.fps))

        # 状态变量
        self.is_calibrated = False
        self.calibration_buffer = [] 
        
        self.last_interaction_ts = -10.0
        self.last_shot_ts = -10.0       
        self.shot_count = 0
        
        self.rim_box = []
        self.high_line = 0
        self.goal_zone = []

    def run(self):
        print(f"🚀 开始: {START_FROM_MINUTES}分 | 特效: {'开' if DRAW_TRAJECTORY else '关'}")
        pbar = tqdm(total=self.end_frame - self.start_frame, unit="frame", ncols=100)
        current_frame_idx = self.start_frame
        
        try:
            while True:
                if current_frame_idx >= self.end_frame: break
                ret, frame = self.cap.read()
                if not ret: break
                
                curr_time = current_frame_idx / self.fps
                
                if not self.is_calibrated:
                    self._run_calibration(frame)
                else:
                    self._run_inference(frame, curr_time, current_frame_idx)

                pbar.update(1)
                current_frame_idx += 1
                
        except KeyboardInterrupt:
            print("\n用户中断")
        finally:
            pbar.close()
            self.cap.release()
            self.shutdown()

    def _run_calibration(self, frame):
        results = self.model.predict(frame, verbose=False, conf=0.1, imgsz=INFERENCE_SIZE, classes=[CLS_RIM], device=self.device)
        if results[0].boxes:
            box = results[0].boxes[0]
            if float(box.conf) > CONF_THRES_RIM_INIT:
                self.calibration_buffer.append(box.xyxy[0].cpu().numpy())
                
        if len(self.calibration_buffer) >= CALIBRATION_SAMPLES:
            box = np.median(self.calibration_buffer, axis=0)
            x1, y1, x2, y2 = map(int, box)
            
            # 定义区域
            self.rim_box = [x1 - 10, y1 - 10, x2 + 10, y2 + 10]
            self.high_line = y1 
            self.goal_zone = [x1 - 30, y1 + 10, x2 + 30, y2 + GOAL_ZONE_OFFSET]
            
            self.is_calibrated = True
            tqdm.write(f"✅ 篮筐锁定")

    def _run_inference(self, frame, current_time, frame_idx):
        results = self.model.predict(frame, verbose=False, conf=0.01, imgsz=INFERENCE_SIZE, classes=[CLS_BALL], device=self.device)
        
        if results[0].boxes:
            best_conf = 0
            best_box = None
            confs = results[0].boxes.conf.cpu().numpy()
            coords = results[0].boxes.xyxy.cpu().numpy()
            
            for i, conf in enumerate(confs):
                if conf > CONF_THRES_BALL and conf > best_conf:
                    best_conf = conf
                    best_box = coords[i]
            
            if best_box is not None:
                x1, y1, x2, y2 = best_box
                cx, cy = int((x1+x2)/2), int((y1+y2)/2)
                radius = int(max(x2-x1, y2-y1) / 2)
                
                # 存入原始数据: (帧号, x, y, 半径)
                self.ball_history.append((frame_idx, cx, cy, radius))
                
                self._check_logic(results, current_time, frame_idx)

    def _check_logic(self, results, current_time, frame_idx):
        if current_time - self.last_shot_ts < SHOT_COOLDOWN: return

        if results[0].boxes is not None:
            coords = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            hit = False
            for i, conf in enumerate(confs):
                if conf > CONF_THRES_BALL:
                    bx1, by1, bx2, by2 = coords[i]
                    cx, cy = (bx1+bx2)/2, (by1+by2)/2
                    
                    is_high = cy < self.high_line
                    rx1, ry1, rx2, ry2 = self.rim_box
                    is_touch = (rx1 < cx < rx2) and (ry1 < cy < ry2)
                    
                    if is_high or is_touch:
                        self.last_interaction_ts = current_time
                        
                    gx1, gy1, gx2, gy2 = self.goal_zone
                    if (gx1 < cx < gx2) and (gy1 < cy < gy2):
                        hit = True
            
            if hit and 0.05 < (current_time - self.last_interaction_ts) < SHOT_WINDOW:
                self.trigger_goal(current_time, frame_idx)

    def trigger_goal(self, current_time, current_frame_idx):
        self.shot_count += 1
        self.last_shot_ts = current_time
        
        tqdm.write(f"🏀 进球! No.{self.shot_count}")
        
        filename = f"goal_{self.shot_count:03d}_{int(current_time)}s.mp4"
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        start_cut = max(0, current_time - CLIP_PRE_TIME)
        duration = CLIP_PRE_TIME + CLIP_POST_TIME
        start_frame = int(start_cut * self.fps)
        end_frame = int((start_cut + duration) * self.fps)
        
        # 🟢 调用清洗算法，获取完美轨迹
        rim_y = self.high_line
        cleaned_trace = TrajectoryProcessor.clean_shot_trajectory(
            list(self.ball_history), start_frame, end_frame, rim_y, self.fps
        )
        
        self.clip_queue.put((self.video_path, start_cut, duration, save_path, cleaned_trace))

    def shutdown(self):
        print(f"\n🏁 结束! 共 {self.shot_count} 球")
        self.clip_queue.join()
        self.worker.running = False
        subprocess.run(["open", OUTPUT_DIR])

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 缺模型: {MODEL_PATH}")
    else:
        detector = AutoMPSDetector(MODEL_PATH, VIDEO_PATH, START_FROM_MINUTES, MAX_PROCESS_MINUTES)
        detector.run()