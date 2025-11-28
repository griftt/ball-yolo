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

# ==================== ⚙️ 核心配置区域 ====================

# 1. ⏱️ 运行时间控制
START_FROM_MINUTES = 15
MAX_PROCESS_MINUTES = 30

# 🔄 [核心设置] 画面旋转开关
ROTATE_VIDEO_180 = False

# 2. 🎯 自动校准参数
CONF_THRES_RIM_INIT = 0.40
CALIBRATION_SAMPLES = 30

# 3. ⚡️ 进球逻辑参数 (优化版)
HIGH_ZONE_OFFSET = 150
GOAL_ZONE_OFFSET = 150
SHOT_WINDOW = 2.5

# 4. 🎬 剪辑参数
CLIP_PRE_TIME = 4.0
CLIP_POST_TIME = 2.0
SHOT_COOLDOWN = 2.0

# 5. 🤖 模型与路径
MODEL_PATH = "./runs/train/yolo11_finetune_new_court/weights/best.mlpackage"
VIDEO_PATH = "/Users/grifftwu/Desktop/历史篮球/1126/111.mp4"
OUTPUT_DIR = "./outputs/auto_mps_clips_1126_rotated1"

# 6. 推理配置
CONF_THRES_BALL = 0.15
INFERENCE_SIZE = 1024 

# ========== 🚀 新增优化配置 ==========
DETECT_WIDTH = 1280     # 将4K画面缩小到1280宽进行检测（极大提升速度，不影响剪辑画质）
FRAME_SKIP = 3          # 每隔2帧检测一次 (相当于15fps检测，完全足够捕捉进球)
# ===================================

CLS_BALL = 0
CLS_RIM = 1

# ==================== 系统初始化 ====================
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

class ClipWorker(threading.Thread):
    """
    剪辑工人：支持旋转导出
    """
    def __init__(self, task_queue, rotate_flag):
        super().__init__()
        self.task_queue = task_queue
        self.rotate_flag = rotate_flag
        self.daemon = True 
        self.running = True

    def run(self):
        while self.running:
            try:
                task = self.task_queue.get(timeout=1) 
            except queue.Empty:
                continue
            if task is None: break
            
            source, start, duration, out_path = task
            self.process_video(source, start, duration, out_path)
            self.task_queue.task_done()

    def process_video(self, source, start, duration, out_path):
        try:
            # 基础命令
            cmd = [
                "ffmpeg", "-nostdin", "-y",
                "-ss", f"{start:.3f}",
                "-i", source,
                "-t", f"{duration:.3f}",
                "-loglevel", "error"
            ]

            if self.rotate_flag:
                cmd.extend([
                    "-vf", "transpose=2,transpose=2", 
                    "-c:v", "libx264",        
                    "-preset", "ultrafast",   
                    "-c:a", "copy"            
                ])
            else:
                cmd.extend([
                    "-c", "copy",
                    "-avoid_negative_ts", "1"
                ])

            cmd.append(out_path)
            subprocess.run(cmd, check=True)
            tqdm.write(f"✅ [已保存] {os.path.basename(out_path)}")
        
        except Exception as e:
            logger.error(f"❌ 剪辑出错: {e}")

class AutoMPSDetector:
    def __init__(self, model_path, video_path, start_min, duration_min):
        self.video_path = video_path
        
        self.clip_queue = queue.Queue()
        self.worker = ClipWorker(self.clip_queue, ROTATE_VIDEO_180)
        self.worker.start()
        
        # 硬件检查
        if MODEL_PATH.endswith(".mlpackage"):
            self.device = 'cpu' # CoreML 自动接管
            print(f"⚠️ 使用 CoreML 模型 (Neural Engine 加速)")
        elif torch.backends.mps.is_available():
            self.device = 'mps'
            print(f"⚡️ MPS 加速已开启")
        else:
            self.device = 'cpu'

        print(f"📦 加载模型: {model_path}")
        self.model = YOLO(model_path)
        
        # 预热 (使用较小的尺寸预热)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, device=self.device, verbose=False, imgsz=INFERENCE_SIZE)
        
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        self.start_frame = int(start_min * 60 * self.fps)
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        
        if duration_min is None:
            self.end_frame = self.total_frames
        else:
            self.end_frame = min(self.total_frames, self.start_frame + int(duration_min * 60 * self.fps))

        self.is_calibrated = False
        self.calibration_buffer = [] 
        self.locked_hoop_box = None  
        
        self.last_interaction_ts = -10.0
        self.last_shot_ts = -10.0       
        self.shot_count = 0
        
        self.rim_box = []
        self.high_line = 0
        self.goal_zone = []

        # 计算 Resize 比例
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resize_scale = DETECT_WIDTH / self.original_width
        self.detect_h = int(self.original_height * self.resize_scale)
        self.detect_w = int(DETECT_WIDTH)
        print(f"📉 检测分辨率优化: {self.original_width}x{self.original_height} -> {self.detect_w}x{self.detect_h}")

    def run(self):
        print(f"🚀 开始运行... | 跳帧设置: 每 {FRAME_SKIP} 帧检测一次")
        
        process_len = self.end_frame - self.start_frame
        # 1. 修改 tqdm 配置，增加动态描述区域
        pbar = tqdm(total=process_len, unit="frame", ncols=110)
        current_frame_idx = self.start_frame
        
        try:
            while True:
                if current_frame_idx >= self.end_frame: break

                ret, frame = self.cap.read()
                if not ret: break
                
                # 计算当前视频时间 (秒)
                current_time = current_frame_idx / self.fps
                
                # 2. 【核心修改】实时更新进度条前缀，显示 "视频时间 MM:SS"
                # 每 10 帧更新一次显示，避免刷新过于频繁闪烁
                if current_frame_idx % 10 == 0:
                    mins = int(current_time // 60)
                    secs = int(current_time % 60)
                    # 显示格式：🔍 [12:05] ...进度条...
                    pbar.set_description(f"🔍 [{mins:02d}:{secs:02d}]")

                # 跳帧逻辑
                if current_frame_idx % FRAME_SKIP != 0 and self.is_calibrated:
                    current_frame_idx += 1
                    pbar.update(1)
                    continue

                # 降采样
                detect_frame = cv2.resize(frame, (self.detect_w, self.detect_h), interpolation=cv2.INTER_LINEAR)

                if ROTATE_VIDEO_180:
                    detect_frame = cv2.rotate(detect_frame, cv2.ROTATE_180)
                
                if not self.is_calibrated:
                    self._run_calibration(detect_frame)
                else:
                    self._run_inference(detect_frame, current_time)

                pbar.update(1)
                current_frame_idx += 1
                
        except KeyboardInterrupt:
            # 3. 【核心修改】中断时计算下一次的启动时间
            stop_time = current_frame_idx / self.fps
            stop_min = stop_time / 60.0
            pbar.close() # 先关闭进度条，防止打印错位
            print(f"\n\n🛑 用户手动停止!")
            print(f"📌 当前视频进度: {int(stop_time//60)}分 {int(stop_time%60)}秒")
            # 建议稍微回退一点点(比如回退0.1分钟)，避免漏掉刚好在边缘的球
            suggested_start = max(0, stop_min - 0.1)
            print(f"👉 下次继续运行请设置: START_FROM_MINUTES = {suggested_start:.2f}\n")
            
        finally:
            if not pbar.disable: # 如果没关再关一次
                pbar.close()
            self.cap.release()
            self.shutdown()    
  
    def _run_calibration(self, frame):
        # 注意：这里的 frame 已经是缩小后的 detect_frame
        results = self.model.predict(
            frame, verbose=False, conf=0.1, iou=0.5, 
            imgsz=INFERENCE_SIZE, classes=[CLS_RIM], device=self.device
        )
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_rim = None
            max_conf = 0.0
            
            for box in boxes:
                conf = float(box.conf[0])
                if conf > max_conf:
                    max_conf = conf
                    best_rim = box.xyxy[0].cpu().numpy()
            
            if best_rim is not None and max_conf > CONF_THRES_RIM_INIT:
                self.calibration_buffer.append(best_rim)

        if len(self.calibration_buffer) >= CALIBRATION_SAMPLES:
            self.locked_hoop_box = np.median(self.calibration_buffer, axis=0)
            
            x1, y1, x2, y2 = map(int, self.locked_hoop_box)
            
            # 这里的坐标是基于 detect_frame (1280宽) 的，这完全没问题
            # 因为后面的推理也是基于 detect_frame 的，坐标系一致。
            self.rim_box = [x1 - 10, y1 - 10, x2 + 10, y2 + 10]
            self.high_line = y1  
            self.goal_zone = [x1 - 30, y1 + 10, x2 + 30, y2 + GOAL_ZONE_OFFSET]
            
            self.is_calibrated = True
            tqdm.write(f"✅ 篮筐锁定 (检测坐标系)! 坐标: {self.locked_hoop_box.astype(int)}")

    def _run_inference(self, frame, current_time):
        # 使用 persist=True 可以在视频流中稍微提升一点内部追踪效率，虽然这里没用 tracker
        results = self.model.predict(
            frame, verbose=False, conf=0.01, iou=0.5, 
            imgsz=INFERENCE_SIZE, classes=[CLS_BALL], device=self.device
        )
        self._check_zones_optimized(results, current_time)

    def _check_zones_optimized(self, results, current_time):
        if current_time - self.last_shot_ts < SHOT_COOLDOWN: return

        if results[0].boxes is not None:
            boxes = results[0].boxes
            coords = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            
            ball_in_goal = False
            
            for i, conf in enumerate(confs):
                if conf > CONF_THRES_BALL:
                    bx1, by1, bx2, by2 = coords[i]
                    cx = (bx1 + bx2) / 2
                    cy = (by1 + by2) / 2
                    
                    is_high = cy < self.high_line
                    
                    rx1, ry1, rx2, ry2 = self.rim_box
                    is_touching_rim = (rx1 < cx < rx2) and (ry1 < cy < ry2)
                    
                    if is_high or is_touching_rim:
                        self.last_interaction_ts = current_time
                        
                    gx1, gy1, gx2, gy2 = self.goal_zone
                    if (gx1 < cx < gx2) and (gy1 < cy < gy2):
                        ball_in_goal = True
            
            if ball_in_goal:
                time_diff = current_time - self.last_interaction_ts
                # SHOT_WINDOW 是秒数，不受帧率影响，所以跳帧也不影响这里的逻辑
                if 0.05 < time_diff < SHOT_WINDOW:
                    self.trigger_goal(current_time)

    def trigger_goal(self, current_time):
        self.shot_count += 1
        self.last_shot_ts = current_time
        
        tqdm.write(f"🏀 [进球!] 时间: {current_time:.2f}s | No.{self.shot_count}")
        
        filename = f"goal_{self.shot_count:03d}_{int(current_time)}s.mp4"
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        # 剪辑逻辑完全依赖时间戳，所以 Detect 缩放与否不影响剪辑出的视频画质
        start_cut = max(0, current_time - CLIP_PRE_TIME)
        duration = CLIP_PRE_TIME + CLIP_POST_TIME
        
        self.clip_queue.put((self.video_path, start_cut, duration, save_path))

    def shutdown(self):
        print(f"\n🏁 扫描结束！共发现: {self.shot_count} 个进球")
        if not self.clip_queue.empty():
            print(f"⏳ 处理剩余视频中...")
        self.clip_queue.join()
        self.worker.running = False
        subprocess.run(["open", OUTPUT_DIR])

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型 {MODEL_PATH}")
    else:
        detector = AutoMPSDetector(
            MODEL_PATH, 
            VIDEO_PATH, 
            start_min=START_FROM_MINUTES, 
            duration_min=MAX_PROCESS_MINUTES
        )
        detector.run()