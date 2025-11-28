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
import psutil 

# ==================== ⚙️ 核心配置区域 ====================

# 1. 📂 视频任务列表
VIDEO_TASKS = [
    # {"path": "/Users/grifftwu/Desktop/历史篮球/1126/111.mp4", "start": 25.25},
    # {"path": "/Users/grifftwu/Desktop/历史篮球/1126/222.mp4", "start": 27.97},
    # {"path": "/Users/grifftwu/Desktop/历史篮球/1126/333.mp4", "start": 9.51},
    # {"path": "/Users/grifftwu/Desktop/历史篮球/1126/444.mp4", "start": 0.5},
    {"path": "/Users/grifftwu/Desktop/历史篮球/1126/555.mp4", "start": 0.5},
]

# 2. ⏱️ 全局配置
MAX_PROCESS_MINUTES = 30     
OUTPUT_DIR = "./outputs/auto_mps_clips_batch_final"

# 3. ⚙️ 性能优化配置
INFERENCE_SIZE = 1024        
FRAME_SKIP = 3               # 每 3 帧检测一次
ROTATE_VIDEO_180 = False     

# 4. 🎯 自动校准与判定参数
CONF_THRES_RIM_INIT = 0.40   
CALIBRATION_SAMPLES = 30     
HIGH_ZONE_OFFSET = 150       
GOAL_ZONE_OFFSET = 150       
SHOT_WINDOW = 2.5            
CONF_THRES_BALL = 0.15       
CLS_BALL = 0
CLS_RIM = 1

# 5. 🎬 剪辑参数
CLIP_PRE_TIME = 4.0          
CLIP_POST_TIME = 2.0         
SHOT_COOLDOWN = 3.0          # ⚡️ 优化：改为3秒，避免庆祝动作导致重复剪辑

# 6. 🤖 模型路径
MODEL_PATH = "./runs/train/yolo11sbest/best.mlpackage"
# MODEL_PATH = "./runs/train/yolo11n_640_train/weights/best.pt"
# ==================== 🛡️ 散热保护配置 ====================
# 建议关闭 (False)，因为使用了 grab() 后负载会变低，且你之前 CPU 闲置率很高
ENABLE_HEAT_PROTECTION = False  
RUN_DURATION_SEC = 600         # 10分钟
REST_DURATION_SEC = 60         # 1分钟

# ==================== 系统初始化 ====================
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

class ClipWorker(threading.Thread):
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
            cmd = ["ffmpeg", "-nostdin", "-y", "-ss", f"{start:.3f}", "-i", source, "-t", f"{duration:.3f}", "-loglevel", "error"]
            if self.rotate_flag:
                cmd.extend(["-vf", "transpose=2,transpose=2", "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "copy"])
            else:
                cmd.extend(["-c", "copy", "-avoid_negative_ts", "1"])
            cmd.append(out_path)
            subprocess.run(cmd, check=True)
            tqdm.write(f"✅ [已保存] {os.path.basename(out_path)}")
        except Exception as e:
            logger.error(f"❌ 剪辑出错: {e}")

class AutoMPSDetector:
    # 🆕 接收全局散热管理器 heat_manager
    def __init__(self, loaded_model, device, video_path, start_min, duration_min, heat_manager):
        self.model = loaded_model
        self.device = device
        self.video_path = video_path
        self.heat_manager = heat_manager # 全局状态
        
        self.clip_queue = queue.Queue()
        self.worker = ClipWorker(self.clip_queue, ROTATE_VIDEO_180)
        self.worker.start()
        
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

        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resize_scale = INFERENCE_SIZE / self.original_width
        self.detect_w = INFERENCE_SIZE
        self.detect_h = int(self.original_height * self.resize_scale)

    def run(self):
        video_name = os.path.basename(self.video_path)
        print(f"\n🎬 正在处理: {video_name} | 跳帧: {FRAME_SKIP} | 🚀 极速模式: ON")
        
        process_len = self.end_frame - self.start_frame
        pbar = tqdm(total=process_len, unit="frame", ncols=120) 
        current_frame_idx = self.start_frame
        
        current_process = psutil.Process(os.getpid())
        
        try:
            while True:
                if current_frame_idx >= self.end_frame: break

                # ==================== 🛡️ 散热保护 (全局) ====================
                if ENABLE_HEAT_PROTECTION:
                    now = time.time()
                    elapsed = now - self.heat_manager['last_rest_time']
                    if elapsed > RUN_DURATION_SEC:
                        pbar.set_description("🧊 [散热降温中...]")
                        rest_pbar = tqdm(range(REST_DURATION_SEC), desc="❄️ 倒计时", leave=False, ncols=80)
                        for _ in rest_pbar: time.sleep(1)
                        
                        # 重置计时并刷新显示
                        self.heat_manager['last_rest_time'] = time.time()
                        pbar.set_description("⚡️ [恢复全速运行]")

                # ==================== 🚀 极速跳帧逻辑 (核心优化) ====================
                # 只有在已经校准、且当前是跳过帧时，才使用 grab()
                # grab() 不解码图像，速度极快
                if current_frame_idx % FRAME_SKIP != 0 and self.is_calibrated:
                    self.cap.grab()
                    current_frame_idx += 1
                    pbar.update(1)
                    continue

                # 需要检测的帧，才真正解码
                ret, frame = self.cap.read()
                if not ret: break
                # ================================================================
                
                # 状态显示
                if current_frame_idx % 10 == 0:
                    current_time = current_frame_idx / self.fps
                    mins = int(current_time // 60)
                    secs = int(current_time % 60)
                    
                    mem_info = current_process.memory_info()
                    script_mem_gb = mem_info.rss / (1024 ** 3) 
                    sys_mem_percent = psutil.virtual_memory().percent
                    warn_sign = "⚠️" if sys_mem_percent > 90 else "🐏"
                    
                    desc_str = f"🔍 [{mins:02d}:{secs:02d}] | {warn_sign} {script_mem_gb:.1f}G/{sys_mem_percent}%"
                    pbar.set_description(desc_str)

                # 降采样
                detect_frame = cv2.resize(frame, (self.detect_w, self.detect_h), interpolation=cv2.INTER_LINEAR)
                if ROTATE_VIDEO_180:
                    detect_frame = cv2.rotate(detect_frame, cv2.ROTATE_180)
                
                if not self.is_calibrated:
                    self._run_calibration(detect_frame)
                else:
                    current_time = current_frame_idx / self.fps
                    self._run_inference(detect_frame, current_time)

                pbar.update(1)
                current_frame_idx += 1
                
        except KeyboardInterrupt:
            pbar.close()
            stop_time = current_frame_idx / self.fps
            stop_min = stop_time / 60.0
            print(f"\n\n🛑 [当前文件中断] {video_name}")
            print(f"📌 中断时间点: {int(stop_time//60)}分 {int(stop_time%60)}秒")
            print(f"👉 该文件恢复参数: \"path\": \"{self.video_path}\", \"start\": {max(0, stop_min - 0.1):.2f}")
            self.shutdown()
            raise KeyboardInterrupt

        finally:
            if not pbar.disable: pbar.close()
            self.cap.release()
            self.shutdown()

    def _run_calibration(self, frame):
        results = self.model.predict(frame, verbose=False, conf=0.1, iou=0.5, imgsz=INFERENCE_SIZE, classes=[CLS_RIM], device=self.device)
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
            self.rim_box = [x1 - 10, y1 - 10, x2 + 10, y2 + 10]
            self.high_line = y1  
            self.goal_zone = [x1 - 30, y1 + 10, x2 + 30, y2 + GOAL_ZONE_OFFSET]
            self.is_calibrated = True
            tqdm.write(f"✅ 篮筐锁定! 坐标: {self.locked_hoop_box.astype(int)}")

    def _run_inference(self, frame, current_time):
        results = self.model.predict(frame, verbose=False, conf=0.01, iou=0.5, imgsz=INFERENCE_SIZE, classes=[CLS_BALL], device=self.device)
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
                if 0.05 < time_diff < SHOT_WINDOW:
                    self.trigger_goal(current_time)

    def trigger_goal(self, current_time):
        self.shot_count += 1
        self.last_shot_ts = current_time
        video_base = os.path.splitext(os.path.basename(self.video_path))[0]
        tqdm.write(f"🏀 [进球] {video_base} | 时间: {current_time:.2f}s")
        filename = f"{video_base}_goal_{self.shot_count:03d}_{int(current_time)}s.mp4"
        save_path = os.path.join(OUTPUT_DIR, filename)
        start_cut = max(0, current_time - CLIP_PRE_TIME)
        duration = CLIP_PRE_TIME + CLIP_POST_TIME
        self.clip_queue.put((self.video_path, start_cut, duration, save_path))

    def shutdown(self):
        if self.worker.running:
            if not self.clip_queue.empty():
                print(f"⏳ 正在完成剩余剪辑任务...")
            self.clip_queue.join()
            self.worker.running = False

# ==================== 主控制流程 ====================
if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型 {MODEL_PATH}")
        exit()

    print("📦 正在加载 YOLO 模型...")
    device = 'cpu'
    if MODEL_PATH.endswith(".mlpackage"):
        print(f"⚠️ 使用 CoreML 模型 (Neural Engine 加速)")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print(f"⚡️ MPS 加速已开启")
    
    loaded_model = YOLO(MODEL_PATH)
    print("✅ 模型加载完毕，开始处理任务列表...")

    # 🔥 全局散热管理器 (所有视频共享同一个计时器)
    GLOBAL_HEAT_MANAGER = {'last_rest_time': time.time()}

    try:
        for i, task in enumerate(VIDEO_TASKS):
            path = task["path"]
            start_min = task.get("start", 0.0)
            
            if not os.path.exists(path):
                print(f"⚠️ 跳过无效路径: {path}")
                continue

            print(f"\n========================================")
            print(f"📂 任务 [{i+1}/{len(VIDEO_TASKS)}]: {os.path.basename(path)}")
            print(f"========================================")

            detector = AutoMPSDetector(
                loaded_model, 
                device, 
                path, 
                start_min, 
                MAX_PROCESS_MINUTES,
                GLOBAL_HEAT_MANAGER # 传入全局管理器
            )
            detector.run()
            
        print("\n🎉🎉🎉 所有任务处理完成！")
        subprocess.run(["open", OUTPUT_DIR])

    except KeyboardInterrupt:
        print("\n\n⛔️ ----------------------------------------")
        print("⛔️ 用户全局中断，程序停止。")
        print("⛔️ ----------------------------------------")