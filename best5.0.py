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

# 1. 📂 视频任务列表 (支持多个文件)
# 格式: {"path": "视频路径", "start": 起始分钟数}
VIDEO_TASKS = [
    {
        "path": "/Users/grifftwu/Desktop/历史篮球/1126/111.mp4", 
        "start": 25.25  # 例如上次检测到了18分 44秒
    },
    {
        "path": "/Users/grifftwu/Desktop/历史篮球/1126/222.mp4", 
        "start": 0.5   # 新视频从 0.5 分钟开始
    },
    {
        "path": "/Users/grifftwu/Desktop/历史篮球/1126/333.mp4", 
        "start": 0.5  # 例如上次检测到了13分50秒，这里填13.8继续
    },
    {
        "path": "/Users/grifftwu/Desktop/历史篮球/1126/444.mp4", 
        "start": 0.5   # 新视频从 0.5 分钟开始
    },
    {
        "path": "/Users/grifftwu/Desktop/历史篮球/1126/555.mp4", 
        "start": 0.5  # 例如上次检测到了13分50秒，这里填13.8继续
    },
    
     

    # 你可以继续添加更多...
]



# 2. ⏱️ 全局配置
MAX_PROCESS_MINUTES = 30     # 每个视频最多检测多少分钟 (设为 None 则检测到结尾)
OUTPUT_DIR = "./outputs/auto_mps_clips_batch_01"

# 3. ⚙️ 性能优化配置 (针对你的 M3 Pro + 1024模型)
INFERENCE_SIZE = 1024        # ⚠️ 保持和你训练时的尺寸一致，不要改
FRAME_SKIP = 3               # ⚡️ 跳帧优化: 每 3 帧检测一次 (大幅提速)
ROTATE_VIDEO_180 = False     # 是否旋转画面

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
SHOT_COOLDOWN = 2.0          

# 6. 🤖 模型路径
MODEL_PATH = "./runs/train/yolo11_finetune_new_court/weights/best.mlpackage"



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
                    "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "copy"
                ])
            else:
                cmd.extend(["-c", "copy", "-avoid_negative_ts", "1"])

            cmd.append(out_path)
            subprocess.run(cmd, check=True)
            tqdm.write(f"✅ [已保存] {os.path.basename(out_path)}")
        except Exception as e:
            logger.error(f"❌ 剪辑出错: {e}")

class AutoMPSDetector:
    def __init__(self, loaded_model, device, video_path, start_min, duration_min):
        """
        初始化变动: 现在接收已经加载好的 model 对象，而不是路径
        """
        self.model = loaded_model
        self.device = device
        self.video_path = video_path
        
        self.clip_queue = queue.Queue()
        self.worker = ClipWorker(self.clip_queue, ROTATE_VIDEO_180)
        self.worker.start()
        
        # 预热 (使用较小的尺寸预热，仅为了唤醒管道)
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

        # 逻辑变量
        self.is_calibrated = False
        self.calibration_buffer = [] 
        self.locked_hoop_box = None  
        self.last_interaction_ts = -10.0
        self.last_shot_ts = -10.0       
        self.shot_count = 0
        self.rim_box = []
        self.high_line = 0
        self.goal_zone = []

        # Resize 优化
        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # 直接缩放到模型需要的 1024 宽，减少模型内部处理压力
        self.resize_scale = INFERENCE_SIZE / self.original_width
        self.detect_w = INFERENCE_SIZE
        self.detect_h = int(self.original_height * self.resize_scale)

   



    def run(self):
        video_name = os.path.basename(self.video_path)
        print(f"\n🎬 正在处理: {video_name} | 跳帧: {FRAME_SKIP}")
        
        process_len = self.end_frame - self.start_frame
        # 宽一点的进度条，以便显示更多信息
        pbar = tqdm(total=process_len, unit="frame", ncols=120) 
        current_frame_idx = self.start_frame
        
        # 散热 & 监控 计时器
        job_start_time = time.time()
        last_rest_time = time.time()
        
        #以此进程对象监控自身内存
        current_process = psutil.Process(os.getpid())
        
        try:
            while True:
                if current_frame_idx >= self.end_frame: break

               
            
                # =======================================================

                ret, frame = self.cap.read()
                if not ret: break
                
                # ==================== 📊 实时状态显示 (时间 + 内存) ====================
                # 每 10 帧更新一次显示，避免刷新太快看不清
                if current_frame_idx % 10 == 0:
                    current_time = current_frame_idx / self.fps
                    mins = int(current_time // 60)
                    secs = int(current_time % 60)
                    
                    # 1. 获取本脚本占用的内存 (GB)
                    mem_info = current_process.memory_info()
                    script_mem_gb = mem_info.rss / (1024 ** 3) 
                    
                    # 2. 获取系统总内存使用率 (%)
                    sys_mem_percent = psutil.virtual_memory().percent
                    
                    # 3. 组合显示字符串
                    # 格式: 🔍 [14:20] | 🐏 1.2G/85%
                    # 解释: 视频进度 | 脚本占了1.2G内存 / 系统总内存用了85%
                    
                    # ⚠️ 内存预警颜色: 如果系统内存 > 90%，添加一个警告标记
                    warn_sign = "⚠️" if sys_mem_percent > 90 else "🐏"
                    
                    desc_str = f"🔍 [{mins:02d}:{secs:02d}] | {warn_sign} {script_mem_gb:.1f}GB / {sys_mem_percent}%"
                    pbar.set_description(desc_str)
                # =================================================================

                # 跳帧优化
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
                    current_time = current_frame_idx / self.fps # 确保传给推理的时间也是准确的
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
            self.rim_box = [x1 - 10, y1 - 10, x2 + 10, y2 + 10]
            self.high_line = y1  
            self.goal_zone = [x1 - 30, y1 + 10, x2 + 30, y2 + GOAL_ZONE_OFFSET]
            self.is_calibrated = True
            tqdm.write(f"✅ 篮筐锁定! 坐标: {self.locked_hoop_box.astype(int)}")

    def _run_inference(self, frame, current_time):
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

    # 1. 统一加载模型 (只加载一次，节省显存和时间)
    print("📦 正在加载 YOLO 模型...")
    
    # 硬件判断
    device = 'cpu'
    if MODEL_PATH.endswith(".mlpackage"):
        print(f"⚠️ 使用 CoreML 模型 (Neural Engine 加速)")
    elif torch.backends.mps.is_available():
        device = 'mps'
        print(f"⚡️ MPS 加速已开启")
    
    loaded_model = YOLO(MODEL_PATH)
    print("✅ 模型加载完毕，开始处理任务列表...")

    # 2. 遍历任务列表
    try:
        for i, task in enumerate(VIDEO_TASKS):
            path = task["path"]
            start_min = task.get("start", 0.0)
            
            if not os.path.exists(path):
                print(f"⚠️ 跳过无效路径: {path}")
                continue

            print(f"\n========================================")
            print(f"📂 任务 [{i+1}/{len(VIDEO_TASKS)}]: {os.path.basename(path)}")
            print(f"⏱️ 起始时间: {start_min} 分钟")
            print(f"========================================")

            detector = AutoMPSDetector(
                loaded_model, # 传入已加载的模型
                device,
                path, 
                start_min=start_min, 
                duration_min=MAX_PROCESS_MINUTES
            )
            detector.run()
            
        print("\n🎉🎉🎉 所有任务处理完成！")
        subprocess.run(["open", OUTPUT_DIR])

    except KeyboardInterrupt:
        print("\n\n⛔️ ----------------------------------------")
        print("⛔️ 用户全局中断，程序停止。")
        print("⛔️ ----------------------------------------")