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
    {"path": "/Users/grifftwu/Desktop/历史篮球/20260304/111.mp4", "start": 0},
]

# 2. ⏱️ 全局配置
MAX_PROCESS_MINUTES = 60     
OUTPUT_DIR = "./outputs/auto_clips_20260304"

# 3. ⚙️ 性能优化配置
INFERENCE_SIZE = 640        
FRAME_SKIP = 3               # 每 3 帧检测一次
ROTATE_VIDEO_180 = False     

# 4. 🎯 动态检测参数
CONF_THRES_RIM = 0.4        # 篮筐检测置信度
CONF_THRES_BALL = 0.4       # 篮球检测置信度
HIGH_ZONE_OFFSET = 150       # 高位区范围（篮筐上方）
GOAL_ZONE_OFFSET = 100       # 进球区范围（篮筐下方）
SHOT_WINDOW = 1.2          # 投篮时间窗口（秒）
CLS_BALL = 0
CLS_RIM = 1

# 5. 🎬 剪辑参数
CLIP_PRE_TIME = 4.0          
CLIP_POST_TIME = 2.0         
SHOT_COOLDOWN = 3.0          # 防止重复检测冷却时间

# 6. 🤖 模型路径
# MODEL_PATH = "runs/detect/runs/train/yolo26s_640_train_hd_person/weights/best.pt"
MODEL_PATH = "runs/yolo26s/best.pt"

# ==================== 🛡️ 散热保护配置 ====================
ENABLE_HEAT_PROTECTION = False  
RUN_DURATION_SEC = 600         
REST_DURATION_SEC = 60         

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

class DynamicDetector:
    """动态篮筐检测版本 - 每帧都实时检测篮筐位置"""
    
    def __init__(self, loaded_model, device, video_path, start_min, duration_min, heat_manager):
        self.model = loaded_model
        self.device = device
        self.video_path = video_path
        self.heat_manager = heat_manager
        
        self.clip_queue = queue.Queue()
        self.worker = ClipWorker(self.clip_queue, ROTATE_VIDEO_180)
        self.worker.start()
        
        # 预热模型
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

        # 动态检测状态
        self.last_interaction_ts = -10.0
        self.last_shot_ts = -10.0       
        self.shot_count = 0

        self.original_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.original_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.resize_scale = INFERENCE_SIZE / self.original_width
        self.detect_w = INFERENCE_SIZE
        self.detect_h = int(self.original_height * self.resize_scale)

    def run(self):
        video_name = os.path.basename(self.video_path)
        print(f"\n🎬 正在处理: {video_name} | 跳帧: {FRAME_SKIP} | 🚀 动态检测模式: ON")
        
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
                        
                        self.heat_manager['last_rest_time'] = time.time()
                        pbar.set_description("⚡️ [恢复全速运行]")

                # ==================== 🚀 跳帧逻辑 ====================
                if current_frame_idx % FRAME_SKIP != 0:
                    self.cap.grab()
                    current_frame_idx += 1
                    pbar.update(1)
                    continue

                # 需要检测的帧，真正解码
                ret, frame = self.cap.read()
                if not ret: break
                
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
                
                current_time = current_frame_idx / self.fps
                self._run_dynamic_detection(detect_frame, current_time)

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

    def _run_dynamic_detection(self, frame, current_time):
        """动态检测：每帧同时检测篮球和篮筐"""
        
        # 同时检测篮球和篮筐
        results = self.model.predict(
            frame, 
            verbose=False, 
            conf=0.1,  # 使用较低置信度，后续分别过滤
            iou=0.5, 
            imgsz=INFERENCE_SIZE, 
            device=self.device
        )
        
        if results[0].boxes is None or len(results[0].boxes) == 0:
            return
        
        boxes = results[0].boxes
        coords = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        classes = boxes.cls.cpu().numpy()
        
        # 分离篮球和篮筐
        ball_detections = []
        rim_detections = []
        
        for i, cls in enumerate(classes):
            conf = confs[i]
            box = coords[i]
            
            if cls == CLS_BALL and conf > CONF_THRES_BALL:
                ball_detections.append({
                    'box': box,
                    'conf': conf,
                    'center': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                })
            elif cls == CLS_RIM and conf > CONF_THRES_RIM:
                rim_detections.append({
                    'box': box,
                    'conf': conf,
                    'center': ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                })
        
        # 如果没有检测到篮筐或篮球，跳过
        if not rim_detections or not ball_detections:
            return
        
        # 选择置信度最高的篮筐
        best_rim = max(rim_detections, key=lambda x: x['conf'])
        
        # 检查每个篮球与篮筐的关系
        self._check_goal_with_dynamic_rim(ball_detections, best_rim, current_time)

    def _check_goal_with_dynamic_rim(self, ball_detections, rim_detection, current_time):
        """使用动态检测到的篮筐位置判断进球"""
        
        # 防止重复触发
        if current_time - self.last_shot_ts < SHOT_COOLDOWN:
            return
        
        rim_box = rim_detection['box']
        rx1, ry1, rx2, ry2 = rim_box
        
        # 动态计算三个区域
        high_line = ry1 - HIGH_ZONE_OFFSET  # 高位区线
        rim_zone = [rx1 - 10, ry1 - 10, rx2 + 10, ry2 + 10]  # 触框区
        goal_zone = [rx1 - 30, ry1 + 10, rx2 + 30, ry2 + GOAL_ZONE_OFFSET]  # 进球区
        
        ball_in_goal = False
        
        for ball in ball_detections:
            bx, by = ball['center']
            
            # 检查高位区或触框区
            is_high = by < high_line
            is_touching_rim = (rim_zone[0] < bx < rim_zone[2]) and (rim_zone[1] < by < rim_zone[3])
            
            if is_high or is_touching_rim:
                self.last_interaction_ts = current_time
            
            # 检查进球区
            if (goal_zone[0] < bx < goal_zone[2]) and (goal_zone[1] < by < goal_zone[3]):
                ball_in_goal = True
        
        # 判断是否进球
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

    # 🔥 全局散热管理器
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

            detector = DynamicDetector(
                loaded_model, 
                device, 
                path, 
                start_min, 
                MAX_PROCESS_MINUTES,
                GLOBAL_HEAT_MANAGER
            )
            detector.run()
            
        print("\n🎉🎉🎉 所有任务处理完成！")
        subprocess.run(["open", OUTPUT_DIR])

    except KeyboardInterrupt:
        print("\n\n⛔️ ----------------------------------------")
        print("⛔️ 用户全局中断，程序停止。")
        print("⛔️ ----------------------------------------")
