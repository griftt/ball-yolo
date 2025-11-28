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
START_FROM_MINUTES = 5    # 建议调整到你有漏检的时间段测试
MAX_PROCESS_MINUTES = 1      
ROTATE_VIDEO_180 = True  # 是否旋转180
# 2. 🎯 自动校准参数
CONF_THRES_RIM_INIT = 0.40   # 稍微降低篮筐门槛，确保能锁定
CALIBRATION_SAMPLES = 30     

# 3. ⚡️ 进球逻辑参数 (优化版)
# 逻辑改进：只要球接触过篮筐区域(Rim Zone) 或 高空区(High Zone)，并在短时间内进入深层得分区(Deep Goal Zone)，即判定进球。
HIGH_ZONE_OFFSET = 150       
GOAL_ZONE_OFFSET = 150       # 🔼 [修改] 增加到150px！防止快球直接穿过判定区
SHOT_WINDOW = 2.5            # 🔼 [修改] 稍微延长窗口，给颠球入网留时间

# 4. 🎬 剪辑参数
CLIP_PRE_TIME = 4.0          
CLIP_POST_TIME = 2.0         
SHOT_COOLDOWN = 2.0          # 📉 [修改] 降低冷却，防止连续补篮漏检

# 5. 🤖 模型与路径
MODEL_PATH = "./runs/train/yolo11_finetune_new_court/weights/best.mlpackage"
VIDEO_PATH = "/Users/grifftwu/Desktop/历史篮球/1126/111.mp4"
OUTPUT_DIR = "./outputs/auto_mps_clips_1126_optimized1"

# 6. 推理配置
CONF_THRES_BALL = 0.15       # 📉 [修改] 降到 0.15，捕捉模糊的快球
INFERENCE_SIZE = 1024        

CLS_BALL = 0
CLS_RIM = 1

# ==================== 系统初始化 ====================
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

class ClipWorker(threading.Thread):
    def __init__(self, task_queue):
        super().__init__()
        self.task_queue = task_queue
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
                tqdm.write(f"✅ [已保存] {os.path.basename(out_path)}")
            except Exception as e:
                logger.error(f"❌ 剪辑出错: {e}")
            finally:
                self.task_queue.task_done()

class AutoMPSDetector:
    def __init__(self, model_path, video_path, start_min, duration_min):
        self.video_path = video_path
        self.clip_queue = queue.Queue()
        self.worker = ClipWorker(self.clip_queue)
        self.worker.start()
        
        # 硬件检查
        if MODEL_PATH.endswith(".mlpackage"):
            self.device = 'cpu' # mlpackage 通常只能 CPU 跑检测
            print(f"⚠️ 使用 CoreML 模型 (CPU Mode)")
        elif torch.backends.mps.is_available():
            self.device = 'mps'
            print(f"⚡️ MPS 加速已开启")
        else:
            self.device = 'cpu'

        print(f"📦 加载模型: {model_path}")
        self.model = YOLO(model_path)
        
        # 预热
        dummy = np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8)
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

        # 🟢 逻辑变量
        self.is_calibrated = False
        self.calibration_buffer = [] 
        self.locked_hoop_box = None  
        
        # 判定状态
        self.last_interaction_ts = -10.0  # 只要球在篮筐附近就算交互
        self.last_shot_ts = -10.0       
        self.shot_count = 0
        
        # 区域
        self.rim_box = []   # [x1, y1, x2, y2] 篮筐本体
        self.high_line = 0  # y坐标
        self.goal_zone = [] # [x1, y1, x2, y2] 得分区

    def run(self):
        print(f"🚀 开始运行...")
        process_len = self.end_frame - self.start_frame
        pbar = tqdm(total=process_len, unit="frame", ncols=100)
        current_frame_idx = self.start_frame
        
        try:
            while True:
                if current_frame_idx >= self.end_frame: break

                ret, frame = self.cap.read()
                if not ret: break
                if ROTATE_VIDEO_180:
                    # 只要加这一行，YOLO 看到的就是正的，坐标也是正的，一切逻辑都会恢复正常
                    frame = cv2.rotate(frame, cv2.ROTATE_180)
                
                current_time = current_frame_idx / self.fps
                
                if not self.is_calibrated:
                    self._run_calibration(frame)
                else:
                    self._run_inference(frame, current_time)

                pbar.update(1)
                current_frame_idx += 1
                
        except KeyboardInterrupt:
            print("\n用户中断...")
        finally:
            pbar.close()
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
            
            # 🟢 [优化] 区域定义
            self.rim_box = [x1 - 10, y1 - 10, x2 + 10, y2 + 10]  # 篮筐本体(略微扩大)
            self.high_line = y1  # 篮筐上沿
            
            # 🟢 [优化] 得分区：左右放宽(+30px)，深度加深(+GOAL_ZONE_OFFSET)
            # 这样就算球速快穿过了60px，也穿不过150px的网
            self.goal_zone = [x1 - 30, y1 + 10, x2 + 30, y2 + GOAL_ZONE_OFFSET]
            
            self.is_calibrated = True
            tqdm.write(f"✅ 篮筐锁定! 坐标: {self.locked_hoop_box.astype(int)}")

    def _run_inference(self, frame, current_time):
        results = self.model.predict(
            frame, 
            verbose=False, 
            conf=0.01,           
            iou=0.5, 
            imgsz=INFERENCE_SIZE, 
            classes=[CLS_BALL],  
            device=self.device
        )
        
        self._check_zones_optimized(results, current_time)

    def _check_zones_optimized(self, results, current_time):
        """
        [优化版判定逻辑]
        条件1: 球在 'High Zone' (篮板上空) 
               OR 
               球在 'Rim Box' (球与篮筐重叠/接触 -> 解决上篮漏检)
        条件2: 球随后进入 'Goal Zone' (篮筐下方深处)
        """
        # 冷却
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
                    
                    # --- 阶段 1: 预备动作检测 (空中 或 接触篮筐) ---
                    
                    # A. 纯高空逻辑
                    is_high = cy < self.high_line
                    
                    # B. 接触篮筐逻辑 (解决上篮/扣篮)
                    # 检查球的中心点是否在篮筐盒子内
                    rx1, ry1, rx2, ry2 = self.rim_box
                    is_touching_rim = (rx1 < cx < rx2) and (ry1 < cy < ry2)
                    
                    if is_high or is_touching_rim:
                        self.last_interaction_ts = current_time
                        
                    # --- 阶段 2: 进球检测 (深层网袋) ---
                    gx1, gy1, gx2, gy2 = self.goal_zone
                    # x轴宽松判定，y轴严格判定(必须在篮筐线以下)
                    if (gx1 < cx < gx2) and (gy1 < cy < gy2):
                        ball_in_goal = True
            
            # --- 最终触发 ---
            if ball_in_goal:
                time_diff = current_time - self.last_interaction_ts
                # 允许的时间窗口：0.05s (极快球) ~ 2.5s (颠球)
                if 0.05 < time_diff < SHOT_WINDOW:
                    self.trigger_goal(current_time)

    def trigger_goal(self, current_time):
        self.shot_count += 1
        self.last_shot_ts = current_time
        
        tqdm.write(f"🏀 [进球!] 时间: {current_time:.2f}s | No.{self.shot_count}")
        
        filename = f"goal_{self.shot_count:03d}_{int(current_time)}s.mp4"
        save_path = os.path.join(OUTPUT_DIR, filename)
        
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