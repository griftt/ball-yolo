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
START_FROM_MINUTES = 55.0     # 从视频第几分钟开始跑？(例如 12.5)
MAX_PROCESS_MINUTES = 1     # 往后跑多久？(设为 None 则跑完为止)

# 2. 🎯 自动校准参数
CONF_THRES_RIM_INIT = 0.50   # 篮筐校准门槛 (要求清晰)
CALIBRATION_SAMPLES = 30     # 收集多少帧篮筐样本后锁定？(约1秒)

# 3. ⚡️ 进球逻辑参数 (Zone-Based Flash Shot)
# 逻辑：球必须先去过【高空预警区】，然后在【有效窗口】内落入【得分区】
HIGH_ZONE_OFFSET = 150       # 篮筐上沿往上 150px 为高空区
GOAL_ZONE_OFFSET = 60        # 篮筐下沿往下 60px 为得分区
SHOT_WINDOW = 2.0            # 高空->入网的最大允许时间间隔(秒)

# 4. 🎬 剪辑参数
CLIP_PRE_TIME = 5.0          # 进球前截取秒数
CLIP_POST_TIME = 2.0         # 进球后截取秒数
SHOT_COOLDOWN = 3.0          # 进球冷却时间(秒)

# 5. 🤖 模型与路径 (请修改这里)
MODEL_PATH = "./runs/train/yolo11_finetune_new_court/weights/best.pt"
VIDEO_PATH = "/Users/grifftwu/Desktop/历史篮球/1112/1112.mov"
OUTPUT_DIR = "./outputs/auto_mps_clips_1112"

# 6. 推理配置 (针对 M3 Pro 优化)
CONF_THRES_BALL = 0.2       # 极低门槛，捕捉模糊虚影球
INFERENCE_SIZE = 1024        # 高清推理，保证远距离小球可见

# 类别 ID (根据你的训练集)
CLS_BALL = 0
CLS_RIM = 1

# ==================== 系统初始化 ====================
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
# 屏蔽 YOLO 冗余日志
logging.getLogger("ultralytics").setLevel(logging.ERROR)

class ClipWorker(threading.Thread):
    """后台剪辑工人：负责调用 FFmpeg 处理视频队列"""
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
                # FFmpeg 极速流剪辑 (无损不重编码)
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
        
        # 初始化队列
        self.clip_queue = queue.Queue()
        self.worker = ClipWorker(self.clip_queue)
        self.worker.start()
        
        # 1. 硬件检查 (MPS)
        if torch.backends.mps.is_available() and  not MODEL_PATH.endswith(".mlpackage"):
            self.device = 'mps'
            print(f"⚡️ MPS 加速已开启 (M3 Pro 性能全开)")
        else:
            self.device = 'cpu'
            print(f"⚠️ 警告: 未检测到 MPS，将使用 CPU 运行")

        print(f"📦 正在加载模型: {model_path}...")
        self.model = YOLO(model_path)
        
        # GPU 预热 (防止第一帧卡顿)
        print("🔥 正在预热 GPU...")
        dummy = np.zeros((INFERENCE_SIZE, INFERENCE_SIZE, 3), dtype=np.uint8)
        self.model.predict(dummy, device=self.device, verbose=False, imgsz=INFERENCE_SIZE)
        
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 计算跳转帧
        self.start_frame = int(start_min * 60 * self.fps)
        if self.start_frame >= self.total_frames:
            print("❌ 起始时间超过视频总长度")
            exit()
        
        print(f"⏩ 跳转至: {start_min}分 ({self.start_frame}帧)")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        
        # 计算结束帧
        if duration_min is None:
            self.end_frame = self.total_frames
        else:
            self.end_frame = min(self.total_frames, self.start_frame + int(duration_min * 60 * self.fps))

        # 🟢 自动校准变量
        self.is_calibrated = False
        self.calibration_buffer = [] # 存储篮筐坐标样本
        self.locked_hoop_box = None  # 最终锁定的篮筐 [x1, y1, x2, y2]
        
        # 🟢 进球逻辑变量
        self.last_high_ball_ts = -10.0  # 上次球在高空的时间
        self.last_shot_ts = -10.0       # 上次触发进球的时间
        self.shot_count = 0
        
        # 区域 (校准后生成)
        self.high_zone_y = 0
        self.goal_zone = []

    def run(self):
        print(f"🚀 开始运行 | 区间: {START_FROM_MINUTES}分 -> {START_FROM_MINUTES + (MAX_PROCESS_MINUTES or 0)}分")
        
        process_len = self.end_frame - self.start_frame
        pbar = tqdm(total=process_len, unit="frame", ncols=100)
        current_frame_idx = self.start_frame
        
        try:
            while True:
                # 结束检查
                if current_frame_idx >= self.end_frame: break

                ret, frame = self.cap.read()
                if not ret: break
                
                current_time = current_frame_idx / self.fps
                
                # 🟢 核心分支：校准模式 vs 推理模式
                if not self.is_calibrated:
                    self._run_calibration(frame)
                else:
                    self._run_inference(frame, current_time)

                pbar.update(1)
                current_frame_idx += 1
                
        except KeyboardInterrupt:
            print("\n用户手动中断...")
        finally:
            pbar.close()
            self.cap.release()
            self.shutdown()

    def _run_calibration(self, frame):
        """阶段一：自动寻找并锁定篮筐"""
        # 只检测篮筐 (Class 1)
        results = self.model.predict(
            frame, verbose=False, conf=0.1, iou=0.5, 
            imgsz=INFERENCE_SIZE, classes=[CLS_RIM], device=self.device
        )
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            best_rim = None
            max_conf = 0.0
            
            # 寻找当前帧置信度最高的篮筐
            for box in boxes:
                conf = float(box.conf[0])
                if conf > max_conf:
                    max_conf = conf
                    best_rim = box.xyxy[0].cpu().numpy()
            
            # 只有置信度达标才纳入样本
            if best_rim is not None and max_conf > CONF_THRES_RIM_INIT:
                self.calibration_buffer.append(best_rim)
                
                if len(self.calibration_buffer) % 10 == 0:
                    tqdm.write(f"🔎 正在校准篮筐... ({len(self.calibration_buffer)}/{CALIBRATION_SAMPLES})")

        # 样本足够，执行锁定
        if len(self.calibration_buffer) >= CALIBRATION_SAMPLES:
            # 取中位数，消除抖动
            self.locked_hoop_box = np.median(self.calibration_buffer, axis=0)
            
            # 🟢 生成判定区域
            x1, y1, x2, y2 = map(int, self.locked_hoop_box)
            
            # 1. 高空警戒线 (篮筐上沿)
            self.high_zone_y = y1 
            
            # 2. 得分区域 (篮筐范围 + 垂直延伸)
            self.goal_zone = [x1 - 20, y1 + 10, x2 + 20, y2 + GOAL_ZONE_OFFSET]
            
            self.is_calibrated = True
            tqdm.write(f"✅ 篮筐已锁定! 坐标: {self.locked_hoop_box.astype(int)}")
            tqdm.write(f"🚀 切换至进球检测模式 (极速版)...")

    def _run_inference(self, frame, current_time):
        """阶段二：极速检测篮球"""
        # 只检测篮球 (Class 0)，忽略篮筐
        results = self.model.predict(
            frame, 
            verbose=False, 
            conf=0.01,           # 极低门槛，不错过任何虚影
            iou=0.5, 
            imgsz=INFERENCE_SIZE, 
            classes=[CLS_BALL],  # 只看球
            device=self.device
        )
        
        self._check_zones(results, current_time)

    def _check_zones(self, results, current_time):
        """区域关联逻辑：不依赖连续跟踪，只看时空关系"""
        # 冷却期检查
        if current_time - self.last_shot_ts < SHOT_COOLDOWN: return

        if results[0].boxes is not None:
            boxes = results[0].boxes
            # 从 GPU 获取数据
            coords = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            
            ball_in_goal_zone = False
            
            for i, conf in enumerate(confs):
                # 过滤掉极低分的噪音
                if conf > CONF_THRES_BALL:
                    x1, y1, x2, y2 = coords[i]
                    cy = (y1 + y2) / 2
                    cx = (x1 + x2) / 2
                    
                    # 1. 更新高空计时器
                    # 只要有球出现在篮板上方，就认为可能要投篮了
                    if cy < self.high_zone_y:
                        self.last_high_ball_ts = current_time
                        
                    # 2. 检查得分区
                    gx1, gy1, gx2, gy2 = self.goal_zone
                    if (gx1 < cx < gx2) and (gy1 < cy < gy2):
                        ball_in_goal_zone = True
            
            # ⚡️ 判定核心：现在进网了 AND 不久前在天上
            if ball_in_goal_zone:
                time_diff = current_time - self.last_high_ball_ts
                
                # 0.1s < 间隔 < 2.0s
                if 0.1 < time_diff < SHOT_WINDOW:
                    self.trigger_goal(current_time)

    def trigger_goal(self, current_time):
        self.shot_count += 1
        self.last_shot_ts = current_time
        
        tqdm.write(f"🏀 [进球触发] 时间: {current_time:.2f}s | No.{self.shot_count}")
        
        # 生成文件名
        filename = f"goal_{self.shot_count:03d}_{int(current_time)}s.mp4"
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        # 计算剪辑区间
        start_cut = max(0, current_time - CLIP_PRE_TIME)
        duration = CLIP_PRE_TIME + CLIP_POST_TIME
        
        # 发送给后台工人
        self.clip_queue.put((self.video_path, start_cut, duration, save_path))

    def shutdown(self):
        print(f"\n🏁 扫描结束！共发现: {self.shot_count} 个进球")
        if not self.clip_queue.empty():
            print(f"⏳ 正在处理剩余的 {self.clip_queue.qsize()} 个视频，请稍候...")
        
        self.clip_queue.join()
        self.worker.running = False
        print(f"✅ 全部完成。文件夹: {OUTPUT_DIR}")
        # Mac 自动打开输出文件夹
        subprocess.run(["open", OUTPUT_DIR])

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 错误: 找不到模型文件 {MODEL_PATH}")
        print("请检查路径是否正确！")
    else:
        detector = AutoMPSDetector(
            MODEL_PATH, 
            VIDEO_PATH, 
            start_min=START_FROM_MINUTES, 
            duration_min=MAX_PROCESS_MINUTES
        )
        detector.run()