import os
import cv2
import time
import logging
import threading
import subprocess
import queue
import numpy as np
from collections import deque, defaultdict
from ultralytics import YOLO
from tqdm import tqdm

# ==================== ⚙️ 配置区域 ====================

# 1. ⏱️ 时间设置
START_FROM_MINUTES = 5.0    # 从视频的第几分钟开始检测？(如 12.5)
MAX_PROCESS_MINUTES = 5    # 往后检测多久？

# 2. 🎬 剪辑规则 (你的新需求)
CLIP_PRE_TIME = 5.0         # 进球前 5 秒
CLIP_POST_TIME = 2.0        # 进球后 2 秒
SHOT_COOLDOWN = 3.0         # 进球后冷却时间 (防止重复触发)

# 3. 🏀 物理外挂 (篮筐坐标)
# [x1, y1, x2, y2] - 即使视频中途篮筐被遮挡也能识别
LOCKED_HOOP_COORDS = [845, 88, 1023, 172]

# 4. 路径与模型
MODEL_PATH = "./runs/train/yolo11_hd_optimized/weights/best.pt"
VIDEO_PATH = "/Users/grifftwu/Desktop/历史篮球/1122/ball.mov"
OUTPUT_DIR = "./outputs/realtime_clips"

# 5. 推理参数
CONF_THRES_BALL = 0.25      # 极低门槛
INFERENCE_SIZE = 1024       
FRAME_STEP = 1              # 1=最准, 2=更快
CLS_BALL = 0

# ==================== 🚀 系统初始化 ====================
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)
logging.getLogger("ultralytics").setLevel(logging.ERROR)

class ClipWorker(threading.Thread):
    """
    后台剪辑工人：一旦收到任务，立刻执行 FFmpeg，绝不拖延
    """
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
                # FFmpeg 极速剪辑
                # 注意：对于本地文件，ffmpeg 可以直接切出未来的2秒，不需要等待
                cmd = [
                    "ffmpeg", "-nostdin", "-y",
                    "-ss", f"{start:.3f}",
                    "-i", source,
                    "-t", f"{duration:.3f}",
                    "-c", "copy",                # 流复制，毫秒级速度
                    "-avoid_negative_ts", "1",
                    "-loglevel", "error",
                    out_path
                ]
                subprocess.run(cmd, check=True)
                
                # 🟢 剪辑完成，立刻在控制台反馈，不用等主程序跑完
                filename = os.path.basename(out_path)
                tqdm.write(f"✅ [视频生成] {filename} (前{CLIP_PRE_TIME}s + 后{CLIP_POST_TIME}s)")
                
            except Exception as e:
                logger.error(f"❌ 剪辑出错: {e}")
            finally:
                self.task_queue.task_done()

class RealtimeDetector:
    def __init__(self, model_path, video_path, start_min, duration_min):
        self.video_path = video_path
        
        # 启动后台剪辑线程
        self.clip_queue = queue.Queue()
        self.worker = ClipWorker(self.clip_queue)
        self.worker.start()
        
        print(f"⚡️ 加载模型 (M3 Pro)...")
        self.model = YOLO(model_path)
        
        self.cap = cv2.VideoCapture(video_path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # 跳转时间
        self.start_frame = int(start_min * 60 * self.fps)
        if self.start_frame >= total_frames:
            print("❌ 起始时间超过视频长度")
            exit()
            
        print(f"⏩ 跳转至: {start_min}分 ({self.start_frame}帧)")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
        
        # 计算结束点
        if duration_min is None:
            self.end_frame = total_frames
        else:
            self.end_frame = min(total_frames, self.start_frame + int(duration_min * 60 * self.fps))
            
        # 状态机数据
        self.ball_tracks = defaultdict(lambda: {
            'state': 0, 'history': deque(maxlen=40), 
            'last_update_ts': 0, 'max_height': float('inf')
        })
        
        self.locked_hoop_box = np.array(LOCKED_HOOP_COORDS)
        self.last_shot_time = -10
        self.shot_count = 0

    def run(self):
        print(f"🚀 开始检测 | 规则: 进球时刻 [ 前{CLIP_PRE_TIME}s ~ 后{CLIP_POST_TIME}s ]")
        
        frames_to_process = self.end_frame - self.start_frame
        pbar = tqdm(total=frames_to_process, unit="frame", ncols=100)
        
        current_frame_idx = self.start_frame
        
        try:
            while True:
                if current_frame_idx >= self.end_frame: break

                ret, frame = self.cap.read()
                if not ret: break
                
                # 跳帧
                if current_frame_idx % FRAME_STEP != 0:
                    current_frame_idx += 1
                    pbar.update(1)
                    continue
                
                current_time = current_frame_idx / self.fps
                
                # 推理 (只看球)
                results = self.model.track(
                    frame, persist=True, verbose=False, 
                    conf=0.01, iou=0.5, imgsz=INFERENCE_SIZE, 
                    classes=[CLS_BALL], device='mps'
                )
                
                self._run_logic(results, current_time)

                pbar.update(FRAME_STEP)
                current_frame_idx += 1
                
        except KeyboardInterrupt:
            print("\n用户中断...")
        finally:
            pbar.close()
            self.cap.release()
            self.shutdown()

    def _run_logic(self, results, current_time):
        if results[0].boxes.id is not None:
            boxes = results[0].boxes
            ids = boxes.id.cpu().numpy().astype(int)
            coords = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            
            for i, conf in enumerate(confs):
                if conf > CONF_THRES_BALL:
                    ball_id = ids[i]
                    ball_box = coords[i]
                    if self.check_logic(ball_id, ball_box, current_time):
                        self.trigger_goal(current_time)

    def check_logic(self, ball_id, ball_box, current_time):
        if current_time - self.last_shot_time < SHOT_COOLDOWN: return False
        
        track = self.ball_tracks[ball_id]
        bx, by = (ball_box[0]+ball_box[2])/2, (ball_box[1]+ball_box[3])/2
        track['history'].append((int(bx), int(by)))
        track['last_update_ts'] = current_time
        
        hx1, hy1, hx2, hy2 = self.locked_hoop_box
        prev_y = track['history'][-2][1] if len(track['history']) > 1 else by
        curr_y = by
        
        # 状态机逻辑
        if track['state'] == 0:
            if curr_y < prev_y and curr_y < (hy2 + 150):
                track['state'] = 1
                track['max_height'] = curr_y

        elif track['state'] == 1:
            track['max_height'] = min(track['max_height'], curr_y)
            if curr_y > prev_y + 1: 
                hoop_center_y = (hy1 + hy2) / 2
                if track['max_height'] < hoop_center_y + 80:
                    track['state'] = 2
                else:
                    track['state'] = 0

        elif track['state'] == 2:
            hit_x = (hx1 - 40) < bx < (hx2 + 40)
            hit_y = hy1 < by < hy2 + 40
            if hit_x and hit_y:
                track['state'] = 0
                return True # 🎯 触发进球！
            if by > hy2 + 200: track['state'] = 0
        
        if current_time - track['last_update_ts'] > 2.0: track['state'] = 0
        return False

    def trigger_goal(self, current_time):
        """进球后，立即计算时间段并发送剪辑任务"""
        self.shot_count += 1
        self.last_shot_time = current_time
        
        tqdm.write(f"🏀 [进球触发] 时间: {current_time:.2f}s | 正在剪辑 (前{CLIP_PRE_TIME}s+后{CLIP_POST_TIME}s)...")
        
        filename = f"goal_{self.shot_count:03d}_{int(current_time)}s.mp4"
        save_path = os.path.join(OUTPUT_DIR, filename)
        
        # 🟢 核心逻辑修改：
        # 开始时间 = 当前时间 - 5秒
        start_cut = max(0, current_time - CLIP_PRE_TIME)
        # 总时长 = 5秒 + 2秒
        total_duration = CLIP_PRE_TIME + CLIP_POST_TIME
        
        # 放入队列，后台线程会立刻处理
        self.clip_queue.put((self.video_path, start_cut, total_duration, save_path))

    def shutdown(self):
        print(f"\n🏁 扫描结束！共发现: {self.shot_count} 个进球")
        if not self.clip_queue.empty():
            print(f"⏳ 等待最后 {self.clip_queue.qsize()} 个视频生成...")
        
        self.clip_queue.join()
        self.worker.running = False
        print(f"✅ 全部完成。请查看: {OUTPUT_DIR}")
        subprocess.run(["open", OUTPUT_DIR])

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型: {MODEL_PATH}")
    else:
        detector = RealtimeDetector(
            MODEL_PATH, 
            VIDEO_PATH, 
            start_min=START_FROM_MINUTES, 
            duration_min=MAX_PROCESS_MINUTES
        )
        detector.run()