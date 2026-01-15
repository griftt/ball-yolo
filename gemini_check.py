# -*- coding: utf-8 -*-
import cv2
import torch
import numpy as np
from ultralytics import YOLO

# ==================== ⚙️ 用户配置区 ====================
# 模型路径
MODEL_PATH = "/Users/grifftwu/IdeaProjects/ball-yolo/runs/train/yolo11n_640_train_hd/weights/best.pt"
# 视频路径
VIDEO_PATH = "/Users/grifftwu/ball/test2.mp4"

# ⏱️ [起始时间] 从第几分钟开始看？
START_MIN = 0.0  

# 🔍 检测参数
CONF_THRES_BALL = 0.15   # 篮球置信度
CONF_THRES_RIM = 0.10    # 篮筐置信度 (设低一点以防漏检)
INFERENCE_SIZE = 1024    # 推理尺寸 (1024 或 1280 能显著提升远处篮筐的检测率)

# 📐 逻辑判断参数 (基于篮筐宽度的比例因子)
# 解释: 如果篮筐宽 100px, 那么判定高度就是 100 * 1.3 = 130px
NET_HEIGHT_RATIO = 1.3   # 篮网高度相对于篮筐宽度的比例
UPPER_LINE_RATIO = 0.5   # 高空线相对于篮筐宽度的比例
# =====================================================

def run_debug():
    print(f"📦 加载模型: {MODEL_PATH}")
    # 强制开启 MPS 加速 (Mac M系列芯片)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    
    # 加载模型
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 🟢 1. 跳转到指定时间
    start_frame = int(START_MIN * 60 * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    print(f"\n🚀以此开始播放: {START_MIN}分 ({start_frame}/{total_frames}帧)")
    print(f"🖥️ 推理设备: {device.upper()} | 图片尺寸: {INFERENCE_SIZE}")
    print("------------------------------------------------")
    print("⌨️  快捷键说明:")
    print("   [空格]  暂停/继续 (暂停时才可单帧调试)")
    print("   [F]     下一帧 (逐帧分析)")
    print("   [D]     快进 5秒 ⏩")
    print("   [A]     快退 5秒 ⏪")
    print("   [Q]     退出")
    print("------------------------------------------------")

    paused = False
    
    # 预读取第一帧
    ret, frame = cap.read()
    if not ret:
        print("❌ 无法读取视频，请检查路径。")
        return

    while True:
        # 复制画面用于绘图，不影响原图
        debug_frame = frame.copy()
        
        # -----------------------------------------------------------
        # 🧠 YOLO 核心推理 (仅在播放或手动单帧时运行)
        # persist=True: 保持 ID 追踪
        # -----------------------------------------------------------
        results = model.track(debug_frame, 
                              conf=0.01,         # 极低阈值，我们在下面自己过滤
                              device=device, 
                              persist=True, 
                              verbose=False, 
                              imgsz=INFERENCE_SIZE) # 关键：大尺寸检测
        
        rim_box = None 
        
        if results[0].boxes is not None:
            boxes = results[0].boxes
            coords = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            clss = boxes.cls.cpu().numpy()
            ids = boxes.id.cpu().numpy() if boxes.id is not None else [-1] * len(coords)
            
            for i, conf in enumerate(confs):
                cls_id = int(clss[i])
                obj_id = int(ids[i])
                x1, y1, x2, y2 = map(int, coords[i])
                
                # 🏀 篮球 (Class 0)
                if cls_id == 0:
                    if conf > CONF_THRES_BALL:
                        color = (0, 140, 255) # 橙色
                        label = f"ID:{obj_id} {conf:.2f}"
                        # 画框
                        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                        # 画标签背景
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                        cv2.rectangle(debug_frame, (x1, y1 - 20), (x1 + label_size[0], y1), color, -1)
                        cv2.putText(debug_frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
                    else:
                        # 灰色显示被忽略的低置信度球
                        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), (200, 200, 200), 1)
                        
                # 🥅 篮筐 (Class 1)
                elif cls_id == 1:
                    if conf > CONF_THRES_RIM:
                        # 找到置信度最高的那个作为“主篮筐”
                        if rim_box is None or conf > rim_box[4]: 
                            rim_box = [x1, y1, x2, y2, conf] # 记录坐标和置信度

                        color = (0, 255, 0) # 绿色
                        cv2.rectangle(debug_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(debug_frame, f"Rim {conf:.2f}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # -----------------------------------------------------------
        # 📐 动态区域绘制 (Adaptive Geometry)
        # -----------------------------------------------------------
        if rim_box is not None:
            rx1, ry1, rx2, ry2, _ = rim_box
            
            # 1. 计算篮筐当前的像素宽度
            rim_width = rx2 - rx1
            
            # 2. 🔵 蓝线 (Upper Threshold Line) 
            # 逻辑：球必须从这条线上面落下来。位置在篮筐上方 0.5 倍宽度处。
            upper_line_y = int(ry1 - (rim_width * UPPER_LINE_RATIO))
            cv2.line(debug_frame, (0, upper_line_y), (debug_frame.shape[1], upper_line_y), (255, 0, 0), 2)
            cv2.putText(debug_frame, "UPPER THRESHOLD", (10, upper_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # 3. 🔴 红框 (Net Zone / Goal Zone)
            # 逻辑：篮网区域。高度是宽度的 1.3 倍。左右收缩 10% 以避免擦边误判。
            margin_x = int(rim_width * 0.1)
            
            gx1 = rx1 + margin_x
            gy1 = ry1 + int(rim_width * 0.2) # 从篮圈稍微靠下一点开始算
            gx2 = rx2 - margin_x
            gy2 = ry1 + int(rim_width * NET_HEIGHT_RATIO)
            
            cv2.rectangle(debug_frame, (gx1, gy1), (gx2, gy2), (0, 0, 255), 2)
            cv2.putText(debug_frame, "GOAL ZONE", (gx1, gy2+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

        # -----------------------------------------------------------
        # 📺 显示处理
        # -----------------------------------------------------------
        # 缩放显示 (避免 4K 视频撑爆屏幕)
        display_h = 800
        scale = display_h / debug_frame.shape[0]
        display_w = int(debug_frame.shape[1] * scale)
        small_frame = cv2.resize(debug_frame, (display_w, display_h))
        
        # 叠加状态文字
        curr_sec = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
        status_text = "PAUSED (Press Space)" if paused else "PLAYING"
        status_color = (0, 0, 255) if paused else (0, 255, 0)
        
        cv2.putText(small_frame, f"[{status_text}]", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
        cv2.putText(small_frame, f"Time: {curr_sec/60:.2f} min", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow('YOLO Logic Inspector', small_frame)
        
        # -----------------------------------------------------------
        # 🎮 键盘控制逻辑
        # -----------------------------------------------------------
        if paused:
            # 暂停状态：无限等待按键，不跑循环，不跑推理 -> 省电
            key = cv2.waitKey(0) & 0xFF
        else:
            # 播放状态：只等 1ms
            key = cv2.waitKey(1) & 0xFF
            if key == 255: # 无按键
                ret, frame = cap.read()
                if not ret: 
                    print("视频结束")
                    break

        # 按键映射
        if key == ord('q'): 
            break
        elif key == 32: # Space
            paused = not paused
        elif key == ord('f'): # F - 下一帧
            ret, frame = cap.read()
            if not ret: break
            paused = True # 强制进入暂停，方便看结果
        elif key == ord('d'): # D - 快进
            curr_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, curr_pos + (5 * fps))
            ret, frame = cap.read()
        elif key == ord('a'): # A - 快退
            curr_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
            cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, curr_pos - (5 * fps)))
            ret, frame = cap.read()

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_debug()