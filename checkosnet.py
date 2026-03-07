# -*- coding: utf-8 -*-
"""
🏀 篮球进球智能分类系统 Pro 版 (YOLO + DINOv2 + 特征池算法)
优化点：
1. 特征池 (Feature Bank)：每个球员保存多个角度特征，解决正背脸差异。
2. 躯干裁剪 (Torso Crop)：跳过头部和足部，只取球衣区域，消除背景干扰。
3. 高精度阈值：针对 DINOv2 特性优化的相似度逻辑。
"""

import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os
from collections import deque
from ultralytics import YOLO
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

# ==================== ⚙️ 全局配置 ====================
VIDEO_PATH = "outputs/CCC.mp4"
MODEL_PATH = "runs/detect/runs/train/yolo26s_640_train_hd_person/weights/best.pt"
OUTPUT_DIR = "output_pro_highlights"

CLASS_BALL = 0
CLASS_RIM = 1
CLASS_PERSON = 2

class Config:
    CONF_BALL = 0.15
    CONF_RIM = 0.15
    CONF_PERSON = 0.35 # 提高人的置信度，减少误检
    
    # 进球判定
    INTERACTION_DIST = 0.20
    EVENT_COOLDOWN = 3.0
    ZONE_BELOW_HEIGHT = 0.08
    
    # 射手回溯
    HISTORY_SEC = 4.0
    SEARCH_START = 2.5
    SEARCH_END = 0.5
    POCKET_DIST = 0.15
    
    # --- 分类算法优化参数 ---
    # DINOv2 向量非常聚集，不同人相似度也可能在 0.8 以上
    # 建议阈值：0.88 - 0.93。如果总是一个人，就调高；如果总是分出新人，就调低。
    REID_THRESHOLD = 0.90      
    FEATURE_BANK_SIZE = 5     # 每个球员保存 5 张不同角度的“特征名片”

# ==================== 🧠 DINOv2 提取器 ====================

class DinoReIDExtractor:
    def __init__(self, model_name='facebook/dinov2-small'):
        self.device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def extract(self, image_numpy):
        """输入裁剪后的球衣图像，输出 L2 归一化的特征向量"""
        img_rgb = cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        inputs = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            feature = outputs.last_hidden_state[:, 0, :]
            feature = F.normalize(feature, p=2, dim=1)
        return feature.squeeze().cpu().numpy()

# ==================== 🗄️ 球员数据库 (特征池算法) ====================

class PlayerDatabase:
    def __init__(self):
        # 结构: {"Player_A": [vec1, vec2, ...]}
        self.players = {}  
        self.next_id = ord('A') 

    def identify(self, new_feature):
        """多点比对算法：只要新特征与池中任何一个历史特征吻合，即判定为同一个人"""
        if not self.players:
            return self._create_new_player(new_feature)

        best_match_name = None
        max_sim = -1.0

        # 1. 遍历库中所有球员
        for name, feature_bank in self.players.items():
            # 2. 与该球员的所有历史角度（正面、背面、侧面）进行对比
            for known_feature in feature_bank:
                sim = np.dot(new_feature, known_feature)
                if sim > max_sim:
                    max_sim = sim
                    best_match_name = name

        # 3. 判定阈值
        if max_sim > Config.REID_THRESHOLD:
            print(f"🔄 匹配成功: {best_match_name} (最高相似度: {max_sim:.3f})")
            # 如果池子还没满，就把这个新角度的特征也存进去，丰富档案
            if len(self.players[best_match_name]) < Config.FEATURE_BANK_SIZE:
                self.players[best_match_name].append(new_feature)
            return best_match_name
        else:
            print(f"🆕 相似度低 ({max_sim:.3f})，创建新球员...")
            return self._create_new_player(new_feature)

    def _create_new_player(self, feature):
        name = f"Player_{chr(self.next_id)}"
        self.players[name] = [feature] 
        self.next_id += 1
        os.makedirs(os.path.join(OUTPUT_DIR, name), exist_ok=True)
        return name

# ==================== 🎯 射手锁定逻辑 ====================

class ShooterCandidate:
    def __init__(self, track_id, box):
        self.track_id = track_id
        self.current_box = box
        self.score = 0
        self.best_frame_img = None
        self.best_box = None
        self.max_area = 0

    def update_best_frame(self, frame_img, box):
        area = box['width'] * box['height']
        if area > self.max_area:
            self.max_area = area
            self.best_frame_img = frame_img.copy()
            self.best_box = box

def extract_and_classify(goal_time, history, frame_w, frame_h, goal_idx, extractor, db):
    # 1. 回溯筛选
    start, end = goal_time - Config.SEARCH_START, goal_time - Config.SEARCH_END
    frames = [f for f in history if start <= f['timestamp'] <= end]
    if not frames: return

    active_tracks = []
    # 2. 持球积分逻辑
    for data in frames:
        ball, persons, img = data['ball'], data['persons'], data['image']
        curr_tracks = []
        for p_box in persons:
            # 简单的 IoU 追踪同一人
            best_iou, best_t = 0, None
            for t in active_tracks:
                xA, yA = max(t.current_box['minX'], p_box['minX']), max(t.current_box['minY'], p_box['minY'])
                xB, yB = min(t.current_box['maxX'], p_box['maxX']), min(t.current_box['maxY'], p_box['maxY'])
                inter = max(0, xB - xA) * max(0, yB - yA)
                iou = inter / (t.current_box['width']*t.current_box['height'] + p_box['width']*p_box['height'] - inter)
                if iou > best_iou and iou > 0.3: best_iou, best_t = iou, t
            
            if best_t:
                best_t.current_box = p_box
                curr_tracks.append(best_t)
            else:
                new_t = ShooterCandidate(len(active_tracks), p_box)
                active_tracks.append(new_t)
                curr_tracks.append(new_t)
        
        if ball:
            b_c = (ball['centerX'], ball['centerY'])
            for t in curr_tracks:
                p_p = (t.current_box['centerX'], t.current_box['minY'] + t.current_box['height'] * 0.25)
                if np.hypot(b_c[0]-p_p[0], b_c[1]-p_p[1]) < Config.POCKET_DIST:
                    t.score += 1
                    t.update_best_frame(img, t.current_box)

    # 3. 锁定胜出者
    winners = [t for t in active_tracks if t.score >= 3]
    if not winners: return print("⚠️ 未锁定明确射手")
    winner = max(winners, key=lambda x: x.score)

    # 4. 【核心升级：躯干裁剪】
    # 排除头顶灯光和地板干扰，只取球衣区域
    x1, y1, x2, y2 = int(winner.best_box['minX']*frame_w), int(winner.best_box['minY']*frame_h), \
                     int(winner.best_box['maxX']*frame_w), int(winner.best_box['maxY']*frame_h)
    
    h_total = y2 - y1
    w_total = x2 - x1
    # 裁剪区域：高度从 15% 到 70% (躯干)，宽度收缩 10%
    ty1, ty2 = int(y1 + h_total * 0.15), int(y1 + h_total * 0.75)
    tx1, tx2 = int(x1 + w_total * 0.10), int(x2 - w_total * 0.10)
    
    torso_crop = winner.best_frame_img[max(0, ty1):min(frame_h, ty2), max(0, tx1):min(frame_w, tx2)]
    
    if torso_crop.size == 0: return

    # 5. DINOv2 识别与分类
    feat = extractor.extract(torso_crop)
    p_name = db.identify(feat)
    
    # 6. 保存结果
    save_img = winner.best_frame_img.copy()
    cv2.rectangle(save_img, (x1, y1), (x2, y2), (0, 0, 255), 4)
    cv2.putText(save_img, f"{p_name} (Score:{winner.score})", (x1, y1-15), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)
    
    path = os.path.join(OUTPUT_DIR, p_name, f"goal_{goal_idx}_{p_name}.jpg")
    cv2.imwrite(path, save_img)
    print(f"📸 结果保存至: {path}")

# ==================== 🚀 主循环 ====================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    yolo = YOLO(MODEL_PATH)
    reid = DinoReIDExtractor()
    db = PlayerDatabase()
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    fps, w, h = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    history = deque(maxlen=int(Config.HISTORY_SEC * fps))
    
    rim_box, calib_buf, below_z = None, [], None
    last_goal_t, last_inter_t, goal_cnt = -10, -10, 0

    print("▶️ 开始 Pro 级智能剪辑分析...")

    while True:
        ret, frame = cap.read()
        if not ret: break
        ts = cap.get(cv2.CAP_PROP_POS_FRAMES) / fps
        res = yolo.predict(frame, conf=0.01, device=device, verbose=False, imgsz=1024)
        
        ball, persons = None, []
        if res[0].boxes is not None:
            for box in res[0].boxes:
                cls, conf = int(box.cls[0]), float(box.conf[0])
                b = box.xyxy[0].cpu().numpy()
                norm = {'minX': b[0]/w, 'minY': b[1]/h, 'maxX': b[2]/w, 'maxY': b[3]/h,
                        'width': (b[2]-b[0])/w, 'height': (b[3]-b[1])/h,
                        'centerX': (b[0]+b[2])/2/w, 'centerY': (b[1]+b[3])/2/h}
                
                if cls == CLASS_RIM and conf > Config.CONF_RIM:
                    if rim_box is None:
                        calib_buf.append(norm)
                        if len(calib_buf) >= 30:
                            avg_x, avg_y, avg_w = np.mean([x['minX'] for x in calib_buf]), \
                                                  np.mean([x['maxY'] for x in calib_buf]), \
                                                  np.mean([x['width'] for x in calib_buf])
                            rim_box = norm
                            below_z = {'minX': avg_x-0.01, 'maxX': avg_x+avg_w+0.01, 
                                       'minY': avg_y, 'maxY': avg_y+Config.ZONE_BELOW_HEIGHT}
                            print("✅ 篮筐校准完成")
                elif cls == CLASS_BALL and conf > Config.CONF_BALL: ball = norm
                elif cls == CLASS_PERSON and conf > Config.CONF_PERSON: persons.append(norm)

        history.append({'timestamp': ts, 'image': frame.copy(), 'ball': ball, 'persons': persons})

        # 判定进球逻辑
        if ball and rim_box and (ts - last_goal_t > Config.EVENT_COOLDOWN):
            dist = np.hypot(ball['centerX']-rim_box['centerX'], ball['centerY']-rim_box['centerY'])
            if dist < Config.INTERACTION_DIST: last_inter_t = ts
            
            if below_z['minX'] <= ball['centerX'] <= below_z['maxX'] and \
               below_z['minY'] <= ball['centerY'] <= below_z['maxY']:
                if abs(ts - last_inter_t) <= 2.5:
                    last_goal_t = ts
                    goal_cnt += 1
                    extract_and_classify(ts, history, w, h, goal_cnt, reid, db)

    cap.release()
    print(f"✅ 处理完成，共识别到 {goal_cnt} 个进球。")

if __name__ == "__main__":
    main()