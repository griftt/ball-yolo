# -*- coding: utf-8 -*-
import os
import subprocess
import cv2
import numpy as np

# ==================== ⚙️ 配置区域 ====================
INPUT_FOLDER = "./outputs/auto_mps_clips_1112_optimized_rotated"
OUTPUT_FILE = "./outputs/Final_Highlight_Timeline_Accurate.mp4"

# 🎨 视觉配置
FOOTER_HEIGHT = 80
BG_COLOR = (20, 20, 20)      # 底部背景
LINE_COLOR = (80, 80, 80)    # 轨道颜色
DONE_COLOR = (0, 140, 255)   # 篮球/进度颜色 (橙色)
TEXT_COLOR = (200, 200, 200) 
# ====================================================

def get_video_meta(path):
    """获取视频时长和宽度"""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened(): return 0, 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()
    duration = frames / fps if fps > 0 else 0
    return duration, width

def create_timeline_footer(width, clips_meta, save_path):
    """
    生成一张符合真实时间比例的静态图片
    clips_meta: List of (filename, duration, start_time)
    """
    footer = np.zeros((FOOTER_HEIGHT, width, 3), dtype=np.uint8)
    footer[:] = BG_COLOR
    
    total_duration = clips_meta[-1][2] + clips_meta[-1][1] # 最后一段的开始+时长
    if total_duration == 0: total_duration = 1
    
    # 左右边距 (防止球画在屏幕边缘被切掉)
    margin = 40
    timeline_width = width - (margin * 2)
    
    # 1. 画轨道 (全灰)
    cv2.line(footer, (margin, int(FOOTER_HEIGHT/2)), (width-margin, int(FOOTER_HEIGHT/2)), LINE_COLOR, 4)
    
    # 2. 按时间比例画篮球节点
    for i, (filename, duration, start_time) in enumerate(clips_meta):
        # 核心算法：当前开始时间 / 总时间 = 横坐标百分比
        ratio = start_time / total_duration
        cx = int(margin + (ratio * timeline_width))
        cy = int(FOOTER_HEIGHT / 2)
        
        # 画连接线 (橙色，表示这段是这个球的)
        # 计算下一段的起点作为终点
        end_ratio = (start_time + duration) / total_duration
        end_x = int(margin + (end_ratio * timeline_width))
        # 稍微留一点缝隙
        cv2.line(footer, (cx, cy), (end_x - 2, cy), DONE_COLOR, 4)

        # 画节点球
        cv2.circle(footer, (cx, cy), 12, (255, 255, 255), 2) # 白边
        cv2.circle(footer, (cx, cy), 10, DONE_COLOR, -1)     # 橙心
        
        # 画序号
        text = str(i + 1)
        cv2.putText(footer, text, (cx-5, cy+30), cv2.FONT_HERSHEY_SIMPLEX, 0.4, TEXT_COLOR, 1, cv2.LINE_AA)

    cv2.imwrite(save_path, footer)

def merge_timeline_accurate():
    if not os.path.exists(INPUT_FOLDER): return

    files = [f for f in os.listdir(INPUT_FOLDER) if f.endswith(".mp4") and not f.startswith(".")]
    files.sort()
    count = len(files)
    if count == 0: return

    print(f"📂 扫描到 {count} 个片段，正在计算时间轴...")

    # 1. 第一轮扫描：计算每个视频的精确时长和起止点
    clips_meta = [] # [(filename, duration, start_time), ...]
    current_head = 0.0
    video_w = 0
    
    # 随便读一个获取宽度 (假设所有视频宽度一致)
    _, video_w = get_video_meta(os.path.join(INPUT_FOLDER, files[0]))

    # 生成合并列表 list.txt
    list_txt = "temp_list_accurate.txt"
    with open(list_txt, "w") as f:
        for filename in files:
            path = os.path.join(INPUT_FOLDER, filename)
            duration, _ = get_video_meta(path)
            
            clips_meta.append((filename, duration, current_head))
            current_head += duration
            
            f.write(f"file '{os.path.abspath(path)}'\n")
            
    total_time = current_head
    print(f"⏱️ 总时长: {total_time:.2f} 秒")

    # 2. 生成“时间准确”的底部图片
    footer_path = "temp_footer_accurate.png"
    create_timeline_footer(video_w, clips_meta, footer_path)
    print(f"🎨 底部时间轴图片已生成")

    # 3. FFmpeg 硬件加速合成
    # 原理：[视频流]变高 -> [图片]贴到底部 -> 硬件编码
    print("🚀 启动硬件渲染 (这需要一点时间，因为在重写画面)...")
    
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-f", "concat", "-safe", "0", "-i", list_txt, # 输入0
        "-loop", "1", "-i", footer_path,              # 输入1 (无限循环图片)
        "-filter_complex",
        f"[0:v]pad=iw:ih+{FOOTER_HEIGHT}:0:0:{'#141414'}[bg];" # 扩展高度
        f"[bg][1:v]overlay=0:main_h-overlay_h:shortest=1",     # 贴图
        "-c:v", "h264_videotoolbox", # M3 硬件加速
        "-b:v", "6000k",             # 码率 6M (保证清晰)
        "-c:a", "aac",               # 音频
        "-preset", "ultrafast",      # 牺牲一点压缩率换取最快速度
        "-loglevel", "error",
        OUTPUT_FILE
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ 合并完成！\n💾 文件: {os.path.abspath(OUTPUT_FILE)}")
        
        # 清理
        os.remove(footer_path)
        os.remove(list_txt)
        
        subprocess.run(["open", OUTPUT_FILE])
        
    except Exception as e:
        print(f"❌ 失败: {e}")

if __name__ == "__main__":
    merge_timeline_accurate()