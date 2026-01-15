import cv2
import os
import glob

# ==================== 配置区域 ====================
VIDEO_FOLDER = "/Users/grifftwu/ball"  # 视频文件夹路径
OUTPUT_DIR = "/Users/grifftwu/Desktop/历史篮球/multi/ball"
VIDEO_EXTENSIONS = ["*.mp4", "*.mov", "*.avi", "*.MP4", "*.MOV"]  # 支持的视频格式
# ================================================

# 全局变量
cap = None
total_frames = 0
is_trackbar_active = False
video_list = []
current_video_index = 0
playback_speed = 1  # 播放速度倍率
save_count = 0

def load_video_list(folder_path):
    """加载文件夹中的所有视频文件"""
    videos = []
    for ext in VIDEO_EXTENSIONS:
        videos.extend(glob.glob(os.path.join(folder_path, ext)))
    videos.sort()  # 按文件名排序
    return videos

def on_trackbar_change(pos):
    """进度条回调函数：当用户拖动滑块时触发"""
    global is_trackbar_active
    if is_trackbar_active: 
        return
    
    # 用户手动拖动了，跳转视频位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    ret, frame = cap.read()
    if ret:
        show_frame(frame, pos)

def show_frame(frame, frame_pos):
    """统一的画面显示函数"""
    display_img = frame.copy()
    
    # 计算时间戳
    fps = cap.get(cv2.CAP_PROP_FPS)
    seconds = frame_pos / fps
    m, s = divmod(seconds, 60)
    time_str = f"{int(m):02d}:{s:05.2f}"
    
    # 当前视频信息
    video_name = os.path.basename(video_list[current_video_index])
    video_info = f"[{current_video_index + 1}/{len(video_list)}] {video_name}"
    
    # 绘制文字信息
    y_offset = 30
    line_height = 35
    
    # 视频信息
    cv2.putText(display_img, video_info, (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
    y_offset += line_height
    
    # 时间和帧数
    cv2.putText(display_img, f"Time: {time_str} | Frame: {frame_pos}/{total_frames}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_offset += line_height
    
    # 播放速度
    speed_color = (0, 255, 255) if playback_speed == 1 else (0, 165, 255)
    cv2.putText(display_img, f"Speed: {playback_speed}x", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, speed_color, 2)
    y_offset += line_height
    
    # 保存计数
    cv2.putText(display_img, f"Saved: {save_count}", (20, y_offset), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    # 底部帮助信息
    help_text = "Space:Pause | S:Save | A/D:Frame | </>:Video | +/-:Speed | Q:Quit"
    h, w = display_img.shape[:2]
    cv2.putText(display_img, help_text, (20, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
    
    # 缩放显示
    scale = 1280 / w if w > 1280 else 1
    if scale != 1:
        new_w = 1280
        new_h = int(h * scale)
        display_img = cv2.resize(display_img, (new_w, new_h))
    
    cv2.imshow('Multi-Video Label Tool', display_img)

def load_video(video_index):
    """加载指定索引的视频"""
    global cap, total_frames, current_video_index
    
    if cap is not None:
        cap.release()
    
    current_video_index = video_index
    video_path = video_list[current_video_index]
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return False
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 更新进度条最大值
    window_name = 'Multi-Video Label Tool'
    cv2.setTrackbarMax("Seek", window_name, total_frames)
    cv2.setTrackbarPos("Seek", window_name, 0)
    
    print(f"📹 加载视频 [{current_video_index + 1}/{len(video_list)}]: {os.path.basename(video_path)}")
    print(f"   总帧数: {total_frames}, FPS: {cap.get(cv2.CAP_PROP_FPS):.2f}")
    
    return True

def switch_video(direction):
    """切换视频 (direction: 1=下一个, -1=上一个)"""
    global current_video_index
    
    new_index = current_video_index + direction
    
    # 循环切换
    if new_index < 0:
        new_index = len(video_list) - 1
    elif new_index >= len(video_list):
        new_index = 0
    
    load_video(new_index)

def adjust_speed(direction):
    """调整播放速度 (direction: 1=加速, -1=减速)"""
    global playback_speed
    
    speed_levels = [0.25, 0.5, 1, 2, 4, 8]
    
    try:
        current_idx = speed_levels.index(playback_speed)
        new_idx = current_idx + direction
        
        if 0 <= new_idx < len(speed_levels):
            playback_speed = speed_levels[new_idx]
            print(f"⚡ 播放速度: {playback_speed}x")
    except ValueError:
        playback_speed = 1

def run_tool():
    global cap, total_frames, save_count, is_trackbar_active, video_list
    
    # 加载视频列表
    video_list = load_video_list(VIDEO_FOLDER)
    
    if not video_list:
        print(f"❌ 在文件夹 {VIDEO_FOLDER} 中没有找到视频文件")
        print(f"支持的格式: {', '.join(VIDEO_EXTENSIONS)}")
        return
    
    print(f"✅ 找到 {len(video_list)} 个视频文件")
    for i, video in enumerate(video_list):
        print(f"   {i+1}. {os.path.basename(video)}")
    
    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_count = len([f for f in os.listdir(OUTPUT_DIR) if f.endswith('.jpg')])
    
    # 创建窗口
    window_name = 'Multi-Video Label Tool'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 800)
    
    # 加载第一个视频
    if not load_video(0):
        return
    
    # 创建进度条
    cv2.createTrackbar("Seek", window_name, 0, total_frames, on_trackbar_change)
    
    is_paused = False
    
    print("\n🚀 启动成功！")
    print("\n⌨️  快捷键说明:")
    print("  空格    : 暂停/播放")
    print("  S       : 保存当前帧")
    print("  A / D   : 后退/前进一帧")
    print("  < / >   : 上一个/下一个视频")
    print("  + / -   : 加速/减速播放")
    print("  Q       : 退出程序")
    print("  鼠标拖动: 快速定位\n")
    
    while True:
        # 计算等待时间（根据播放速度）
        wait_time = int(30 / playback_speed) if not is_paused else 0
        
        # 如果未暂停，自动读取下一帧
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                # 当前视频播放完毕，自动切换到下一个
                print(f"✅ 视频 [{current_video_index + 1}] 播放完毕")
                if current_video_index < len(video_list) - 1:
                    switch_video(1)
                    continue
                else:
                    print("🎉 所有视频播放完毕！")
                    # 循环播放
                    load_video(0)
                    continue
        else:
            # 暂停时重新读取当前帧
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos)
            ret, frame = cap.read()
            if not ret:
                break
        
        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
        
        # 更新进度条
        is_trackbar_active = True
        cv2.setTrackbarPos("Seek", window_name, current_pos)
        is_trackbar_active = False
        
        # 显示
        show_frame(frame, current_pos)
        
        # 按键处理
        key = cv2.waitKey(wait_time) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            is_paused = not is_paused
            status = "⏸️  暂停" if is_paused else "▶️  播放"
            print(status)
        elif key == ord('s'):
            # 保存原图
            save_count += 1
            video_name = os.path.splitext(os.path.basename(video_list[current_video_index]))[0]
            filename = f"{video_name}_frame_{save_count:05d}.jpg"
            path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(path, frame)
            print(f"📸 已保存: {filename}")
            
            # 视觉反馈
            cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (255, 255, 255), 15)
            show_frame(frame, current_pos)
            cv2.waitKey(100)
        elif key == ord('a'):
            # 后退一帧
            is_paused = True
            target = max(0, current_pos - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        elif key == ord('d'):
            # 前进一帧
            is_paused = True
        elif key == ord(',') or key == ord('<'):
            # 上一个视频
            switch_video(-1)
            is_paused = True
        elif key == ord('.') or key == ord('>'):
            # 下一个视频
            switch_video(1)
            is_paused = True
        elif key == ord('+') or key == ord('='):
            # 加速
            adjust_speed(1)
        elif key == ord('-') or key == ord('_'):
            # 减速
            adjust_speed(-1)
    
    cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ 程序结束，共保存 {save_count} 张图片到: {OUTPUT_DIR}")

if __name__ == "__main__":
    run_tool()
