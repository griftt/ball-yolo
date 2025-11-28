import cv2
import os

# ==================== 配置区域 ====================
VIDEO_PATH = "/Users/grifftwu/Desktop/历史篮球/1112/1112.mov" 
OUTPUT_DIR = "/Users/grifftwu/Desktop/历史篮球/1112/manual_dataset"
# ================================================

# 全局变量，用于回调函数
cap = None
total_frames = 0
is_trackbar_active = False # 防止程序自动更新进度条时触发回调

def on_trackbar_change(pos):
    """进度条回调函数：当用户拖动滑块时触发"""
    global is_trackbar_active
    if is_trackbar_active: return # 如果是程序自己在更新，忽略
    
    # 用户手动拖动了，跳转视频位置
    cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
    # 立即读取一帧显示，防止画面卡顿
    ret, frame = cap.read()
    if ret:
        show_frame(frame, pos)

def show_frame(frame, frame_pos):
    """统一的画面显示函数"""
    # 1. 拷贝一份用于显示（不污染原图）
    display_img = frame.copy()
    
    # 2. 绘制 UI 信息
    # 计算时间戳
    fps = cap.get(cv2.CAP_PROP_FPS)
    seconds = frame_pos / fps
    m, s = divmod(seconds, 60)
    time_str = f"{int(m):02d}:{s:05.2f}"
    
    # 绘制文字
    cv2.putText(display_img, f"Time: {time_str} | Frame: {frame_pos}/{total_frames}", (30, 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(display_img, f"Saved: {save_count}", (30, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    
    # 3. 缩放显示 (适应屏幕，比如缩放到宽 1280)
    h, w = display_img.shape[:2]
    scale = 1280 / w
    new_h = int(h * scale)
    display_img = cv2.resize(display_img, (1280, new_h))
    
    cv2.imshow('Pro Label Tool', display_img)

def run_tool():
    global cap, total_frames, save_count, is_trackbar_active
    
    if not os.path.exists(VIDEO_PATH):
        print("❌ 找不到视频文件")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    save_count = len(os.listdir(OUTPUT_DIR))
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 创建窗口
    window_name = 'Pro Label Tool'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 800)
    
    # ✅ 创建进度条
    # 参数: 进度条名, 窗口名, 默认值, 最大值, 回调函数
    cv2.createTrackbar("Seek", window_name, 0, total_frames, on_trackbar_change)
    
    is_paused = False
    
    print("🚀 启动成功！")
    print("🖱️  用鼠标拖动底部滑块可快速定位")
    print("⌨️  [空格]: 暂停/播放 | [S]: 保存 | [A/D]: 微调")

    while True:
        # 如果未暂停，自动读取下一帧
        if not is_paused:
            ret, frame = cap.read()
            if not ret: # 循环播放
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue
        else:
            # 暂停时，我们需要不断刷新界面以响应进度条拖动，但不需要读新帧
            # 这里为了简单，我们重新读取当前帧（性能损耗可忽略）
            current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_pos) # 修正位置
            ret, frame = cap.read()
            if not ret: break

        current_pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES))

        # ✅ 更新进度条位置 (这是程序更新，不是用户拖动)
        is_trackbar_active = True
        cv2.setTrackbarPos("Seek", window_name, current_pos)
        is_trackbar_active = False
        
        # 显示
        show_frame(frame, current_pos)
        
        # 按键处理
        key = cv2.waitKey(30 if not is_paused else 0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            is_paused = not is_paused
        elif key == ord('s'):
            # 保存原图
            save_count += 1
            filename = f"train_hd_{save_count:05d}.jpg"
            path = os.path.join(OUTPUT_DIR, filename)
            cv2.imwrite(path, frame)
            print(f"📸 已保存: {filename}")
            
            # 视觉反馈
            cv2.rectangle(frame, (0,0), (frame.shape[1], frame.shape[0]), (255,255,255), 10)
            show_frame(frame, current_pos)
            cv2.waitKey(50)
            
        elif key == ord('a'): # 后退
            is_paused = True
            target = max(0, current_pos - 2) # openCV读取后会自动+1，所以回退要-2
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            
        elif key == ord('d'): # 前进
            is_paused = True
            # 读取下一帧自然会前进，无需 set

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_tool()