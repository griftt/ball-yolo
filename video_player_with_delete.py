#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频播放器 - 支持快进快退、删除视频
功能：
1. 读取文件夹中所有视频进行播放
2. 快进快退：上一个/下一个视频
3. 删除按钮：删除当前视频并自动播放下一个
"""

import cv2
import os
import glob
import shutil

# ==================== 配置区域 ====================
VIDEO_FOLDER = "/Users/grifftwu/Documents/HighlightClips/111"  # 视频文件夹路径
VIDEO_EXTENSIONS = [
    "*.mp4",
    "*.mov",
    "*.avi",
    "*.MP4",
    "*.MOV",
    "*.mkv",
]  # 支持的视频格式
# ================================================

# 全局变量
cap = None
total_frames = 0
is_trackbar_active = False
video_list = []
current_video_index = 0
playback_speed = 1.5  # 播放速度倍率
deleted_count = 0  # 删除的视频计数


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
    cv2.putText(
        display_img,
        video_info,
        (20, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 0),
        2,
    )
    y_offset += line_height

    # 时间和帧数
    cv2.putText(
        display_img,
        f"Time: {time_str} | Frame: {frame_pos}/{total_frames}",
        (20, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (0, 255, 0),
        2,
    )
    y_offset += line_height

    # 播放速度
    speed_color = (0, 255, 255) if playback_speed == 1 else (0, 165, 255)
    cv2.putText(
        display_img,
        f"Speed: {playback_speed}x",
        (20, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        speed_color,
        2,
    )
    y_offset += line_height

    # 删除计数
    cv2.putText(
        display_img,
        f"Deleted: {deleted_count}",
        (20, y_offset),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
    )

    # 底部帮助信息
    help_text = "Space:Pause | </>:Video | X:Delete | A/D:Frame | +/-:Speed | Q:Quit"
    h, w = display_img.shape[:2]
    cv2.putText(
        display_img,
        help_text,
        (20, h - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        2,
    )

    # 缩放显示 (使用 INTER_LINEAR 更快)
    scale = 1280 / w if w > 1280 else 1
    if scale != 1:
        new_w = 1280
        new_h = int(h * scale)
        display_img = cv2.resize(
            display_img, (new_w, new_h), interpolation=cv2.INTER_LINEAR
        )

    cv2.imshow("Video Player with Delete", display_img)


def load_video(video_index):
    """加载指定索引的视频"""
    global cap, total_frames, current_video_index

    if video_index < 0 or video_index >= len(video_list):
        print(f"❌ 无效的视频索引: {video_index}")
        return False

    if cap is not None:
        cap.release()

    current_video_index = video_index
    video_path = video_list[current_video_index]

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)  # 增加缓冲区减少卡顿
    if not cap.isOpened():
        print(f"❌ 无法打开视频: {video_path}")
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # 更新进度条最大值
    window_name = "Video Player with Delete"
    cv2.setTrackbarMax("Seek", window_name, total_frames)
    cv2.setTrackbarPos("Seek", window_name, 0)

    print(
        f"📹 加载视频 [{current_video_index + 1}/{len(video_list)}]: {os.path.basename(video_path)}"
    )
    print(f"   总帧数: {total_frames}, FPS: {cap.get(cv2.CAP_PROP_FPS):.2f}")

    return True


def switch_video(direction):
    """切换视频 (direction: 1=下一个, -1=上一个)"""
    global current_video_index

    if len(video_list) == 0:
        print("❌ 视频列表为空")
        return

    new_index = current_video_index + direction

    # 循环切换
    if new_index < 0:
        new_index = len(video_list) - 1
    elif new_index >= len(video_list):
        new_index = 0

    load_video(new_index)


def delete_current_video():
    """删除当前视频并加载下一个"""
    global video_list, current_video_index, deleted_count, cap

    if len(video_list) == 0:
        print("❌ 视频列表为空，无法删除")
        return False

    current_video_path = video_list[current_video_index]
    video_name = os.path.basename(current_video_path)

    # 确认删除
    print(f"🗑️  确认删除视频: {video_name}?")
    print(f"   按 'X' 确认删除，或按其他键取消")

    # 释放当前视频
    if cap is not None:
        cap.release()
        cap = None

    # 删除文件到废纸篓（Mac）
    try:
        # 使用 os.remove 直接删除，或者用 shutil.move 到废纸篓
        # 这里直接删除
        if os.path.exists(current_video_path):
            os.remove(current_video_path)
            print(f"✅ 已删除: {video_name}")
            deleted_count += 1
        else:
            print(f"⚠️  文件不存在: {current_video_path}")
    except Exception as e:
        print(f"❌ 删除失败: {e}")
        return False

    # 从列表中移除
    video_list.pop(current_video_index)

    # 调整索引
    if current_video_index >= len(video_list):
        current_video_index = 0

    # 加载下一个视频
    if len(video_list) > 0:
        load_video(current_video_index)
    else:
        print("📭 所有视频已删除完毕！")
        return False

    return True


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
    global cap, total_frames, deleted_count, is_trackbar_active, video_list

    # 加载视频列表
    video_list = load_video_list(VIDEO_FOLDER)

    if not video_list:
        print(f"❌ 在文件夹 {VIDEO_FOLDER} 中没有找到视频文件")
        print(f"支持的格式: {', '.join(VIDEO_EXTENSIONS)}")
        return

    print(f"✅ 找到 {len(video_list)} 个视频文件")
    for i, video in enumerate(video_list):
        print(f"   {i + 1}. {os.path.basename(video)}")

    # 创建窗口
    window_name = "Video Player with Delete"
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
    print("  A / D   : 后退/前进一帧")
    print("  < / >   : 上一个/下一个视频")
    print("  X       : 删除当前视频（自动播放下一个）")
    print("  + / -   : 加速/减速播放")
    print("  Q       : 退出程序")
    print("  鼠标拖动: 快速定位\n")

    while True:
        # 计算等待时间（根据视频实际帧率和播放速度）
        fps = cap.get(cv2.CAP_PROP_FPS)
        wait_time = max(1, int(1000 / (fps * playback_speed))) if not is_paused else 0

        # 如果未暂停，自动读取下一帧
        if not is_paused:
            ret, frame = cap.read()
            if not ret:
                # 当前视频播放完毕，自动切换到下一个
                print(f"✅ 视频 [{current_video_index + 1}] 播放完毕")
                if len(video_list) > 1:
                    switch_video(1)
                    continue
                else:
                    print("📭 只有一个视频，循环播放")
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

        if key == ord("q"):
            break
        elif key == ord(" "):
            is_paused = not is_paused
            status = "⏸️  暂停" if is_paused else "▶️  播放"
            print(status)
        elif key == ord("a"):
            # 后退一帧
            is_paused = True
            target = max(0, current_pos - 2)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target)
        elif key == ord("d"):
            # 前进一帧
            is_paused = True
        elif key == ord(",") or key == ord("<"):
            # 上一个视频
            switch_video(-1)
            is_paused = True
        elif key == ord(".") or key == ord(">"):
            # 下一个视频
            switch_video(1)
            is_paused = True
        elif key == ord("x") or key == ord("X"):
            # 删除当前视频
            if delete_current_video():
                # 删除成功后继续播放
                continue
            else:
                # 删除失败或没有更多视频
                if len(video_list) == 0:
                    break
        elif key == ord("+") or key == ord("="):
            # 加速
            adjust_speed(1)
        elif key == ord("-") or key == ord("_"):
            # 减速
            adjust_speed(-1)

    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print(f"\n✅ 程序结束，共删除 {deleted_count} 个视频")


if __name__ == "__main__":
    run_tool()
