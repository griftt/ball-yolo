# -*- coding: utf-8 -*-
import os
import subprocess

# ==================== ⚙️ 配置区域 ====================

# 1. 📂 输入文件夹列表 (支持多个，按顺序合并)
# 格式：[ "文件夹A", "文件夹B", "文件夹C" ]
INPUT_FOLDERS = [
    # "./outputs/auto_mps_clips_1126_rotated1",
    # "./outputs/auto_mps_clips_batch_01",  # 举例：第二个文件夹
     "/Users/grifftwu/Documents/HighlightClips/04", 
     "/Users/grifftwu/Documents/HighlightClips/03", 
     "/Users/grifftwu/Documents/HighlightClips/02", 
     "/Users/grifftwu/Documents/HighlightClips/01", 
    # "outputs" 
]

# 2. 💾 输出文件
OUTPUT_FILE = "./runs/output/0325.mp4"

# ====================================================

def instant_merge_multi():
    # 1. 准备合并列表文件
    list_txt_path = "temp_merge_list.txt"
    total_files_count = 0
    
    # 用于收集所有有效的视频路径
    valid_video_paths = []

    print(f"🚀 开始扫描 {len(INPUT_FOLDERS)} 个文件夹...")

    # 2. 遍历每个文件夹
    for folder in INPUT_FOLDERS:
        # 清理路径字符串可能多余的空格
        folder = folder.strip()
        
        if not os.path.exists(folder):
            print(f"⚠️ 跳过不存在的文件夹: {folder}")
            continue


        # 获取该文件夹下的 MP4
        files = [f for f in os.listdir(folder) if f.endswith(".mp4") and not f.startswith(".")]
        
        # 排序：保证该文件夹内的视频是按 goal_001, goal_002 顺序播放的
        files.sort()
        
        if not files:
            print(f"⚠️ 文件夹为空: {folder}")
            continue
            
        print(f"📂 [{folder}] -> 找到 {len(files)} 个视频")
        
        # 将完整路径加入列表
        for filename in files:
            abs_path = os.path.abspath(os.path.join(folder, filename))
            valid_video_paths.append(abs_path)

    # 3. 检查是否有文件
    total_files_count = len(valid_video_paths)
    if total_files_count == 0:
        print("❌ 没有找到任何 MP4 视频，终止合并。")
        return

    print(f"📊 总计待合并视频: {total_files_count} 个")

    # 4. 写入 FFmpeg 列表文件
    with open(list_txt_path, "w", encoding="utf-8") as f:
        for video_path in valid_video_paths:
            # 格式: file '/path/to/video.mp4'
            f.write(f"file '{video_path}'\n")

    # 5. 调用 FFmpeg 执行“流拷贝”
    # -safe 0 : 允许读取任意路径（这是读取多文件夹的关键）
    cmd = [
        "ffmpeg", "-y", "-nostdin",
        "-f", "concat",
        "-safe", "0",
        "-i", list_txt_path,
        "-c", "copy",  # 🔥 极速流拷贝
        "-loglevel", "error",
        OUTPUT_FILE
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✅ 合并完成！\n💾 文件保存为: {os.path.abspath(OUTPUT_FILE)}")
        
        # 清理临时列表
        if os.path.exists(list_txt_path):
            os.remove(list_txt_path)
        
        # 自动打开
        subprocess.run(["open", OUTPUT_FILE])
        
    except Exception as e:
        print(f"❌ 合并失败: {e}")

if __name__ == "__main__":
    instant_merge_multi()