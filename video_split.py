import os
import subprocess
import json

def get_video_duration(video_path):
    """
    获取视频时长（秒）
    :param video_path: 视频路径
    :return: 时长（秒）
    """
    try:
        # 使用 ffprobe 获取视频时长，避免依赖 opencv-python
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as e:
        print(f"❌ 获取视频时长失败: {e}")
        # 如果 ffprobe 失败，尝试抛出异常提示
        raise ValueError(f"无法获取视频时长，请确保已安装 ffmpeg (包含 ffprobe): {video_path}")

def split_video_in_half(video_path, output_dir=None, accurate=False):
    """
    将视频平均拆分成两段
    
    :param video_path: 视频文件路径
    :param output_dir: 输出目录，如果不指定，默认保存在原视频同级目录下
    :param accurate: 是否精准拆分。
                     False (默认): 使用流拷贝模式，速度快，无损，但只能在关键帧处切割，时间可能不精准。
                     True: 重新编码，时间精准，但速度慢且可能会有画质损失。
    :return: 生成的两个视频文件路径列表
    """
    if not os.path.exists(video_path):
        print(f"❌ 错误: 文件不存在 -> {video_path}")
        return []

    # 1. 确定输出目录和文件名
    if output_dir is None:
        output_dir = os.path.dirname(video_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    file_name = os.path.basename(video_path)
    name, ext = os.path.splitext(file_name)
    
    output_part1 = os.path.join(output_dir, f"{name}_part1{ext}")
    output_part2 = os.path.join(output_dir, f"{name}_part2{ext}")
    
    print(f"正在分析视频: {video_path} ...")
    
    try:
        # 2. 获取视频总时长
        duration = get_video_duration(video_path)
        half_duration = duration / 2
        
        print(f"视频总时长: {duration:.2f} 秒")
        print(f"拆分点: {half_duration:.2f} 秒")
        
        # 3. 构建 ffmpeg 命令
        # 通用参数
        common_args = ['ffmpeg', '-y']
        
        # 编码参数
        # 如果是精准模式，不使用 copy，而是默认重新编码 (libx264) 或者让 ffmpeg 自动选择
        # 如果是快速模式，使用 copy
        codec_args = [] if accurate else ['-c', 'copy']
        
        # 第一段: 从开始到中间
        cmd1 = common_args + [
            '-i', video_path,
            '-t', str(half_duration)
        ] + codec_args + [output_part1]
        
        # 第二段: 从中间到结束
        # 注意：为了精准定位，-ss 放在 -i 之前（输入选项），这样会快速定位到关键帧，
        # 但如果是 copy 模式，-ss 放在 -i 之前可能会导致时间戳重置问题，
        # 这里为了兼容性和简单性，对于 copy 模式，我们把 -ss 放在 -i 之前通常也是推荐的做法，
        # 但有时为了精准剪切（非 copy 模式），ffmpeg 推荐 -ss 放在 -i 之前。
        
        if accurate:
            # 精准模式：-ss 在前，且重新编码
            cmd2 = common_args + [
                '-ss', str(half_duration),
                '-i', video_path
            ] + codec_args + [output_part2]
        else:
            # 快速模式：-ss 在前结合 copy
             cmd2 = common_args + [
                '-ss', str(half_duration),
                '-i', video_path
            ] + codec_args + ['-avoid_negative_ts', '1', output_part2]
        
        # 4. 执行命令
        print(f"正在生成第一段: {output_part1} ...")
        subprocess.run(cmd1, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print(f"正在生成第二段: {output_part2} ...")
        subprocess.run(cmd2, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        print("✅ 拆分完成！")
        return [output_part1, output_part2]
        
    except Exception as e:
        print(f"❌ 拆分失败: {str(e)}")
        return []

if __name__ == "__main__":
    # 示例用法
    # 请替换为实际的视频路径进行测试
    sample_video = "/Users/grifftwu/Desktop/历史篮球/20260205/0205.mov"
    if os.path.exists(sample_video):
        split_video_in_half(sample_video)
    else:
        print("请在代码中修改 'sample_video' 变量为有效的视频路径来运行测试。")
