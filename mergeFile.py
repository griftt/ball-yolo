import os
import shutil
from pathlib import Path

def get_unique_path(destination_dir, filename):
    """
    如果目标文件已存在，则生成一个新的文件名。
    例如: data.txt -> data_copy_1.txt -> data_copy_2.txt
    """
    base_name, extension = os.path.splitext(filename)
    counter = 1
    new_filename = filename
    destination_path = os.path.join(destination_dir, new_filename)

    while os.path.exists(destination_path):
        new_filename = f"{base_name}_copy_{counter}{extension}"
        destination_path = os.path.join(destination_dir, new_filename)
        counter += 1
    
    return destination_path

def merge_folders(source_folders, target_folder):
    """
    将多个源文件夹的内容合并到一个目标文件夹中。
    """
    # 1. 确保目标文件夹存在
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"📁 已创建目标文件夹: {target_folder}")
    else:
        print(f"📁 目标文件夹已存在: {target_folder}")

    total_files_copied = 0
    total_files_renamed = 0

    print("-" * 50)

    # 2. 遍历每一个源文件夹
    for src_folder in source_folders:
        src_folder = os.path.normpath(src_folder) # 规范化路径
        if not os.path.exists(src_folder):
            print(f"⚠️ 跳过不存在的源文件夹: {src_folder}")
            continue

        print(f"🚀 正在处理源文件夹: {os.path.basename(src_folder)} ...")

        # 3. 遍历源文件夹下的所有文件和子文件夹 (os.walk)
        for root, dirs, files in os.walk(src_folder):
            # 计算当前路径相对于源文件夹根目录的相对路径
            # 例如: src/sub/a.txt -> relative_path 是 "sub"
            relative_path = os.path.relpath(root, src_folder)
            
            # 确定在目标文件夹中的对应目录
            dest_dir = os.path.join(target_folder, relative_path)
            
            # 如果目标子目录不存在，创建它
            if not os.path.exists(dest_dir):
                os.makedirs(dest_dir)

            # 4. 复制文件
            for file in files:
                src_file_path = os.path.join(root, file)
                
                # 获取唯一的目标路径（处理重名）
                final_dest_path = get_unique_path(dest_dir, file)
                
                try:
                    # 使用 copy2 保留文件元数据（如创建时间、修改时间）
                    shutil.copy2(src_file_path, final_dest_path)
                    
                    # 检查是否发生了重命名
                    if os.path.basename(final_dest_path) != file:
                        print(f"   ⚠️ 重名处理: {file} -> {os.path.basename(final_dest_path)}")
                        total_files_renamed += 1
                    else:
                        # print(f"   ✅ 复制: {file}") # 如果文件太多，可以注释掉这行
                        pass
                        
                    total_files_copied += 1
                except Exception as e:
                    print(f"   ❌ 复制失败 {file}: {e}")

    print("-" * 50)
    print(f"🎉 合并完成！")
    print(f"📂 目标位置: {os.path.abspath(target_folder)}")
    print(f"📄 共复制文件: {total_files_copied} 个")
    print(f"🏷️ 因重名自动改名: {total_files_renamed} 个")

# ==================== ⚙️ 配置区域 ====================
if __name__ == "__main__":
    
    # 1. 在这里填入你要合并的文件夹路径 (支持任意数量)
    source_list = [
        r"/Users/grifftwu/Desktop/历史篮球/1112/images/train",
        r"/Users/grifftwu/Desktop/历史篮球/1122/images/train",
    ]

    # 2. 在这里填入你想生成的新的文件夹路径
    output_folder = r"/Users/grifftwu/Desktop/历史篮球/1126/images/train"

    # 执行合并
    merge_folders(source_list, output_folder)