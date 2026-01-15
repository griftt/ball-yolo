#!/usr/bin/env python3
"""
YOLO 模型转换脚本
将 best.pt 转换为跨平台格式（ONNX / TFLite）

使用方法:
1. 安装依赖: pip install ultralytics onnx onnxruntime
2. 将 best.pt 放在同目录下
3. 运行: python convert_model.py
"""

import os
import sys

def check_dependencies():
    """检查必要的依赖"""
    try:
        from ultralytics import YOLO
        import onnx
        print("✅ 依赖检查通过")
        return True
    except ImportError as e:
        print(f"❌ 缺少依赖: {e}")
        print("\n请先安装依赖:")
        print("  pip install ultralytics onnx onnxruntime")
        return False

def convert_to_onnx(model_path: str, output_dir: str = "./converted_models"):
    """
    转换为 ONNX 格式（iOS + Android 通用）
    """
    from ultralytics import YOLO
    
    print(f"\n📦 加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 导出 ONNX
    print("\n🔄 转换为 ONNX 格式...")
    onnx_path = model.export(
        format='onnx',
        imgsz=640,           # 输入尺寸
        simplify=True,       # 简化模型
        opset=12,            # ONNX opset 版本
        dynamic=False,       # 固定输入尺寸（移动端更稳定）
    )
    
    print(f"✅ ONNX 模型已保存: {onnx_path}")
    return onnx_path

def convert_to_tflite(model_path: str, output_dir: str = "./converted_models"):
    """
    转换为 TensorFlow Lite 格式（备选方案）
    """
    from ultralytics import YOLO
    
    print(f"\n📦 加载模型: {model_path}")
    model = YOLO(model_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n🔄 转换为 TFLite 格式...")
    tflite_path = model.export(
        format='tflite',
        imgsz=640,
        int8=False,          # 不使用 INT8 量化（保持精度）
    )
    
    print(f"✅ TFLite 模型已保存: {tflite_path}")
    return tflite_path

def verify_onnx_model(onnx_path: str):
    """验证 ONNX 模型"""
    import onnx
    import onnxruntime as ort
    import numpy as np
    
    print(f"\n🔍 验证 ONNX 模型: {onnx_path}")
    
    # 1. 检查模型结构
    model = onnx.load(onnx_path)
    onnx.checker.check_model(model)
    print("  ✅ 模型结构验证通过")
    
    # 2. 获取输入输出信息
    print("\n📊 模型信息:")
    for input in model.graph.input:
        shape = [d.dim_value for d in input.type.tensor_type.shape.dim]
        print(f"  输入: {input.name}, 形状: {shape}")
    
    for output in model.graph.output:
        shape = [d.dim_value for d in output.type.tensor_type.shape.dim]
        print(f"  输出: {output.name}, 形状: {shape}")
    
    # 3. 测试推理
    print("\n🧪 测试推理...")
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    input_shape = session.get_inputs()[0].shape
    
    # 创建测试输入
    test_input = np.random.randn(*[1 if isinstance(d, str) else d for d in input_shape]).astype(np.float32)
    
    # 运行推理
    outputs = session.run(None, {input_name: test_input})
    print(f"  ✅ 推理测试通过，输出形状: {outputs[0].shape}")
    
    return True

def print_usage_guide(onnx_path: str):
    """打印使用指南"""
    print("\n" + "="*60)
    print("🎉 模型转换完成！")
    print("="*60)
    
    print(f"""
📁 生成的文件:
   {onnx_path}

📱 Flutter 项目使用方法:

1. 复制模型到 Flutter 项目:
   cp {onnx_path} your_flutter_project/assets/models/

2. 在 pubspec.yaml 中添加:
   flutter:
     assets:
       - assets/models/best.onnx

3. 添加依赖:
   dependencies:
     onnxruntime: ^1.16.0

4. 加载并使用模型:
   final session = await OrtSession.create('assets/models/best.onnx');

📖 详细 Flutter 集成代码请参考项目文档。
""")

def main():
    # 检查依赖
    if not check_dependencies():
        sys.exit(1)
    
    # 模型路径（默认当前目录下的 best.pt）
    model_path = "./runs/train/yolo11n_640_train/weights/best.pt"
    
    # 支持命令行参数指定路径
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"❌ 找不到模型文件: {model_path}")
        print("\n请确保 best.pt 在当前目录，或指定路径:")
        print("  python convert_model.py /path/to/best.pt")
        sys.exit(1)
    
    print("="*60)
    print("🏀 篮球检测模型转换工具")
    print("="*60)
    print(f"源模型: {model_path}")
    
    # 转换为 ONNX（主要格式）
    onnx_path = convert_to_onnx(model_path)
    
    # 验证模型
    verify_onnx_model(onnx_path)
    
    # 打印使用指南
    print_usage_guide(onnx_path)
    
    # 可选：也转换 TFLite
    print("\n" + "-"*60)
    convert_tflite = input("是否也转换为 TFLite 格式？(y/n): ").strip().lower()
    if convert_tflite == 'y':
        convert_to_tflite(model_path)
        print("✅ TFLite 模型也已生成")

if __name__ == "__main__":
    main()
