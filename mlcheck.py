import coremltools as ct
# 加载模型
model = ct.models.MLModel("runs/detect/runs/train/yolo26s_640_train_hd_person/weights/best.mlpackage")
# 获取规格
spec = model.get_spec()
# 检查是否包含 NMS
def has_nms(spec):
    # 检查 Neural Network 类型的 NMS 层
    for layer in spec.neuralNetwork.layers:
        if layer.WhichOneof('layer') == 'nonMaximumSuppression':
            return True
    # 检查 ML Program 类型的 NMS
    if spec.HasField('pipeline'):
        for model_spec in spec.pipeline.models:
            if has_nms(model_spec):
                return True
    return False
print("Model has NMS:", has_nms(spec))