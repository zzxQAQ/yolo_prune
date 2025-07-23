import torch
import torch.nn as nn
from ultralytics import YOLO

def compute_model_sparsity(model, only_conv=True):
    total_params = 0
    zero_params = 0

    for module in model.modules():
        if only_conv:
            if isinstance(module, nn.Conv2d):
                if hasattr(module, 'weight') and module.weight is not None:
                    weight = module.weight.data
                    total_params += weight.numel()
                    zero_params += (weight == 0).sum().item()
        else:
            # 统计模块所有参数（包括权重和偏置）
            for param in module.parameters(recurse=False):
                total_params += param.numel()
                zero_params += (param == 0).sum().item()

    sparsity = zero_params / total_params if total_params > 0 else 0
    print(f"Total params: {total_params}, Zero params: {zero_params}, Sparsity: {sparsity:.4f}")
    return sparsity

# 加载模型
model = YOLO("/home/zhengxiuzhang/ultralytics-main/runs/detect/25-35-45-6-75-875/weights/last.pt")
print(f'{"layer name":35s}  keep  sparsity')
print('-'*55)
for name, m in model.named_modules():
    if isinstance(m, nn.Conv2d):
        w = m.weight.data
        total = w.numel()
        zeros = (w == 0).sum().item()
        sparsity = zeros / total
        print(f'{name:35s}  {1-sparsity:5.3f}  {sparsity:5.3f}')
# 统计剪枝后卷积层权重稀疏率
print("Conv2d weights sparsity:")
compute_model_sparsity(model.model, only_conv=True)

# 如果想统计所有层所有参数稀疏率
print("All parameters sparsity:")
compute_model_sparsity(model.model, only_conv=False)