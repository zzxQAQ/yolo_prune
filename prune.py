import torch
import torch.nn.utils.prune as prune
from ultralytics import YOLO
from copy import deepcopy
import csv
import os

# 配置项
model_path = 'yolov8s-pose.pt'  # 替换为你的模型路径
data_yaml = '/home/zhengxiuzhang/ultralytics-main/coco8-pose.yaml'  # 替换为你的数据配置文件
sparsity_level = 0.1  # 剪枝比例
csv_file = 'prune_sensitivity.csv'
save_model_dir = 'pruned_models'
os.makedirs(save_model_dir, exist_ok=True)

# 模块组
module_groups = [
    ('Backbone_CBS1', ['model.0']),
    ('Backbone_CBS2', ['model.1']),
    ('Backbone_C2f1', ['model.2']),
    ('Backbone_CBS3', ['model.3']),
    ('Backbone_C2f2', ['model.4']),
    ('Backbone_CBS4', ['model.5']),
    ('Backbone_C2f3', ['model.6']),
    ('Backbone_CBS5', ['model.7']),
    ('Backbone_SPPF', ['model.8']),
    ('Neck_FPN1', ['model.9']),
    ('Neck_PAN1', ['model.10']),
    ('Neck_Upsample1', ['model.11']),
    ('Neck_Concat1', ['model.12']),
    ('Neck_FPN2', ['model.13']),
    ('Neck_PAN2', ['model.14']),
    ('Neck_Upsample2', ['model.15']),
    ('Neck_Concat2', ['model.16']),
    ('Head_Detect', ['model.17']),
    ('Head_Pose', ['model.18']),
    ('Head_key_point1', ['model.19']),
    ('Head_key_point2', ['model.20']),
    ('Head_key_point3', ['model.21'])
]

# 稀疏率计算函数
def compute_sparsity(layer):
    if not hasattr(layer, 'weight'):
        return 0.0
    total = layer.weight.numel()
    zeros = (layer.weight == 0).sum().item()
    return zeros / total

# 写入 CSV 头
with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Group Name', 'Sparsity', 'Actual Sparsity (%)', 'mAP@50', 'mAP@50-95', 'Precision', 'Recall'])

    base_yolo = YOLO(model_path)

    for group_name, group_layers in module_groups:
        print(f'\n🔧 正在剪枝模块组: {group_name}')

        # 深拷贝 YOLO 模型
        temp_yolo = deepcopy(base_yolo)
        temp_model = temp_yolo.model
        total_sparsity = []

        # 对指定模块剪枝
        for layer_name in group_layers:
            try:
                layer = temp_model.get_submodule(layer_name)
            except AttributeError:
                print(f"❌ 找不到层: {layer_name}")
                continue

            if isinstance(layer, torch.nn.Conv2d):
                prune.l1_unstructured(layer, name='weight', amount=sparsity_level)
                with torch.no_grad():
                    layer.weight_orig[layer.weight_mask == 0] = 0
                prune.remove(layer, 'weight')  # ✅ 真正移除掩码
                sparsity = compute_sparsity(layer)
                total_sparsity.append(sparsity)
                print(f"✅ {layer_name} 剪枝完成，实际稀疏率: {sparsity:.2%}")
            else:
                print(f"⚠️ 跳过非Conv2d层: {layer_name}")

        # 验证剪枝后的模型
        metrics = temp_yolo.val(data=data_yaml, split='val')

        # 提取指标
        map50 = metrics.pose.map50
        map5095 = metrics.pose.map
        precision = metrics.pose.p
        recall = metrics.pose.r
        avg_sparsity = sum(total_sparsity) / len(total_sparsity) * 100 if total_sparsity else 0

        print(f'📊 {group_name} 结果: mAP@50 = {map50:.4f}, mAP@50-95 = {map5095:.4f}, Precision = {precision:.4f}, Recall = {recall:.4f}')

        # 写入结果
        writer.writerow([
            group_name, sparsity_level, f"{avg_sparsity:.2f}%",
            float(map50), float(map5095), float(precision), float(recall)
        ])

        # 保存模型
        save_path = os.path.join(save_model_dir, f'{group_name}_pruned.pt')
        temp_yolo.save(save_path)
        print(f'💾 剪枝模型已保存至: {save_path}')

print(f'\n✅ 所有剪枝结果已保存至 CSV: {os.path.abspath(csv_file)}')
