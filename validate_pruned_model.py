#!/usr/bin/env python3
"""
验证剪枝后的模型
"""
import torch
from ultralytics import YOLO
from ultralytics.wanda_pp.prune import apply_mask_to_model_weights, compute_sparsity
import os

def validate_pruned_model(model_path, data_yaml='coco.yaml'):
    """
    验证剪枝后的模型
    
    参数:
        model_path: 剪枝模型路径
        data_yaml: 数据配置文件
    """
    print(f"加载剪枝模型: {model_path}")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        print(f"错误: 模型文件不存在 {model_path}")
        return
    
    # 加载保存的数据
    checkpoint = torch.load(model_path, map_location='cpu')
    
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        # 新格式：包含model, masks, sparsity
        model_state = checkpoint['model']
        masks = checkpoint.get('masks', None)
        sparsity = checkpoint.get('sparsity', 0.0)
        
        print(f"模型稀疏率: {sparsity:.3f} ({sparsity*100:.1f}%)")
        if masks:
            print(f"掩码包含 {len(masks)} 层")
    else:
        # 旧格式：只有state_dict
        model_state = checkpoint
        masks = None
        print("警告: 旧格式模型，无掩码信息")
    
    # 创建基础模型
    base_model = YOLO('yolov8s.pt')  # 使用对应的基础模型
    
    # 加载权重
    base_model.model.load_state_dict(model_state)
    
    # 如果有掩码，重新应用（确保一致性）
    if masks:
        print("重新应用掩码确保一致性...")
        apply_mask_to_model_weights(base_model.model, masks)
        
        # 验证稀疏率
        actual_sparsity = compute_sparsity(masks)
        print(f"实际稀疏率: {actual_sparsity:.3f} ({actual_sparsity*100:.1f}%)")
    
    # 设置为评估模式
    base_model.model.eval()
    
    # 进行验证
    print("开始验证...")
    results = base_model.val(data=data_yaml, verbose=True)
    
    # 打印结果
    print(f"\n验证结果:")
    print(f"mAP50: {results.box.map50:.4f}")
    print(f"mAP50-95: {results.box.map:.4f}")
    
    return results

if __name__ == "__main__":
    # 验证剪枝后的模型
    model_path = "/home/zhengxiuzhang/ultralytics-main/pruned_model/1e-3-0.75_schedule+wanda++_all_layer_yolov8s_pruned_state.pt"
    results = validate_pruned_model(model_path) 