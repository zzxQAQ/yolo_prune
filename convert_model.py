from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.checks import check_yaml
from ultralytics import YOLO
import torch
import os
import csv

def convert(path):
    cfg_path = check_yaml('yolov8s.yaml')  # 配置文件路径
    model = DetectionModel(cfg=cfg_path, ch=3, nc=1)

    # 加载剪枝后的权重
    state_dict = torch.load(path, map_location='cpu')
    model.load_state_dict(state_dict)

    # 保存权重（仅state_dict）
    torch.save(model.state_dict(), 'yolov8s_pruned_weights.pth')

    # 用 YOLO 加载模型结构并加载权重
    yolo_model = YOLO(cfg_path)
    yolo_model.model.load_state_dict(torch.load('yolov8s_pruned_weights.pth', map_location='cpu'))

    # 保存为 .pt 文件
    yolo_model.save('yolov8s_pruned.pt')

def save_metrics(csv_path, target_layer, metrics):
    # 从 metrics 提取需要的数值
    metrics_dict = {
        'metrics/mAP50-95': metrics.box.map,
        'metrics/mAP50': metrics.box.map50,
    }

    selected_keys = ['metrics/mAP50-95', 'metrics/mAP50']
    selected_values = [metrics_dict[k] for k in selected_keys]

    # 如果文件不存在就写 header
    file_exists = os.path.exists(csv_path)
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['layer'] + selected_keys)
        writer.writerow([target_layer] + selected_values)
convert('/home/zhengxiuzhang/ultralytics-main/pruned_model/7-16-schedule+weight_magnitude_all_layer_yolov8s_pruned_state.pt')