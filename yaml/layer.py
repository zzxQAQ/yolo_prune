import torch
import sys, os
sys.path.append(os.path.abspath('/home/zhengxiuzhang/ultralytics-main/ultralytics'))
from wanda_pp.prune import WandaPlusElementwiseScorer,prune_all_conv_layers_by_score,apply_mask_to_model_weights,prune_single_layer_by_score
from ultralytics import YOLO
from convert_model import convert,save_metrics

scores = torch.load('wanda_scores.pt')
for target_layer in scores.keys():
    print(f"\n===> 正在敏感性分析：剪枝层 {target_layer}")

    model = YOLO('yolov8s.pt')

    # 4. 应用 mask
    model.train(data='/home/zhengxiuzhang/ultralytics-main/coco.yaml', imgsz=640,device='0,1,2,3',lr0=0.0001,lrf=0.2,epochs=10,batch=32,target_layer=target_layer)
    
    save_dir = '/home/zhengxiuzhang/ultralytics-main/pruned_model'
    filename = f"layer_{target_layer}_yolov8s_pruned_state.pt"
    pruned_model_path = os.path.join(save_dir, filename)
    convert(pruned_model_path)
    model = YOLO('yolov8s_pruned.pt')
    metrics = model.val(data='/home/zhengxiuzhang/ultralytics-main/coco.yaml', imgsz=640,device='0')
    csv_path='/home/zhengxiuzhang/ultralytics-main/result.csv'
    save_metrics(csv_path, target_layer, metrics)

    # # 5. 验证模型精度，记录
    # metrics = model.val(data='/your/dataset.yaml', imgsz=640, batch=32, device='cuda')
    # print(f"Layer {target_layer} 剪枝后验证结果: {metrics['metrics/mAP50']:.4f}")