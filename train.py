from ultralytics import YOLO
import pdb
print('begin training')
import torch
model = YOLO('yolov8s.pt')

if __name__ == '__main__':
    # 训练模型
    model.train(
        data='/home/zhengxiuzhang/ultralytics-main/coco.yaml', 
        imgsz=640,
        device='0,1,2,3',
        epochs=150,
        batch=96,
        prune_method='abs',
        patience=150,
        prune_fre=100,
        prune_final_sparsity=0.875,
        prune_lr0=5e-4,
        prune_lrf=5e-6)

    
