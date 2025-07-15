from ultralytics import YOLO
import pdb
print('begin training')
import torch
model = YOLO('yolov8s.pt')

if __name__ == '__main__':
    # 训练模型
    # results = model.train(data='/home/zhengxiuzhang/ultralytics-main/coco8-pose.yaml', epochs=200, imgsz=640,device='4,5,6,7')

    model.train(data='/home/zhengxiuzhang/ultralytics-main/coco.yaml', imgsz=640,device='4,5',lr0=0.0001,lrf=1e-5,epochs=100,batch=32,prune_method='wanda')
    # print('begin training')
    # model = YOLO('yolov8s.pt')
    # model.train(data='/home/zhengxiuzhang/ultralytics-main/coco.yaml', imgsz=640,device='4,5,6,7',lr0=0.0001,lrf=0.2,epochs=100,batch=96,prune_method='abs')
    # model = YOLO('yolov8s.pt')  # 从YAML构建并传输权重
    # model.train(data='/home/zhengxiuzhang/ultralytics-main/coco.yaml', imgsz=640,device='0,1,2,3',lr0=0.002,lrf=0.1,epochs=20,batch=32)
    # model.val(data='/home/zhengxiuzhang/ultralytics-main/coco.yaml', imgsz=640,device='4')
    
