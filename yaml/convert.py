import json
import os

import json
import os

def coco_pose_to_yolo(json_file, images_dir, labels_dir):
    with open(json_file, 'r') as f:
        coco = json.load(f)

    os.makedirs(labels_dir, exist_ok=True)

    image_info = {img['id']: img for img in coco['images']}
    category_ids = {cat['id']: i for i, cat in enumerate(coco['categories'])}

    for ann in coco['annotations']:
        if ann.get('iscrowd', 0) == 1:
            continue
        if 'keypoints' not in ann or not ann['keypoints']:
            continue

        image_id = ann['image_id']
        image = image_info[image_id]
        width, height = image['width'], image['height']
        file_name = image['file_name']
        label_path = os.path.join(labels_dir, os.path.splitext(file_name)[0] + '.txt')

        cls_id = 0  # 假设只训练 person

        # bbox: x, y, w, h -> 转为 YOLO 格式 [xc, yc, w, h]
        x, y, w, h = ann['bbox']
        xc = (x + w / 2) / width
        yc = (y + h / 2) / height
        w /= width
        h /= height

        # 防止溢出
        xc = min(max(xc, 0), 1)
        yc = min(max(yc, 0), 1)
        w = min(max(w, 0), 1)
        h = min(max(h, 0), 1)

        # 关键点归一化
        keypoints = ann['keypoints']
        keypoint_list = []
        for i in range(0, len(keypoints), 3):
            xk = keypoints[i] / width
            yk = keypoints[i+1] / height
            v = keypoints[i+2]
            xk = min(max(xk, 0), 1)
            yk = min(max(yk, 0), 1)
            keypoint_list.extend([f"{xk:.6f}", f"{yk:.6f}", str(v)])

        line = [str(cls_id), f"{xc:.6f}", f"{yc:.6f}", f"{w:.6f}", f"{h:.6f}"] + keypoint_list

        with open(label_path, 'a') as f:
            f.write(' '.join(line) + '\n')

    print(f"✅ 标签已生成到 {labels_dir}")


# 示例用法
json_file = '/home/zhengxiuzhang/ultralytics-main/coco-person/annotations/person_keypoints_val2017.json'
images_dir = '/home/zhengxiuzhang/ultralytics-main/coco-person/images/val2017'
labels_dir = '/home/zhengxiuzhang/ultralytics-main/coco-person/labels/val2017'

coco_pose_to_yolo(json_file, images_dir, labels_dir)
