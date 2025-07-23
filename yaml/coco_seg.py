import json
import os

# === 设置路径 ===
coco_json_path = '/home/zhengxiuzhang/ultralytics-main/coco_seg/annotations/instances_train2017.json'
images_dir = '/home/zhengxiuzhang/ultralytics-main/coco_seg/images/train2017'
labels_dir = '/home/zhengxiuzhang/ultralytics-main/coco_seg/labels/train2017'

os.makedirs(labels_dir, exist_ok=True)

print(f"加载 COCO 标注文件: {coco_json_path}")
with open(coco_json_path, 'r') as f:
    coco = json.load(f)

# === 类别ID映射（原始ID → 0起始索引） ===
cat_id_to_idx = {cat['id']: i for i, cat in enumerate(coco['categories'])}

# === 图片信息字典（image_id → (width, height, file_name)） ===
img_info = {img['id']: (img['width'], img['height'], img['file_name']) for img in coco['images']}

# === 按图片分组标注 ===
anns_per_img = {}
for ann in coco['annotations']:
    img_id = ann['image_id']
    anns_per_img.setdefault(img_id, []).append(ann)

print(f"开始转换 {len(anns_per_img)} 张图片的标注...")

# === 转换每张图的标注 ===
for img_id, anns in anns_per_img.items():
    w, h, fname = img_info[img_id]
    label_lines = []

    for ann in anns:
        if 'segmentation' not in ann or not ann['segmentation']:
            continue
        if ann.get('iscrowd', 0) == 1:
            continue

        cls_id = cat_id_to_idx[ann['category_id']]
        seg = ann['segmentation']

        if isinstance(seg, list) and len(seg) > 0 and isinstance(seg[0], list):
            poly = seg[0]
        elif isinstance(seg, list):
            poly = seg
        else:
            continue

        # === 归一化多边形点坐标 ===
        poly_norm = []
        for i in range(0, len(poly), 2):
            x = min(max(poly[i] / w, 0), 1)
            y = min(max(poly[i + 1] / h, 0), 1)
            poly_norm.append(x)
            poly_norm.append(y)

        # === YOLOv8 Seg 标签行：cls_id x1 y1 x2 y2 ...
        line = f"{cls_id} " + " ".join(f"{p:.6f}" for p in poly_norm)
        label_lines.append(line)

    # === 保存标签文件（与图像同名） ===
    label_path = os.path.join(labels_dir, os.path.splitext(fname)[0] + '.txt')
    with open(label_path, 'w') as f:
        f.write("\n".join(label_lines))

print(f"✅ 转换完成！YOLOv8 标签保存在：{labels_dir}")
