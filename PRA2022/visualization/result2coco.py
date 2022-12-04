"""
result:
- image_id
- type
- x
- y
- width
- height
- segmentation

coco:
- images
- - file_name
- - height
- - width
- - id
- annotations
- - image_id
- - id
- - category_id
- - bbox
- - area
- categories
"""
import os
import json
from collections import defaultdict

import argparse

parser = argparse.ArgumentParser(description='Convert result to coco format')
parser.add_argument('--input_file_name', type=str, default="../results/result.json", help='input result file name')
parser.add_argument('--output_file_name', type=str, default="../results/coco_result.json", help='output result file name')

opt = parser.parse_known_args()[0]

# 为了将原始标签的 categories 直接读取
with open("../datasets/detection/val.json", "r") as f:
    val_data = json.load(f)

result = {
    "images": [],
    "annotations": [],
    "categories": val_data["categories"]
}

with open(opt.input_file_name, "r") as f:
    result_data = json.load(f)['result']

# 建立哈希表 image_id -> annotation
hash_map = defaultdict(list)
for res in result_data:
    tmp = {}
    tmp['type'] = res['type']
    tmp['x'] = res['x']
    tmp['y'] = res['y']
    tmp['width'] = res['width']
    tmp['height'] = res['height']
    tmp['segmentation'] = res['segmentation']
    hash_map[res["image_id"]].append(tmp)

ann_id = 0
for k, v in hash_map.items():
    temp_images = {}
    temp_images["file_name"] = f"{k}".zfill(5) + ".jpg"
    temp_images["height"] = 720
    temp_images["width"] = 1280
    temp_images["id"] = k
    result["images"].append(temp_images)
    
    for ann in v:
        if ann['type'] > 7:
            continue
        temp_annotations = {}
        temp_annotations["image_id"] = k
        temp_annotations["id"] = ann_id
        temp_annotations["category_id"] = ann['type']
        temp_annotations["bbox"] = [ann['x'], ann['y'], ann['width'], ann['height']]
        temp_annotations["area"] = ann['width'] * ann['height']
        ann_id += 1
        result["annotations"].append(temp_annotations)

with open(opt.output_file_name, "w") as f:
    json.dump(result, f)