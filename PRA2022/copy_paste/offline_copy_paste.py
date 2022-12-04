import os
import json
from collections import defaultdict
import time
import random

import cv2

from tqdm import tqdm

random.seed(42)

root = "../datasets/det_all/"
root_image = "../datasets/det_all/images/"
root_target = "../datasets/copy_paste/"
root_target_image = "../datasets/copy_paste/images/"

with  open(os.path.join(root, "train_all.json"), "r") as f:
    train_all = json.load(f)

ducks = defaultdict(list)
imgid2img_path = {}


for image in tqdm(train_all['images']):
    imgid2img_path[image['id']] = os.path.join(root_image, image['file_name'])

cnt = 0
for ann in tqdm(train_all['annotations']):
    cat_id = ann['category_id']
    if  cat_id > 1 and cat_id < 8:
        img = cv2.imread(imgid2img_path[ann['image_id']])
        x, y, w, h = map(int, ann['bbox'])
        patch = img[y:y+h, x:x+w, :]
        ducks[f'cat{cat_id}'].append(
            {"patch" : patch, "category_id":  ann['category_id'], "bbox": ann['bbox']}
        )
    cnt += 1
    if cnt > 10000:
        break
    
cnt = 0
for key, value in ducks.items():
    print(f"{key}: {len(value)}")
    if key == "cat2":
        for v in value:
            cv2.imwrite(os.path.join("non_motor", f"{cnt}.png"), v['patch']) 
            cnt += 1
    if key == "cat3":
        for v in value:
            cv2.imwrite(os.path.join("pedestrian", f"{cnt}.png"), v['patch']) 
            cnt += 1
    if key == "cat4":
        for v in value:
            cv2.imwrite(os.path.join("red_light", f"{cnt}.png"), v['patch']) 
            cnt += 1
    elif key == "cat5":
        for v in value:
            cv2.imwrite(os.path.join("yellow_light", f"{cnt}.png"), v['patch'])
            cnt += 1
    elif key == "cat6":
        for v in value:
            cv2.imwrite(os.path.join("green_light", f"{cnt}.png"), v['patch'])
            cnt += 1
    elif key == "cat7":
        for v in value:
            cv2.imwrite(os.path.join("off_light", f"{cnt}.png"), v['patch'])
            cnt += 1

# cp = {
#     "images" : [],
#     "annotations" : train_all['annotations'].copy(),
#     "categories" : train_all['categories'].copy()
# }

# annotation_id = 200000

# cnt = 0
# for image in tqdm(train_all['images']):
#     temp_image = {}
#     temp_image['file_name'] = image['file_name'].split(".")[0] + ".jpg"
#     temp_image['height'] = image['height']
#     temp_image['width'] = image['width']
#     temp_image['id'] = image['id']
#     cp['images'].append(temp_image)

#     img = cv2.imread(os.path.join(root_image, image['file_name']))
#     duck_list = []
    
#     if random.randint(0, 9) == 0:
#         index = random.randint(0, len(ducks['cat5']) - 1)
#         duck = ducks['cat5'][index]
#         duck_list.append(duck)

#         for duck in duck_list:
#             H, W, C = img.shape
#             H_, W_, C_ = duck['patch'].shape
#             px, py = random.randint(0, W - W_), random.randint(0, H - H_)

#             img[py:py+H_, px:px+W_, :] = duck['patch']

#             temp_ann = {}
#             temp_ann['image_id'] = temp_image["id"]
#             temp_ann['id'] = annotation_id
#             temp_ann['category_id'] = duck['category_id']
#             temp_ann['bbox'] = [px, py, W_, H_]
#             temp_ann['area'] = W_ * H_
#             temp_ann['iscrowd'] = 0
#             temp_ann['segmentation'] = []
#             cp['annotations'].append(temp_ann)
#             annotation_id += 1
#             # print(temp_ann)
#             # cv2.imshow("patch", duck['patch'])
#             # cv2.waitKey(0)

#     cv2.imwrite(os.path.join(root_target_image, temp_image['file_name']), img)

    
#     # cnt += 1
#     # if cnt > 1000:
#     #     break


# with open(os.path.join(root_target, "cp.json"), "w") as f:
#     json.dump(cp, f)


