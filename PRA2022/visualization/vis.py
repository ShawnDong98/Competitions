"""
写一个字典：
{
    图像id : 
    {
        "real" : [{
            x, y, w, h, segmentation 
        }] 
        "predict" : [{
            x, y, w, h, segmentation
        }]
    }
} 
"""

import os
import json
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image

from tqdm import tqdm

img_root = "../datasets/detection/JPEGImages/"
seg_label_root = "../datasets/seg/Annotations/"

with open("result_20221020.json", "r") as f:
    json_result = json.load(f)['result']

compare_dict = defaultdict(lambda:defaultdict(list))

# 获取预测的结果

for res in tqdm(json_result):
    compare_dict[res['image_id']]['pred'].append(
        {
            'label' : res['type'],
            'x' : res['x'],
            'y' : res['y'],
            'w' : res['width'],
            'h' : res['height'],
            'segmentation' : res['segmentation']
        }
    )



with open("../datasets/detection/train.json", "r") as f:
    annotations = json.load(f)['annotations']

# 获取 detection 的 ground truth

for ann in tqdm(annotations):
    if ann['image_id'] in compare_dict:
        compare_dict[ann['image_id']]['real'].append(
        {
            'label' : ann['category_id'],
            'x' : ann['bbox'][0],
            'y' : ann['bbox'][1],
            'w' : ann['bbox'][2],
            'h' : ann['bbox'][3],
            'segmentation' : ann['segmentation']
        }
    )

# 获取 segmentation 的 ground truth

seg_gt = defaultdict(list)
image_id_set = set([res['image_id'] for res in json_result])
for image_id in tqdm(image_id_set):
    label_img = np.asarray(Image.open(os.path.join(seg_label_root, str(image_id).zfill(5) + ".png")))
    max_value = np.max(label_img)
    # print(max_value)
    if max_value > 0:
        for iiii in range(1, max_value + 1):
            gt = np.zeros(label_img.shape, dtype=np.uint8)
            gt[label_img == iiii] = 255
            label = iiii + 7
            contours, hierarchy = cv2.findContours(gt.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # print(contours)
            for contour in contours: 
                rect = cv2.boundingRect(contour)
                segmentations = []
                for point in contour:
                    segmentations.append(round(point[0][0], 0))
                    segmentations.append(round(point[0][1], 0))

                compare_dict[image_id]['real'].append(
                    {
                        'label' : label,
                        'x' : round(rect[0] * 1., 0),
                        'y' : round(rect[1] * 1., 0),
                        'w' : round(rect[2] * 1., 0),
                        'h' : round(rect[3] * 1., 0),
                        'segmentation' : [segmentations]
                    }
                )

    # print(label.shape)
    # cv2.imshow("label", label)
    # cv2.waitKey(0)


# print(compare_dict)

for img_id, v in compare_dict.items():
    real = v['real']
    pred = v['pred']
    img = cv2.imread(os.path.join(img_root, str(img_id).zfill(5) + ".jpg"))
    real_contours_cat7 = []
    real_contours_cat8 = []
    real_contours_cat9 = []
    for r in real:
        if r['label'] == 8:
            contour = []
            for pt1, pt2 in zip(r['segmentation'][0][:-1:2], r['segmentation'][0][1::2]):
                point = [pt1, pt2]
                contour.append(np.array(point))
            real_contours_cat7.append(np.stack(contour, axis=0)[:, None, :].astype(int))
        if r['label'] == 9:
            contour = []
            for pt1, pt2 in zip(r['segmentation'][0][:-1:2], r['segmentation'][0][1::2]):
                point = [pt1, pt2]
                contour.append(np.array(point))
            real_contours_cat8.append(np.stack(contour, axis=0)[:, None, :].astype(int))
        if r['label'] == 10:
            contour = []
            for pt1, pt2 in zip(r['segmentation'][0][:-1:2], r['segmentation'][0][1::2]):
                point = [pt1, pt2]
                contour.append(np.array(point))
            real_contours_cat9.append(np.stack(contour, axis=0)[:, None, :].astype(int))
    cv2.drawContours(img, real_contours_cat7, -1,(255,0,0),3)  
    cv2.drawContours(img, real_contours_cat8, -1,(0,255,0),3)  
    cv2.drawContours(img, real_contours_cat9, -1,(0,0,255),3)  


    pred_contours_cat8 = []
    pred_contours_cat9 = []
    pred_contours_cat10 = []
    for p in pred:
        # if len(p['segmentation']) == 0:
        img = cv2.rectangle(img, (int(p['x']), int(p['y'])), (int(p['x']) + int(p['w']), int(p['y']) + int(p['h'])), (0, 255, 255), 3)
        # else:
            # if p['label'] == 8:
            #     contour = []
            #     for pt1, pt2 in zip(p['segmentation'][0][:-1:2], p['segmentation'][0][1::2]):
            #         point = [pt1, pt2]
            #         contour.append(np.array(point))
            #     pred_contours_cat8.append(np.stack(contour, axis=0)[:, None, :].astype(int))
            # elif p['label'] == 9:
            #     contour = []
            #     for pt1, pt2 in zip(p['segmentation'][0][:-1:2], p['segmentation'][0][1::2]):
            #         point = [pt1, pt2]
            #         contour.append(np.array(point))
            #     pred_contours_cat9.append(np.stack(contour, axis=0)[:, None, :].astype(int))
            # elif p['label'] == 10:
            #     contour = []
            #     for pt1, pt2 in zip(p['segmentation'][0][:-1:2], p['segmentation'][0][1::2]):
            #         point = [pt1, pt2]
            #         contour.append(np.array(point))
            #     pred_contours_cat10.append(np.stack(contour, axis=0)[:, None, :].astype(int))
            # pass

    cv2.drawContours(img, pred_contours_cat8, -1,(255,255,0),3)        
    cv2.drawContours(img, pred_contours_cat9, -1,(0,255,255),3)        
    cv2.drawContours(img, pred_contours_cat10, -1,(255,0,255),3)        

    cv2.imshow("test", img)
    cv2.waitKey(0)



"""
for res in json_result:
    path = os.path.join(img_root, str(res['image_id']).zfill(5) + '.jpg')
    print(path)
    img = cv2.imread(path)
    if len(res['segmentation']) == 0:
        pass
        # x, y, w, h = int(res['x']), int(res['y']), int(res['width']), int(res['height'])
        # img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0))
    else:
        contours = []
        contour = []
        for pt1, pt2 in zip(res['segmentation'][0][:-1:2], res['segmentation'][0][1::2]):
            point = [pt1, pt2]
            contour.append(np.array(point))
            
        contours.append(np.stack(contour, axis=0)[:, None, :].astype(int))
        cv2.drawContours(img, contours, -1,(0,255,0),3)        
        cv2.imshow("test", img)
        cv2.waitKey(0)
"""

    
