import os
import json

import cv2
import numpy as np

from glob import glob


# img_paths = sorted(glob(os.path.join("./datasets/det_all/images/", "*.jpg")))

# ann_paths = sorted(glob(os.path.join("./datasets/TrainData/初赛/train/label/", "*.json")))

img = cv2.imread(os.path.join("./datasets/det_all/images/", "00032.jpg"))
with open(os.path.join("./datasets/TrainData/初赛/train/label/", "00032.json")) as f: 
    anns = json.load(f)

# contours_cat7 = []
# contours_cat8 = []
# contours_cat9 = []

# for img_path, ann_path in zip(img_paths, ann_paths):
#     img_id = img_path.split("/")[-1]
#     img = cv2.imread(img_path)
#     with open(ann_path) as f: 
#        anns = json.load(f)
#     for ann in anns:
#         if ann['type'] > 3 and ann['type'] < 8:
#             cv2.rectangle(img, (ann['x'], ann['y']), (ann['x']+ann['width'], ann['y']+ann['height']), (0, 255, 0), 2)


#     cv2.imshow(img_id, img)
#     # cv2.imwrite("task.png", img)
#     cv2.waitKey(0)


for ann in anns:
    if ann['type'] > 3 and ann['type'] < 8:
        cv2.rectangle(img, (ann['x'], ann['y']), (ann['x']+ann['width'], ann['y']+ann['height']), (0, 255, 0), 2)
# if ann['type'] == 8:
#     contour = []
#     for pt1, pt2 in zip(ann['segmentation'][0][:-1:2], ann['segmentation'][0][1::2]):
#         point = [pt1, pt2]
#         contour.append(np.array(point))
#     contours_cat7.append(np.stack(contour, axis=0)[:, None, :].astype(int))
# if ann['type'] == 9:
#     contour = []
#     for pt1, pt2 in zip(ann['segmentation'][0][:-1:2], ann['segmentation'][0][1::2]):
#         point = [pt1, pt2]
#         contour.append(np.array(point))
#     contours_cat8.append(np.stack(contour, axis=0)[:, None, :].astype(int))
# if ann['type'] == 10:
#     contour = []
#     for pt1, pt2 in zip(ann['segmentation'][0][:-1:2], ann['segmentation'][0][1::2]):
#         point = [pt1, pt2]
#         contour.append(np.array(point))
#     contours_cat9.append(np.stack(contour, axis=0)[:, None, :].astype(int))
# cv2.drawContours(img, contours_cat7, -1, (180, 0, 0), -1)  
# cv2.drawContours(img, contours_cat8, -1, (0, 180, 0), -1)  
# cv2.drawContours(img, contours_cat9, -1, (0, 0, 180), -1)  


cv2.imshow("img", img)
cv2.imwrite("task.png", img)
cv2.waitKey(0)