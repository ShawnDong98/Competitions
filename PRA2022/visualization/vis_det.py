import os
import json
import cv2
from pycocotools.coco import COCO


# with open("../results/result.json", "r") as f:
#     result = json.load(f)['result']

# print(len(result))

image_path = "../datasets/my_dataset/images"
coco = COCO("../results/coco_result.json")
# coco = COCO("../datasets/detection/val.json")

# image_path = "../datasets/copy_paste/images"
# coco = COCO("../datasets/copy_paste/train_all_cp.json")


catIds = []

list_imgIds = coco.getImgIds(catIds=catIds)

print(len(list_imgIds))
for imgId in list_imgIds:
    img = coco.loadImgs(imgId)[0] 

    print(img)
    image = cv2.imread(os.path.join(image_path, img['file_name']))
    print(os.path.join(image_path, img['file_name']))
    # print(image)
    
    # 每张图片中可能有多个bbox框， 输出它们的ID
    img_annIds = coco.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None) # 读取这张图片的所有seg_id
    img_anns = coco.loadAnns(img_annIds)

    for i in range(len(img_annIds)):
        x, y, w, h = img_anns[i]['bbox']  # 读取边框
        if img_anns[i]['category_id'] == 1:
            cv2.putText(image, "Motor Vehicle", (int(x) - 20, int(y) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.4, (255, 0, 0), 1, 4)
            image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 2)
        if img_anns[i]['category_id'] == 2:
            cv2.putText(image, "Non-motorized Vehicle", (int(x) - 20, int(y) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 255, 0), 1, 4)
            image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 255, 0), 2)
        if img_anns[i]['category_id'] == 3:
            cv2.putText(image, "Pedestrian", (int(x) - 20, int(y) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 255), 1, 4)
            image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 255), 2)
        if img_anns[i]['category_id'] == 4:
            cv2.putText(image, "Traffic Light-Red Light", (int(x) - 20, int(y) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.4, (128, 0, 0), 1, 4)
            image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (128, 0, 0), 2)
        if img_anns[i]['category_id'] == 5:
            cv2.putText(image, "Traffic Light-Red Light", (int(x) - 20, int(y) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.4, (128, 0, 0), 1, 4)
            image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 128, 0), 2)
        if img_anns[i]['category_id'] == 6:
            cv2.putText(image, "Traffic Light-Yellow Light", (int(x) - 20, int(y) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.4, (0, 0, 128), 1, 4)
            image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (0, 0, 128), 2)
        if img_anns[i]['category_id'] == 7:
            cv2.putText(image, "Traffic Light-Off", (int(x) - 20, int(y) - 5), cv2.FONT_HERSHEY_COMPLEX, 0.4, (128, 128, 128), 1, 4)
            image = cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (128, 128, 128), 2)

    cv2.imshow(str(img['id']), image)
    cv2.waitKey(0)
    cv2.destroyWindow(str(img['id']))