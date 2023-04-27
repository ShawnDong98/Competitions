import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import os
from PIL import Image

class COCOCaptionInferenceDataset(Dataset):
    """
    输入文件 为 Karpthy 格式的 COCO 文件
    args: 
        code : 
     
    return: 
        code : 
    """
    def __init__(
        self,
        root_dir,
        images,
        trans = None
    ):
        super().__init__()
        self.root_dir = root_dir
        self.trans = trans

        self.images = images

    def __getitem__(self, idx):
        item = {}
        image = Image.open(os.path.join(self.root_dir, self.images[idx]['filename'])).convert("RGB")

        if self.trans:
            image = self.trans(image)

        item['image'] = image
        item['filename'] = self.images[idx]['filename']
        item['image_id'] = self.images[idx]['imgid']

        return item

    def __len__(self):
        return len(self.images)


class COCOStyleEvalInferenceDataset(Dataset):
    """
    COCO Caption 输入文件 为 评价指标所需的 annotation
    args: 
        code : 
     
    return: 
        code : 
    """
    def __init__(
        self,
        root_dir,
        images,
        trans = None
    ):
        super().__init__()
        self.root_dir = root_dir
        self.trans = trans

        self.images = images

    def __getitem__(self, idx):
        item = {}
        image = Image.open(os.path.join(self.root_dir, self.images[idx]['file_name'])).convert("RGB")

        if self.trans:
            image = self.trans(image)

        item['image'] = image
        item['filename'] = self.images[idx]['file_name']
        item['image_id'] = self.images[idx]['id']

        return item

    def __len__(self):
        return len(self.images)
