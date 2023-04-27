import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import os
from PIL import Image


class HLBCaptionInferenceDataset(Dataset):
    
    def __init__(
        self,
        root_dir,
        trans = None
    ):
        super().__init__()
        from glob import glob
        self.root_dir = root_dir
        self.images = glob(os.path.join(self.root_dir, "*.jpg"))

        self.trans = trans

    def __getitem__(self, idx):
        item = {}
        image = Image.open(self.images[idx]).convert("RGB")

        if self.trans:
            image = self.trans(image)

        item['image'] = image
        item['filename'] = self.images[idx].split('/')[-1]

        return item

    def __len__(self):
        return len(self.images)