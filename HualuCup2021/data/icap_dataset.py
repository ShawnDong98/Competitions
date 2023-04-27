import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

import os
from PIL import Image

class ImageCaptionDataset(Dataset):
    def __init__(
        self,
        root_dir,
        filenames,
        captions, 
        tokenizer,
        max_length,
        trans = None
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.filenames = filenames
        self.captions = captions
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.trans = trans

    def __getitem__(self, index: int):
        item = {}
        ids, tokens = self.tokenizer(self.captions[index], return_tokens = True)
        item['input_ids'] = torch.tensor(ids, dtype=torch.long)
        item['tokens'] = self.captions[index]
        image = Image.open(os.path.join(self.root_dir, self.filenames[index])).convert("RGB")
        if self.trans is not None:
            image = self.trans(image)
        item['image'] = image
        item['filename'] = self.filenames[index]

        return item

    def __len__(self):
        return len(self.captions)


