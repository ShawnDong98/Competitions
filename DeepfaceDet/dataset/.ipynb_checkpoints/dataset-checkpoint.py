import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


class FackFaceDetDataset(Dataset):
    def __init__(
        self, 
        df, 
        transform=None
    ):
        super().__init__()
        self.df = df
        self._X = self.df['fnames']
        self._Y = self.df['label']
        self.transform = transform

    def __len__(self):
        return len(self._X)

    def __getitem__(self, idx):
        fname = self._X[idx]
        label = self._Y[idx]
        img = cv2.imread(fname)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)['image']
        return img, label



class FackFaceDetLoader():
    def __init__(
        self,
        train_df,
        val_df,
        cfg,
    ):
        self._train_df = train_df
        self._val_df = val_df
        self._cfg = cfg

        self.train_transform = A.Compose(
            [
                A.Resize(cfg.image_size, cfg.image_size),
                A.Normalize(
                    mean = [0.485, 0.456, 0.406],
                    std = [0.229, 0.224, 0.225],
                ),
                A.HorizontalFlip(p = 0.5),
                A.VerticalFlip(p = 0.5),
                A.Rotate(limit = 180, p = 0.7),
                A.ShiftScaleRotate(
                    shift_limit = 0.1,
                    scale_limit = 0.1,
                    rotate_limit = 45,
                    p = 0.5
                ),
                A.HueSaturationValue(
                    hue_shift_limit = 0.2, 
                    sat_shift_limit = 0.2, 
                    val_shift_limit = 0.2, 
                    p = 0.5
                ),
                A.RandomBrightnessContrast(
                    brightness_limit = (-0.1, 0.1),
                    contrast_limit = (-0.1, 0.1),
                    p = 0.5
                ),
                ToTensorV2(p=1.0)
            ]
        )

        self.val_transform = A.Compose(
            [
                A.Resize(cfg.image_size, cfg.image_size),
                A.Normalize(
                    mean = [0.485, 0.456, 0.406],
                    std = [0.229, 0.224, 0.225],
                ),
                ToTensorV2(p=1.0)
            ]
        )
        self.test_transform = A.Compose(
            [
                A.Resize(cfg.image_size_tta, cfg.image_size_tta),
                A.Normalize(
                    mean = [0.485, 0.456, 0.406],
                    std = [0.229, 0.224, 0.225],
                ),
                ToTensorV2(p=1.0)
            ]
        )

    def __create_dataset(self, mode='train'):
        if mode == 'train':
            return FackFaceDetDataset(self._train_df, self.train_transform)
        elif mode == 'val':
            return FackFaceDetDataset(self._val_df, self.val_transform)
        else:
            return FackFaceDetDataset(self._val_df, self.test_transform)

    def train_dataloader(self):
        dataset = self.__create_dataset(mode='train')
        return DataLoader(dataset, **self._cfg.train_loader)

    def val_dataloader(self):
        dataset = self.__create_dataset(mode='val')
        return DataLoader(dataset, **self._cfg.val_loader)
    
    def test_dataloader(self):
        dataset = self.__create_dataset(mode='test')
        return DataLoader(dataset, **self._cfg.val_loader)