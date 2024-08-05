import os
import re
import random
from collections import OrderedDict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup


import timm

import cv2
import numpy as np
import pandas as pd
from glob import glob
from PIL import Image

from tqdm import tqdm

import pydicom



def set_random_seed(seed: int = 8620, deterministic: bool = False):
    """Set seeds"""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = deterministic  # type: ignore

set_random_seed(2024)

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split(r'(\d+)', text) ]


class RSNA24Dataset(Dataset):
    def __init__(self, root_dir, df, series_desc_df,  image_shape, transform=None):
        self.root_dir = root_dir
        self.df = df
        self.series_desc_df = series_desc_df
        self.image_shape = image_shape
        self.transform = transform
    
    def __len__(self):
        return len(self.df)
    
    def get_img_paths(self, study_id, series_desc):
        pdf = self.series_desc_df[self.series_desc_df['study_id']==study_id]
        pdf_ = pdf[pdf['series_description']==series_desc]
        allimgs = []
        for i, row in pdf_.iterrows():
            pimgs = glob(f'{self.root_dir}/train_images/{study_id}/{row["series_id"]}/*.dcm')
            pimgs = sorted(pimgs, key=natural_keys)
            allimgs.extend(pimgs)
            
        return allimgs
    
    def read_dcm_ret_arr(self, src_path):
        dicom_data = pydicom.dcmread(src_path)
        image = dicom_data.pixel_array
        image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
        img = cv2.resize(image, (self.image_shape[0], self.image_shape[1]),interpolation=cv2.INTER_CUBIC)
        assert img.shape==(self.image_shape[0], self.image_shape[1])
        return img
    
    def __getitem__(self, idx):
        x = np.zeros(self.image_shape, dtype=np.uint8)
        t = self.df.iloc[idx]
        st_id = int(t['study_id'])
        label = t[1:].values.astype(np.int64)

        # Sagittal T1
        allimgs_st1 = self.get_img_paths(st_id, 'Sagittal T1')
        if len(allimgs_st1)==0:
            print(st_id, ': Sagittal T1, has no images')
        else:
            step = len(allimgs_st1) / 10.0
            st = len(allimgs_st1)/2.0 - 4.0*step
            end = len(allimgs_st1)+0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                try:
                    ind2 = max(0, int((i-0.5001).round()))
                    img = self.read_dcm_ret_arr(allimgs_st1[ind2])
                    x[..., j] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Sagittal T1')
                    pass

        # Sagittal T2/STIR
        allimgs_st2 = self.get_img_paths(st_id, 'Sagittal T2/STIR')
        if len(allimgs_st2)==0:
            print(st_id, ': Sagittal T2/STIR, has no images')
            
        else:
            step = len(allimgs_st2) / 10.0
            st = len(allimgs_st2)/2.0 - 4.0*step
            end = len(allimgs_st2)+0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                try:
                    ind2 = max(0, int((i-0.5001).round()))
                    img = self.read_dcm_ret_arr(allimgs_st2[ind2])
                    x[..., j+10] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Sagittal T2/STIR')
                    pass

        # Axial T2
        allimgs_at2 = self.get_img_paths(st_id, 'Axial T2')
        if len(allimgs_at2)==0:
            print(st_id, ': Axial T2, has no images')
            
        else:
            step = len(allimgs_at2) / 10.0
            st = len(allimgs_at2)/2.0 - 4.0*step
            end = len(allimgs_at2)+0.0001

            for j, i in enumerate(np.arange(st, end, step)):
                try:
                    ind2 = max(0, int((i-0.5001).round()))
                    img = self.read_dcm_ret_arr(allimgs_at2[ind2])
                    x[..., j+20] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Axial T2')
                    pass    
            
        assert np.sum(x)>0
            
        if self.transform is not None:
            x = self.transform(image=x)['image']

        x = x.transpose(2, 0, 1)
                
        return x, label
    
    
class RSNA24TestDataset(Dataset):
    def __init__(self, root_dir, df, study_ids, image_shape, transform=None):
        self.root_dir = root_dir
        self.df = df
        self.study_ids = study_ids
        self.image_shape = image_shape
        self.transform = transform


    def __len__(self):
        return len(self.study_ids)
    
    def get_img_paths(self, study_id, series_desc):
        pdf = self.df[self.df['study_id']==study_id]
        pdf_ = pdf[pdf['series_description']==series_desc]
        allimgs = []
        for i, row in pdf_.iterrows():
            pimgs = glob(f'{self.root_dir}/test_images/{study_id}/{row["series_id"]}/*.dcm')
            pimgs = sorted(pimgs, key=natural_keys)
            allimgs.extend(pimgs)
            
        return allimgs
    
    def read_dcm_ret_arr(self, src_path):
        dicom_data = pydicom.dcmread(src_path)
        image = dicom_data.pixel_array
        image = (image - image.min()) / (image.max() - image.min() + 1e-6) * 255
        img = cv2.resize(image, (self.image_shape[0], self.image_shape[1]),interpolation=cv2.INTER_CUBIC)
        assert img.shape==(self.image_shape[0], self.image_shape[1])
        return img

    def __getitem__(self, idx):
        x = np.zeros(self.image_shape, dtype=np.uint8)
        st_id = self.study_ids[idx] 

        # Sagittal T1
        allimgs_st1 = self.get_img_paths(st_id, 'Sagittal T1')
        if len(allimgs_st1)==0:
            print(st_id, ': Sagittal T1, has no images')
        else:
            step = len(allimgs_st1) / 10.0
            st = len(allimgs_st1)/2.0 - 4.0*step
            end = len(allimgs_st1)+0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                try:
                    ind2 = max(0, int((i-0.5001).round()))
                    img = self.read_dcm_ret_arr(allimgs_st1[ind2])
                    x[..., j] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Sagittal T1')
                    pass

        # Sagittal T2/STIR
        allimgs_st2 = self.get_img_paths(st_id, 'Sagittal T2/STIR')
        if len(allimgs_st2)==0:
            print(st_id, ': Sagittal T2/STIR, has no images')
            
        else:
            step = len(allimgs_st2) / 10.0
            st = len(allimgs_st2)/2.0 - 4.0*step
            end = len(allimgs_st2)+0.0001
            for j, i in enumerate(np.arange(st, end, step)):
                try:
                    ind2 = max(0, int((i-0.5001).round()))
                    img = self.read_dcm_ret_arr(allimgs_st2[ind2])
                    x[..., j+10] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Sagittal T2/STIR')
                    pass

        # Axial T2
        allimgs_at2 = self.get_img_paths(st_id, 'Axial T2')
        if len(allimgs_at2)==0:
            print(st_id, ': Axial T2, has no images')
            
        else:
            step = len(allimgs_at2) / 10.0
            st = len(allimgs_at2)/2.0 - 4.0*step
            end = len(allimgs_at2)+0.0001

            for j, i in enumerate(np.arange(st, end, step)):
                try:
                    ind2 = max(0, int((i-0.5001).round()))
                    img = self.read_dcm_ret_arr(allimgs_at2[ind2])
                    x[..., j+20] = img.astype(np.uint8)
                except:
                    print(f'failed to load on {st_id}, Axial T2')
                    pass  

        if self.transform is not None:
            x = self.transform(image=x)['image']

        x = x.transpose(2, 0, 1)
                
        return x, str(st_id)
    
class RSNA24Model(nn.Module):
    def __init__(self, model_name, in_c=30, n_classes=75, pretrained=True, features_only=False):
        super().__init__()
        self.model = timm.create_model(
                                    model_name,
                                    pretrained=pretrained, 
                                    features_only=features_only,
                                    in_chans=in_c,
                                    num_classes=n_classes,
                                    global_pool='avg'
                                    )
    
    def forward(self, x):
        y = self.model(x)
        return y
    





if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root_dir = "../../datasets/Kaggle/RSNA2024/"
    train_df = pd.read_csv(os.path.join(root_dir, 'train.csv'))
    train_series_desc = pd.read_csv(os.path.join(root_dir, "train_series_descriptions.csv"))
    label2id = {'Normal/Mild': 0, 'Moderate':1, 'Severe':2}
    train_df = train_df.replace(label2id)
    dataset = RSNA24Dataset(
        root_dir = root_dir,
        df = train_df, 
        series_desc_df = train_series_desc,
        image_shape = (512, 512, 30), 
        phase = "train",
    )
    print(dataset[0])

    train_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=True,
            pin_memory=True,
            drop_last=False,
            num_workers=8
    )

    val_loader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=8
    )

    # for model in timm.list_models():
    #     if "convnext" in model:
    #         print(model)

    N_LABELS = 25

    model = RSNA24Model('convnext_base_384_in22ft1k', in_c=30, n_classes=N_LABELS*3, pretrained=False)

    model.to(device)

    EPOCHS = 20

    optimizer = AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)

    warmup_steps = EPOCHS/10 * len(train_loader) 
    num_total_steps = EPOCHS * len(train_loader)
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=warmup_steps,
                                                num_training_steps=num_total_steps)
    

    weights = torch.tensor([1.0, 2.0, 4.0])
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
    criterion2 = nn.CrossEntropyLoss(weight=weights)

    best_loss = 1.2
    best_wll = 1.2
    es_step = 0

    USE_AMP = True

    autocast = torch.cuda.amp.autocast(enabled=USE_AMP, dtype=torch.half) # you can use with T4 gpu. or newer

    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=4096)

    OUTPUT_DIR = "save"

    for epoch in range(1, EPOCHS+1):
        print(f'start epoch {epoch}')
        model.train()
        total_loss = 0
        with tqdm(train_loader, leave=True) as pbar:
            optimizer.zero_grad()
            for idx, (x, t) in enumerate(pbar):  
                x = x.to(device)
                t = t.to(device)
                
                with autocast:
                    loss = 0
                    y = model(x)
                    for col in range(N_LABELS):
                        pred = y[:,col*3:col*3+3]
                        gt = t[:,col]
                        loss = loss + criterion(pred, gt) / N_LABELS

                pbar.set_postfix(
                    OrderedDict(
                        loss=f'{loss.item():.6f}',
                        lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                    )
                )
                scaler.scale(loss).backward()

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if scheduler is not None:
                    scheduler.step()  

        train_loss = total_loss/len(train_loader)
        print(f'train_loss:{train_loss:.6f}')

        total_loss = 0
        y_preds = []
        labels = []
        
        model.eval()
        with tqdm(val_loader, leave=True) as pbar:
            with torch.no_grad():
                for idx, (x, t) in enumerate(pbar):
                    x = x.to(device)
                    t = t.to(device)
                        
                    with autocast:
                        loss = 0
                        loss_ema = 0
                        y = model(x)
                        for col in range(N_LABELS):
                            pred = y[:,col*3:col*3+3]
                            gt = t[:,col]
 
                            loss = loss + criterion(pred, gt) / N_LABELS
                            y_pred = pred.float()
                            y_preds.append(y_pred.cpu())
                            labels.append(gt.cpu())

                        total_loss += loss.item()   
    
        val_loss = total_loss/len(val_loader)
        
        y_preds = torch.cat(y_preds, dim=0)
        labels = torch.cat(labels)
        val_wll = criterion2(y_preds, labels)
        
        print(f'val_loss:{val_loss:.6f}, val_wll:{val_wll:.6f}')

        if val_loss < best_loss or val_wll < best_wll:

            if val_loss < best_loss:
                print(f'epoch:{epoch}, best loss updated from {best_loss:.6f} to {val_loss:.6f}')
                best_loss = val_loss
                
            if val_wll < best_wll:
                print(f'epoch:{epoch}, best wll_metric updated from {best_wll:.6f} to {val_wll:.6f}')
                best_wll = val_wll
                fname = f'{OUTPUT_DIR}/best_wll_model_epoch{epoch}_{best_wll:.6f}.pt'
                torch.save(model.state_dict(), fname)

