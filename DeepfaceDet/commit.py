import os
import gc
import copy

import torch
from torch import nn
import pandas as pd

from glob import glob
from tqdm import tqdm
import numpy as np

from config import config
from dataset import FackFaceDetLoader
from model import FackFaceDetModel

df = pd.read_csv(os.path.join(config.root, 'submission.csv'), sep='\t') if not config.debug else pd.read_csv(os.path.join(config.root, 'submission.csv'), sep='\t')[:100]
df_org = copy.deepcopy(df)
df['fnames'] = df['fnames'].apply(lambda x: os.path.join(config.root, 'image', 'test', x))
df.head()

test_loader = FackFaceDetLoader(df, df, config).val_dataloader()

models_dir = './checkpoint/commit'
device = "cuda:1" 

predicted_labels = None
for model_name in glob(models_dir + '/*'):

    model = FackFaceDetModel(config)
    model.load_state_dict(torch.load(model_name)['state_dict'])
    model = model.cuda()
    model.eval()
    
    temp_preds = None
    with torch.no_grad():
        for (images, target) in tqdm(test_loader, desc=f'Predicting. '):
            images = images.to("cuda", non_blocking=True)
            predictions = torch.sigmoid(model(images)).to('cpu').numpy()
            
            if temp_preds is None:
                temp_preds = predictions
            else:
                temp_preds = np.vstack((temp_preds, predictions))

    if predicted_labels is None:
        predicted_labels = temp_preds
    else:
        predicted_labels += temp_preds
    
#     del model
    gc.collect()
predicted_labels /= (len(glob(models_dir + '/*')))
predicted_labels[predicted_labels >= 0.5] = 1
predicted_labels[predicted_labels < 0.5] = 0

sub_df = pd.DataFrame()
sub_df['fnames'] = df_org['fnames']
sub_df['label'] = predicted_labels.astype(int)

sub_df.to_csv('submission.csv', index=False, sep='\t')