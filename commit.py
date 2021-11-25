import os
import gc
import copy

import torch
from torch import nn
import pandas as pd
import ttach as tta

from glob import glob
from tqdm import tqdm
import numpy as np

from config import config
from dataset import PetfinderLoader
from model import PetfinderModel

df = pd.read_csv(os.path.join(config.root, 'sample_submission.csv')) if not config.debug else pd.read_csv(os.path.join(config.root, 'sample_submission.csv'))[:100]
df['file_path'] = df['Id'].apply(lambda x: os.path.join(config.root, 'test', x + '.jpg'))
df.head()

test_loader = PetfinderLoader(df, df, config).test_dataloader()

models_dir = './checkpoint/commit'
device = "cuda:1"

predicted_labels = None
for model_name in glob(models_dir + '/*'):

    model = PetfinderModel(config)
    model.load_state_dict(torch.load(model_name)['state_dict'])
    model = model.to(device)
    model.eval()
    tta_model = tta.ClassificationTTAWrapper(model, tta.aliases.five_crop_transform(config.image_size, config.image_size))

    temp_preds = None
    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f'Predicting. '):
            images = batch['image'].to(device, non_blocking=True)
            predictions = torch.sigmoid(tta_model(images)).to('cpu').numpy() * 100

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

sub_df = pd.DataFrame()
sub_df['Id'] = df['Id']
sub_df['Pawpularity'] = predicted_labels

sub_df.to_csv('submission.csv', index=False)
print(sub_df)
