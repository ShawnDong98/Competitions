import os
from collections import OrderedDict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from config import config
from dataset import PetfinderLoader, mixup
from model import PetfinderModel

from engine.trainer.trainer import Trainer
from engine.trainer.utils import seed_everything
from engine.utils.path import mkdir_or_exist


def batch_processor(model, batch, train_mode):
    image, label, filename = batch['image'], batch['label'], batch['filename']
    image = image.float().cuda()
    label = label.float().cuda() / 100.
    logit = model(image).squeeze(1)
    loss = F.binary_cross_entropy_with_logits(logit, label)

    pred =  logit.sigmoid().detach().cpu() * 100
    label  =  label.detach().cpu() * 100

    mse = torch.sqrt(((label - pred) ** 2).mean())

    log_vars = OrderedDict()
    log_vars['loss'] = loss.item()
    log_vars['mse'] = mse.item()
    log_vars['pred'] = pred
    log_vars['label'] = label
    outputs = dict(loss=loss, log_vars=log_vars, num_samples=image.size(0))
    return outputs



def main():
    seed_everything(config.seed)
    df = pd.read_csv(os.path.join(config.root, 'train.csv')) if not config.debug else pd.read_csv(os.path.join(config.root, 'train.csv'))[:1000]
    num_bins = int(np.floor(1 + 3.3 * np.log2(len(df))))
    df.loc[:, 'bins'] = pd.cut(df['Pawpularity'], bins=num_bins, labels=False)
    df['file_path'] = df['Id'].apply(lambda x: os.path.join(config.root, 'train', x + '.jpg'))
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df['file_path'], df['bins'])):
        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df = df.loc[val_idx].reset_index(drop=True)

        Loader = PetfinderLoader(train_df, val_df, config)

        model = PetfinderModel(config)

        if torch.cuda.is_available():
            model = nn.DataParallel(model).cuda()

        work_dir = os.path.join(config.work_dir,
                                config.model.name + '_1202',   f'version_{fold}')
        mkdir_or_exist(work_dir)

        trainer = Trainer(
            config,
            model,
            batch_processor,
            config.optimizer,
            work_dir,
            config.log_level
        )

        trainer.register_training_hooks(
            lr_config = config.lr_config,
            optimizer_config = config.optimizer_config,
            checkpoint_config = config.checkpoint_config,
            log_config = config.log_config,
            momentum_config = config.momentum_config,
            earlystopping_config = config.earlystopping_config
        )

        trainer.fit([Loader.train_dataloader(), Loader.val_dataloader()], config.workflow, config.epochs)

if __name__ == "__main__":
    main()
