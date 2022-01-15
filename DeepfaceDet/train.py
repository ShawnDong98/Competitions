import os
from collections import OrderedDict
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

from config import config
from dataset import FackFaceDetLoader, mixup
from model import FackFaceDetModel

from engine.trainer.trainer import Trainer
from engine.trainer.utils import seed_everything
from engine.utils.path import mkdir_or_exist


def batch_processor(model, batch, train_mode):
    img, label = batch
    img = img.float().cuda()
    label = label.float().cuda()
    if torch.rand(1)[0] < 0.5 and train_mode == 'train':
        mix_images, target_a, target_b, lam = mixup(img, label, alpha=0.5)
        pred = model(mix_images).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(pred, target_a) * lam \
            + F.binary_cross_entropy_with_logits(pred, target_b) * (1 - lam)
    else:
        pred = model(img).squeeze(1)
        loss = F.binary_cross_entropy_with_logits(pred, label)
    f1_pred = pred.sigmoid().clone()
    f1_pred[f1_pred > 0.5] =  2
    f1_pred[f1_pred <= 0.5] = 1
    f1 = f1_score((label+1).cpu().numpy(), f1_pred.detach().cpu().numpy())
    log_vars = OrderedDict()
    log_vars['loss'] = loss.item()
    log_vars['f1_score'] = f1
    outputs = dict(loss=loss, log_vars=log_vars, num_samples=img.size(0))
    return outputs

def main():
    seed_everything()
    df = pd.read_csv(os.path.join(config.root, 'train.csv'), sep='\t') if not config.debug else pd.read_csv(os.path.join(config.root, 'train.csv'), sep='\t')[:1000]

    df['fnames'] = df['fnames'].apply(lambda x: os.path.join(config.root, 'image', 'train', x))
    skf = StratifiedKFold(n_splits=config.n_splits, shuffle=True, random_state=config.seed)

    for fold, (train_idx, val_idx) in enumerate(skf.split(df['fnames'], df['label'])):
        train_df = df.loc[train_idx].reset_index(drop=True)
        val_df = df.loc[val_idx].reset_index(drop=True)

        Loader = FackFaceDetLoader(train_df, val_df, config)

        model = FackFaceDetModel(config)

        if torch.cuda.is_available():
            model = nn.DataParallel(model).cuda()

        work_dir = os.path.join(config.work_dir, config.model.name + '1204', f'version_{fold}')
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
