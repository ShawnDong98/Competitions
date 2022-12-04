import pytorch_lightning as pl
pl.seed_everything(42)

import os
import sys
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import collections
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
from tasks.wc_model import WCModel, FGM, PGD
from tasks.wc_data import WeChatDataset

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from category_id_map import lv2id_to_lv1id, CATEGORY_ID_LIST, lv2id_to_category_id, category_id_to_lv1id, category_id_to_lv2id

from losses import focal_f1_loss, FocalLoss, FocalLossWithSmoothing



from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor
import wandb
from pytorch_lightning.loggers import WandbLogger

from pprint import pprint
import argparse

parser = argparse.ArgumentParser(description="LXMERT Finetune")
parser.add_argument("--label_csv", type=str, default="/home/datasets/2022WeChat/data/annotations/label.csv")
parser.add_argument("--test_csv", type=str, default="/home/datasets/2022WeChat/data/annotations/test_b.csv")
parser.add_argument("--train_dir", type=str, default="/home/datasets/2022WeChat/data/zip_feats/labeled")
parser.add_argument("--test_dir", type=str, default="/home/datasets/2022WeChat/data/zip_feats/test_b")
parser.add_argument("--train_json", type=str, default="/home/datasets/2022WeChat/data/annotations/labeled.json")
parser.add_argument("--test_json", type=str, default="/home/datasets/2022WeChat/data/annotations/test_b.json")
parser.add_argument("--text_maxlen", type=int, default=300)
parser.add_argument("--frame_maxlen", type=int, default=32)
parser.add_argument("--fold", type=int, default=0)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--num_workers", type=int, default=4)
parser.add_argument("--num_classes", type=int, default=len(CATEGORY_ID_LIST))
parser.add_argument("--lxmert_lr", type=float, default=2e-5)
parser.add_argument("--others_lr", type=float, default=5e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--checkpoints_dir", type=str, default="checkpoints_finetune")
parser.add_argument("--llayers", type=int, default=9)
parser.add_argument("--xlayers", type=int, default=5)
parser.add_argument("--rlayers", type=int, default=5)
parser.add_argument("--frame_embedding_size", type=int, default=768)
parser.add_argument("--vlad_cluster_size", type=int, default=64)
parser.add_argument("--vlad_hidden_size", type=int, default=1024)
parser.add_argument("--dropout", type=int, default=0.2)
parser.add_argument("--se_ratio", type=int, default=8)
parser.add_argument("--fc_size", type=int, default=768)
parser.add_argument("--model_path", type=str, default="/home/2022Wechat/2021_QQ_AIAC_Track1_1st/input/pretrain-model/chinese-macbert-base")
parser.add_argument("--from_scratch", type=bool, default=False)
parser.add_argument("--pretrained_path", type=str, default="/home/2022Wechat/lxmert/checkpoints_pretrain/Epoch06")
parser.add_argument("--one_cycle_pct", type=float, default=0.25)
parser.add_argument("--adv_mode", type=str, default="pgd")
parser.add_argument("--epsilon", type=int, default=1)
parser.add_argument("--emb_name", type=str, default="word_embeddings.")
parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--adv_k", type=int, default=3)
parser.add_argument("--grad_clip_val", type=float, default=0.1)
parser.add_argument("--n_gpus", type=int, default=1)
parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--DEBUG", type=bool, default=False)

config = parser.parse_known_args()[0]
pprint(config)

# wandb_logger = WandbLogger(
#     project = "lxmert_WeChat",
#     config = config
# )

def evaluate(predictions, labels):
    # prediction and labels are all level-2 class ids

    lv1_preds = [lv2id_to_lv1id(lv2id) for lv2id in predictions]
    lv1_labels = [lv2id_to_lv1id(lv2id) for lv2id in labels]

    lv2_f1_micro = f1_score(labels, predictions, average='micro')
    lv2_f1_macro = f1_score(labels, predictions, average='macro')
    lv1_f1_micro = f1_score(lv1_labels, lv1_preds, average='micro')
    lv1_f1_macro = f1_score(lv1_labels, lv1_preds, average='macro')
    mean_f1 = (lv2_f1_macro + lv1_f1_macro + lv1_f1_micro + lv2_f1_micro) / 4.0

    eval_results = {'mean_f1': mean_f1,
                    'lv1_acc': accuracy_score(lv1_labels, lv1_preds),
                    'lv2_acc': accuracy_score(labels, predictions),
                    'lv1_f1_micro': lv1_f1_micro,
                    'lv1_f1_macro': lv1_f1_macro,
                    'lv2_f1_micro': lv2_f1_micro,
                    'lv2_f1_macro': lv2_f1_macro,
                    }

    return eval_results


class LitDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.label = pd.read_csv(self.config.label_csv)

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            dataset = WeChatDataset(
                model_path = self.config.model_path,
                data_dir = self.config.train_dir,
                get_title=True,
                get_asr=True,
                get_ocr=True,
                get_frame=True,
                label_json=self.config.train_json, 
                get_vid=True,
                get_category=True,
                text_maxlen=self.config.text_maxlen, 
                frame_maxlen=self.config.frame_maxlen,
            )
            
            train_ix = self.label[self.label['fold']!=self.config.fold].index 
            val_ix = self.label[self.label['fold']==self.config.fold].index
            self.train_dataset = torch.utils.data.Subset(dataset, train_ix)
            self.val_dataset = torch.utils.data.Subset(dataset, val_ix)

        if stage == "test" or stage is None:
            self.test_dataset = WeChatDataset(
                model_path = self.config.model_path,
                data_dir = self.config.test_dir,
                get_title=True,
                get_asr=True,
                get_ocr=True,
                get_frame=True,
                label_json=self.config.test_json, 
                get_vid=True,
                get_category=False,
                text_maxlen=self.config.text_maxlen, 
                frame_maxlen=self.config.frame_maxlen,
            )
    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset,train=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, train=False)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset, train=False)

    def _dataloader(self, dataset: WeChatDataset, train: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=train,
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=train,
        )

class LitModule(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.model = WCModel(self.config)
        self.model.lxrt_encoder.load(self.config.pretrained_path)

        
        self.focal_loss_with_smoothing = FocalLossWithSmoothing(
            num_classes = self.config.num_classes,
            gamma = 2,
            lb_smooth = 0.1,
            ignore_index = None,
            alpha = None
        )

        self.automatic_optimization = False

    def forward(self,
                input_ids,
                input_masks,
                segment_ids,
                visual_feats,
                visual_attention_mask,
    ):
        logits = self.model(
            input_ids = input_ids,
            input_masks = input_masks,
            segment_ids = segment_ids,
            feat = visual_feats,
            visual_attention_mask = visual_attention_mask,
        )

        return logits

    def configure_optimizers(self):
        param_group = [
            {'params': [p for n, p in self.model.named_parameters() if 'lxrt_encoder' in n], 'lr' : self.config.lxmert_lr},
            {'params': [p for n, p in self.model.named_parameters() if 'lxrt_encoder' not in n], 'lr': self.config.others_lr}
        ]
        optimizer = torch.optim.AdamW(
            param_group, 
            weight_decay=self.config.weight_decay,
        )
        print(self.config.len_train_dl)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr = [self.config.lxmert_lr, self.config.others_lr],
            steps_per_epoch=self.config.len_train_dl,
            epochs=self.config.epochs,
            pct_start = self.config.one_cycle_pct,
        )
        scheduler = {"scheduler": scheduler, "interval": "step"}

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        scheduler = self.lr_schedulers()
        opt.zero_grad()
        ret = self._step(batch, "train")
        loss = ret['loss']
        self.manual_backward(loss)
        
        if self.config.adv_mode == 'fgm' or self.config.adv_mode == 'adv_all':
            # print("FGM Attack...")
            fgm = FGM(self.model,
                    self.config.epsilon, 
                    self.config.emb_name, 
            )
            fgm.attack()
            adv_ret = self._step(batch, "train")
            adv_loss = adv_ret['loss']
            self.manual_backward(adv_loss)
            fgm.restore()

        elif self.config.adv_mode == 'pgd' or self.config.adv_mode == 'adv_all':
            # print("PGD Attack...")
            pgd = PGD(self.model, 
                    self.config.epsilon, 
                    self.config.emb_name, 
                    self.config.alpha
            )
            pgd.backup_grad()
            for t in range(self.config.adv_k):
                pgd.attack(is_first_attack=(t == 0))
                if t != self.config.adv_k - 1:
                    opt.zero_grad()
                else:
                    pgd.restore_grad()
                adv_ret = self._step(batch, "train")
                adv_loss = adv_ret['loss']
                self.manual_backward(adv_loss)
                
            pgd.restore()
            
        torch.nn.utils.clip_grad_norm_(
            self.parameters(), 
            max_norm = self.config.grad_clip_val, 
            norm_type=2.0)
        opt.step()
        scheduler.step()

        self.trainer.train_loop.running_loss.append(loss)
    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def validation_epoch_end(self, validation_step_outputs):
        preds = torch.cat([x['pred']for x in validation_step_outputs])
        targets = torch.cat([x['target'] for x in validation_step_outputs])
        logits = torch.cat([x['logits'] for x in validation_step_outputs])

        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        logits = logits.cpu().numpy()

        np.save(os.path.join(self.config.checkpoints_dir, f"test_b_logits_fold{self.config.fold}.npy"), logits)
        np.save(os.path.join(self.config.checkpoints_dir, f"test_b_target_fold{self.config.fold}.npy"), targets)

        results = evaluate(preds, targets)

        self.log_dict(results)
        pprint(results)

        lv1_predictions = [category_id_to_lv1id(lv2id_to_category_id(lv2id)) for lv2id in preds]
        lv1_labels = [category_id_to_lv1id(lv2id_to_category_id(lv2id)) for lv2id in targets]

        lv1_conf_mx = confusion_matrix(lv1_labels, lv1_predictions)

        plt.matshow(lv1_conf_mx, cmap=plt.cm.gray)
        plt.savefig(os.path.join(self.config.checkpoints_dir, "lv1_conf_mx.png"))
        

        lv1_row_sums = lv1_conf_mx.sum(axis=1, keepdims=True)
        lv1_norm_conf_mx = lv1_conf_mx / lv1_row_sums
        np.fill_diagonal(lv1_norm_conf_mx, 0)
        plt.matshow(lv1_norm_conf_mx, cmap=plt.cm.gray)
        plt.savefig(os.path.join(self.config.checkpoints_dir, "lv1_norm_conf_mx.png"))

        # wandb_logger.log_image(key="confusion matrix", images=[lv1_conf_mx, lv1_norm_conf_mx])

    def predict_step(self, batch, batch_idx):
        self.eval()

        video_feature = batch['frame_features']
        input_ids = batch['id']
        segment_ids = batch['segment_ids']
        attention_mask = batch['mask']
        video_mask = batch['frame_mask']

        with torch.no_grad():
            logits = self.forward(
            input_ids = input_ids,
            input_masks = attention_mask,
            segment_ids = segment_ids, 
            visual_feats = video_feature, 
            visual_attention_mask = video_mask
        )
            
            preds = torch.argmax(logits, dim=1)
            scores = F.softmax(logits, dim=1)

            res = {
                'preds': preds,
                'scores': scores
            }

        return res

    def _step(self, batch, step):
        video_feature = batch['frame_features']
        input_ids = batch['id']
        segment_ids = batch['segment_ids']
        attention_mask = batch['mask']
        video_mask = batch['frame_mask']
        target = batch['target']

        logits = self.forward(
            input_ids = input_ids,
            input_masks = attention_mask,
            segment_ids = segment_ids, 
            visual_feats = video_feature, 
            visual_attention_mask = video_mask
        )


        loss = self.focal_loss_with_smoothing(logits, target)

        with torch.no_grad():
            pred = torch.argmax(logits, dim=1)
            accuracy = (target == pred).float().sum() / pred.shape[0]

        log_dict = {
                f"{step}_acc": accuracy,
                f"{step}_cls_loss": loss,
            }
        self.log_dict(log_dict, prog_bar=True)

        return {"loss": loss, "pred": pred, "target": target, "logits": logits}


def main(
    mode
):

    datamodule = LitDataModule(
        config
    )

    datamodule.setup()
    len_train_dl = len(datamodule.train_dataloader())

    config.len_train_dl = len_train_dl // config.n_gpus

    module = LitModule(config)
    
    if not os.path.exists(config.checkpoints_dir):
        os.mkdir(config.checkpoints_dir)

    model_checkpoint = ModelCheckpoint(
        config.checkpoints_dir,
        filename="best_pth",
        monitor="mean_f1",
        mode='max'
    )

   
    lr_monitor = LearningRateMonitor(
        logging_interval='step',
        log_momentum=True    
    )
    # wandb_logger.watch(module)


    DEBUG = config.DEBUG

    trainer = pl.Trainer(
            accumulate_grad_batches=1,
            auto_lr_find=False,
            auto_scale_batch_size=False,
            benchmark=True,
            callbacks=[
                model_checkpoint,
                lr_monitor
            ],
            deterministic=True,
            fast_dev_run=False,
            gpus=config.n_gpus if mode == "train" else 1,
            accelerator="ddp",
            max_epochs=1 if DEBUG else config.epochs,
            precision=16,
            stochastic_weight_avg=False,
            limit_train_batches=0.01 if DEBUG else 1.0,
            limit_val_batches=0.01 if DEBUG else 1.0,
            # logger=wandb_logger,
            val_check_interval=10 if DEBUG else 1000
        )

    if mode == 'tune':
        trainer.tune(module, datamodule=datamodule)
    elif mode == 'train':
        trainer.fit(module, datamodule=datamodule)
    elif mode == 'evaluate':
        module.load_state_dict(torch.load(os.path.join(config.checkpoints_dir, "best_pth.ckpt"))['state_dict'], strict=True)
        trainer.validate(module, datamodule=datamodule)
    elif mode == 'inference':
        module.load_state_dict(torch.load(os.path.join(config.checkpoints_dir, "best_pth.ckpt"))['state_dict'], strict=True)
        res = trainer.predict(
            module, 
            dataloaders = datamodule.test_dataloader()
        )
        predictions = torch.cat([_['preds'] for _ in res], axis=0)
        scores = torch.cat([_['scores'] for _ in res], axis=0)
        pred = [lv2id_to_category_id(p) for p in predictions.cpu().numpy()]
        np.save(os.path.join(config.checkpoints_dir, f"test_b_fold{config.fold}.npy"), scores.cpu().numpy())


        test_a_df = pd.read_csv(config.test_csv)
        test_a_df = test_a_df.drop(columns=["Unnamed: 0", "title", "asr", "ocr"])
        test_a_df['pred'] = pred
        test_a_df = test_a_df.set_index('id')
        print(test_a_df)
        test_a_df.to_csv(os.path.join(config.checkpoints_dir,f'result_test_b_fold{config.fold}.csv'), header=None)

if __name__ == "__main__":
    main(
        mode = config.mode
    )