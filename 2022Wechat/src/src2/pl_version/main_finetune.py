import pytorch_lightning as pl
pl.seed_everything(42)

import os
import collections
from typing import Callable
from typing import Dict
from typing import Optional
from typing import Tuple
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler


import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from dataset.wechat_dataset import WeChatDataset
from model.wc_model_multitask import WeChatModel, FGM, PGD
from model.pretrain_mask import MaskLM, MaskVideo
from losses import focal_f1_loss, FocalLoss, FocalLossWithSmoothing

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from category_id_map import lv2id_to_lv1id, CATEGORY_ID_LIST, lv2id_to_category_id, category_id_to_lv1id, LV1_NAME, category_id_to_lv2id

import wandb
from pytorch_lightning.loggers import WandbLogger
# wandb_logger = WandbLogger(project="vlbert_WeChat")


import pytorch_lightning as pl
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.callbacks import LearningRateMonitor, StochasticWeightAveraging
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from utils import evaluate


parser = argparse.ArgumentParser(description="VLBERT WeChat")
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
parser.add_argument("--model_path", type=str, default="/home/2022Wechat_COMMIT/src/input/pretrain-model/chinese-macbert-base")
parser.add_argument("--pretrained_path", type=str, default="/home/2022Wechat_COMMIT/src/pl_version/checkpoints_pretrained/best_pth.ckpt")
parser.add_argument("--num_classes", type=int, default=len(CATEGORY_ID_LIST))
parser.add_argument("--frame_embedding_size", type=int, default=768)
parser.add_argument("--vlad_cluster_size", type=int, default=64)
parser.add_argument("--vlad_hidden_size", type=int, default=1024)
parser.add_argument("--fc_size", type=int, default=768)
parser.add_argument("--se_ratio", type=int, default=8)
parser.add_argument("--dropout", type=float, default=0.2)
parser.add_argument("--mean_pooling", type=bool, default=True)
parser.add_argument("--max_pooling", type=bool, default=True)
parser.add_argument("--median_pooling", type=bool, default=True)
parser.add_argument("--adv_mode", type=str, default="fgm")
parser.add_argument("--epsilon", type=int, default=1)
parser.add_argument("--emb_name", type=str, default="word_embeddings.")
parser.add_argument("--alpha", type=float, default=0.3)
parser.add_argument("--adv_k", type=int, default=3)
parser.add_argument("--bert_lr", type=float, default=1e-5)
parser.add_argument("--others_lr", type=float, default=5e-5)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--one_cycle_pct", type=float, default=0.25)
parser.add_argument("--epochs", type=int, default=6)
parser.add_argument("--n_gpus", type=int, default=1)
parser.add_argument("--mode", type=str, default="train")
parser.add_argument("--tasks", type=list, default=["cls", 'mlm'])
parser.add_argument("--checkpoints_dir", type=str,default="checkpoints_finetune")
parser.add_argument("--DEBUG", type=bool, default=False)

args = parser.parse_known_args()[0]


class LitDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.label = pd.read_csv(self.args.label_csv)
        print("Loaded csv file ...")

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            dataset = WeChatDataset(
                model_path = self.args.model_path,
                data_dir = self.args.train_dir,
                get_title=True,
                get_asr=True,
                get_ocr=True,
                get_frame=True,
                label_json=self.args.train_json, 
                get_vid=True,
                get_category=True,
                text_maxlen=self.args.text_maxlen, 
                frame_maxlen=self.args.frame_maxlen,
            )
            
            train_ix = self.label[self.label['fold']!=self.args.fold].index 
            val_ix = self.label[self.label['fold']==self.args.fold].index
            self.train_dataset = torch.utils.data.Subset(dataset, train_ix)
            self.val_dataset = torch.utils.data.Subset(dataset, val_ix)

        if stage == "test" or stage is None:
            self.test_dataset = WeChatDataset(
                model_path = self.args.model_path,
                data_dir = self.args.test_dir,
                get_title=True,
                get_asr=True,
                get_ocr=True,
                get_frame=True,
                label_json=self.args.test_json, 
                get_vid=True,
                get_category=False,
                text_maxlen=self.args.text_maxlen, 
                frame_maxlen=self.args.frame_maxlen,
            )


    def train_dataloader(self) -> DataLoader:
        return self._dataloader(self.train_dataset, train=True)

    def val_dataloader(self) -> DataLoader:
        return self._dataloader(self.val_dataset, train=False)

    def test_dataloader(self) -> DataLoader:
        return self._dataloader(self.test_dataset, train=False)

    def _dataloader(self, dataset: WeChatDataset, train: bool = False) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.args.batch_size if train else self.args.batch_size * 16,
            shuffle=train,
            num_workers=self.args.num_workers,
            pin_memory=True,
            drop_last=train,
        )


class LitModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        print("adv_mode: ", self.args.adv_mode)
        self.model = WeChatModel(
            model_path = self.args.model_path,
            num_classes = self.args.num_classes,
            frame_embedding_size = self.args.frame_embedding_size,
            vlad_cluster_size = self.args.vlad_cluster_size,
            vlad_hidden_size = self.args.vlad_hidden_size,
            fc_size = self.args.fc_size,
            dropout = self.args.dropout,
            se_ratio = self.args.se_ratio,
            mean_pooling = self.args.mean_pooling,
            max_pooling = self.args.max_pooling,
            median_pooling = self.args.median_pooling,
            tasks = self.args.tasks,
        ).to("cuda")

        self.best_mean_f1 = 0 

        self.focal_loss_with_smoothing = FocalLossWithSmoothing(
            num_classes = self.args.num_classes,
            gamma = 2,
            lb_smooth = 0.1,
            ignore_index = None,
            alpha = None
        )
        self.automatic_optimization = False

       
        if 'mlm' in self.args.tasks:
            self.lm = MaskLM(
                tokenizer_path=self.args.model_path,
                mlm_probability=0.15
            )

        if 'mfm' in self.args.tasks:
            self.vm = MaskVideo(mfm_probability=0.15)
            

    def forward(self,
                text_input_ids,
                text_mask,
                video_feature,
                video_mask,
                tasks = ['mlm', 'cls']
    ):
        outputs = self.model(
            text_input_ids = text_input_ids,
            text_mask = text_mask,
            video_feature = video_feature, 
            video_mask = video_mask,
            tasks = tasks
        )

        return outputs

    def configure_optimizers(self):
        param_group = [
            {'params': [p for n, p in self.model.named_parameters() if 'bert' in n], 'lr' : self.args.bert_lr},
            {'params': [p for n, p in self.model.named_parameters() if 'bert' not in n], 'lr': self.args.others_lr}
        ]
        optimizer = torch.optim.AdamW(
            param_group,
            weight_decay=self.args.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr = [self.args.bert_lr, self.args.others_lr],
            steps_per_epoch = self.args.len_train_dl,
            epochs = self.args.epochs,
            pct_start = self.args.one_cycle_pct,
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
        
        if self.args.adv_mode == 'fgm' or self.args.adv_mode == 'adv_all':
            # print("FGM Attack...")
            fgm = FGM(self.model,
                    self.args.epsilon, 
                    self.args.emb_name, 
            )
            fgm.attack()
            adv_ret = self._step(batch, "train")
            adv_loss = adv_ret['loss']
            self.manual_backward(adv_loss)
            fgm.restore()

        elif self.args.adv_mode == 'pgd' or self.args.adv_mode == 'adv_all':
            # print("PGD Attack...")
            pgd = PGD(self.model, 
                    self.args.epsilon, 
                    self.args.emb_name, 
                    self.args.alpha
            )
            pgd.backup_grad()
            for t in range(self.args.adv_k):
                pgd.attack(is_first_attack=(t == 0))
                if t != self.args.adv_k - 1:
                    opt.zero_grad()
                else:
                    pgd.restore_grad()
                adv_ret = self._step(batch, "train")
                adv_loss = adv_ret['loss']
                self.manual_backward(adv_loss)
            pgd.restore()
            

        opt.step()
        scheduler.step()

        self.trainer.train_loop.running_loss.append(loss)

    
    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def validation_epoch_end(self, validation_step_outputs):
        preds = torch.cat([x['pred'] for x in validation_step_outputs])
        targets = torch.cat([x['target'] for x in validation_step_outputs])
        logits = torch.cat([x['logits'] for x in validation_step_outputs])

        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
        logits = logits.cpu().numpy()

        np.save(os.path.join(self.args.checkpoints_dir,f"test_b_logits_fold{self.args.fold}.npy"), logits)
        np.save(os.path.join(self.args.checkpoints_dir,f"test_b_target_fold{self.args.fold}.npy"), targets)

        results = evaluate(preds, targets)

        self.log_dict(results, prog_bar=True)

        lv1_predictions = [category_id_to_lv1id(lv2id_to_category_id(lv2id)) for lv2id in preds]
        lv1_labels = [category_id_to_lv1id(lv2id_to_category_id(lv2id)) for lv2id in targets]

        lv1_conf_mx = confusion_matrix(lv1_labels, lv1_predictions)

        plt.matshow(lv1_conf_mx, cmap=plt.cm.gray)
        plt.savefig(os.path.join(self.args.checkpoints_dir, "lv1_conf_mx.png"))
        

        lv1_row_sums = lv1_conf_mx.sum(axis=1, keepdims=True)
        lv1_norm_conf_mx = lv1_conf_mx / lv1_row_sums
        np.fill_diagonal(lv1_norm_conf_mx, 0)
        plt.matshow(lv1_norm_conf_mx, cmap=plt.cm.gray)
        plt.savefig(os.path.join(self.args.checkpoints_dir, "lv1_norm_conf_mx.png"))


    def predict_step(self, batch, batch_idx):
        tasks = ['cls']
        self.eval()

        video_feature = batch['frame_features']
        text_input_ids = batch['id']
        text_mask = batch['mask']
        video_mask = batch['frame_mask']

        with torch.no_grad():
            outputs = self.forward(
                text_input_ids = text_input_ids,
                text_mask = text_mask,
                video_feature = video_feature, 
                video_mask = video_mask,
                tasks = tasks
            )
            logits = outputs['cls']
            
            
            preds = torch.argmax(logits, dim=1)
            scores = F.softmax(logits, dim=1)

            res = {
                'preds': preds,
                'scores': scores
            }

        return res
        

    def _step(self, batch, step):
        if step == "train": tasks = self.args.tasks
        else: tasks = ['cls']
        
        video_feature = batch['frame_features']
        text_input_ids = batch['id']
        text_mask = batch['mask']
        video_mask = batch['frame_mask']
        target = batch['target']

        if 'mlm' in tasks:
            input_ids, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu())
            text_input_ids = input_ids.to(text_input_ids.device)
            lm_label = lm_label[:, 1:].to(text_input_ids.device)
        
        if 'mfm' in tasks:
            vm_input = video_feature
            input_feature, video_label = self.vm.torch_mask_frames(video_feature.cpu(), video_mask.cpu())
            video_feature = input_feature.to(video_feature.device)
            video_label = video_label.to(video_feature.device)

        outputs = self.forward(
            text_input_ids = text_input_ids,
            text_mask = text_mask,
            video_feature = video_feature, 
            video_mask = video_mask,
            tasks = tasks
        )
        
        loss = 0
        cls_loss = self.focal_loss_with_smoothing(outputs['cls'], target)
        loss += cls_loss
        with torch.no_grad():
            pred = torch.argmax(outputs['cls'], dim=1)
            accuracy = (target == pred).float().sum() / pred.shape[0]
        
        if 'mlm' in tasks:
            masked_lm_loss = nn.CrossEntropyLoss()(outputs['mlm'], lm_label.contiguous().view(-1))
            loss += 0.2 * masked_lm_loss
        
        if 'mfm' in tasks:
            masked_vm_loss = self.model.calculate_mfm_loss(
                outputs['mfm'], 
                vm_input, 
                video_mask, 
                video_label, 
                normalize=False
            )
            loss += 0.1 * masked_vm_loss

        

        if step == "train":
            log_dict = {
                f"{step}_acc": accuracy,
                f"{step}_cls_loss": cls_loss,
                f"{step}_mlm_loss": masked_lm_loss,
                # f"{step}_mfm_loss": masked_vm_loss,
            }
        else:
            log_dict = {
                f"{step}_acc": accuracy,
                f"{step}_cls_loss": cls_loss,
            }
        self.log_dict(log_dict, prog_bar=True)

        return {"loss": loss, "pred": pred, "target": target, "logits": outputs['cls']}


def main():
    datamodule = LitDataModule(args)
    datamodule.setup()
    len_train_dl = len(datamodule.train_dataloader())
    args.len_train_dl = len_train_dl // args.n_gpus if args.mode == 'train' else len_train_dl
    if not os.path.exists(args.checkpoints_dir): 
        os.mkdir(args.checkpoints_dir)

    module = LitModule(args)

    model_checkpoint = ModelCheckpoint(
            args.checkpoints_dir,
            filename=f"best_pth",
            monitor="mean_f1",
            mode='max'
    )
    early_stop_callback = EarlyStopping(monitor="mean_f1", patience=2, mode="max")
    lr_monitor = LearningRateMonitor(
        logging_interval='step',
        log_momentum=True      
    )

    print("args.DEBUG: ", args.DEBUG)
    trainer = pl.Trainer(
            accumulate_grad_batches=1,
            auto_lr_find=True,
            auto_scale_batch_size=False,
            benchmark=True,
            gpus=args.n_gpus if args.mode == "train" else 1,
            strategy="ddp",
            callbacks=[
                model_checkpoint,
                lr_monitor,
                early_stop_callback,
            ],
            deterministic=True,
            fast_dev_run=False,
            max_epochs=1 if args.DEBUG else args.epochs,
            precision=16,
            limit_train_batches=0.01 if args.DEBUG else 1.0,
            limit_val_batches=0.1 if args.DEBUG else 1.0,
            # logger=wandb_logger,
            val_check_interval=10 if args.DEBUG else 1000,
            replace_sampler_ddp=True
        )

    if args.mode == 'tune':
        trainer.tune(module, datamodule=datamodule)
    elif args.mode == 'train':
        module.load_state_dict(torch.load(args.pretrained_path)['state_dict'], strict=False)
        trainer.fit(module, datamodule=datamodule)
    elif args.mode == 'evaluate':
        module.load_state_dict(torch.load(os.path.join(args.checkpoints_dir, "best_pth.ckpt"))['state_dict'], strict=True)
        trainer.validate(module, datamodule=datamodule)

    elif args.mode == 'inference': 
        module.load_state_dict(torch.load(os.path.join(args.checkpoints_dir, "best_pth.ckpt"))['state_dict'], strict=True)
        res = trainer.predict(
            module, 
            dataloaders = datamodule.test_dataloader()
        )
        predictions = torch.cat([_['preds'] for _ in res], axis=0)
        scores = torch.cat([_['scores'] for _ in res], axis=0)
        pred = [lv2id_to_category_id(p) for p in predictions.cpu().numpy()]
        np.save(os.path.join(args.checkpoints_dir, f"test_b_fold{args.fold}.npy"), scores.cpu().numpy())

        test_a_df = pd.read_csv(args.test_csv)
        test_a_df = test_a_df.drop(columns=["Unnamed: 0", "title", "asr", "ocr"])
        test_a_df['pred'] = pred
        test_a_df = test_a_df.set_index('id')
        print(test_a_df)
        test_a_df.to_csv(os.path.join(args.checkpoints_dir,f'result_test_b_fold{args.fold}.csv'), header=None)
    

if __name__ == '__main__':
    main()