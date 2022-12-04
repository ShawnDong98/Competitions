from operator import truediv
from utils import seed_everything
seed_everything(
    seed = 3407,
    deterministic = True, 
    use_rank_shift = False
)
import os
import random
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import torch
import torch.nn as nn
from torch.nn import functional as F

import tez
from tez import enums
from tez.utils import AverageMeter
from tez.callbacks import EarlyStopping

try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.distributed.xla_multiprocessing as xmp

    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False

import wandb
from pprint import pprint
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from category_id_map import lv2id_to_lv1id, CATEGORY_ID_LIST, lv2id_to_category_id, category_id_to_lv1id, category_id_to_lv2id

from dataset.wechat_dataset import WeChatDataset
from model.wc_model_multitask import WeChatModel, FGM, PGD
from losses import focal_f1_loss, FocalLoss, FocalLossWithSmoothing
from model.pretrain_mask import MaskLM, MaskVideo

import argparse


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
parser.add_argument("--swa", type=bool, default=False)
parser.add_argument("--DEBUG", type=bool, default=False)

args = parser.parse_known_args()[0]

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

class WCModel(tez.Model):
    def __init__(self, configs):
        super().__init__()
        self.model = WeChatModel(
            model_path = configs.model_path,
            num_classes = configs.num_classes,
            frame_embedding_size = configs.frame_embedding_size,
            vlad_cluster_size = configs.vlad_cluster_size,
            vlad_hidden_size = configs.vlad_hidden_size,
            fc_size = configs.fc_size,
            dropout = configs.dropout,
            se_ratio = configs.se_ratio,
            mean_pooling = configs.mean_pooling,
            max_pooling = configs.max_pooling,
            median_pooling = configs.median_pooling,
            tasks = configs.tasks,
        )
        if configs.swa:
            self.swa_model = torch.optim.swa_utils.AveragedModel(self.model)

        self.focal_loss_with_smoothing = FocalLossWithSmoothing(
            num_classes = configs.num_classes,
            gamma = 2,
            lb_smooth = 0.1,
            ignore_index = None,
            alpha = None
        )

        if 'mlm' in configs.tasks:
            self.lm = MaskLM(
                tokenizer_path=configs.model_path,
                mlm_probability=0.15
            )


        self.config = configs

    def fetch_optimizer(self):
        param_group = [
            {'params': [p for n, p in self.model.named_parameters() if 'bert' in n], 'lr' : self.config.bert_lr},
            {'params': [p for n, p in self.model.named_parameters() if 'bert' not in n], 'lr': self.config.others_lr}
        ]
        optimizer = torch.optim.AdamW(
            param_group,
            weight_decay=self.config.weight_decay,
        )
        return optimizer

    def fetch_scheduler(self):
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr = [self.config.bert_lr, self.config.others_lr],
            steps_per_epoch = self.config.len_train_dl,
            epochs = self.config.epochs,
            pct_start = self.config.one_cycle_pct,
        )

        return scheduler

    def forward(self,
                text_input_ids,
                text_mask,
                video_feature,
                video_mask,
                tasks = ['mlm', 'cls'], 
                step = "train"
    ):
        if step == "val" and self.config.swa: 
            outputs = self.swa_model(
                text_input_ids = text_input_ids,
                text_mask = text_mask,
                video_feature = video_feature, 
                video_mask = video_mask,
                tasks = tasks
            )
        else:
            outputs = self.model(
                text_input_ids = text_input_ids,
                text_mask = text_mask,
                video_feature = video_feature, 
                video_mask = video_mask,
                tasks = tasks
            )


        return outputs

    def model_fn(self,
                text_input_ids,
                text_mask,
                video_feature, 
                video_mask,
                tasks,
                step
    ):
        if self.fp16:
            with torch.cuda.amp.autocast():
                outputs = self(
                    text_input_ids = text_input_ids,
                    text_mask = text_mask,
                    video_feature = video_feature,
                    video_mask = video_mask,
                    tasks = tasks,
                    step = step
                )
        else:
            outputs = self(
                text_input_ids = text_input_ids,
                text_mask = text_mask,
                video_feature = video_feature,
                video_mask = video_mask,
                tasks = tasks,
                step = step
            )
        return outputs


    def _step(self, data, step):
        if step == "train": tasks = self.config.tasks
        else: tasks = ['cls']
        
        video_feature = data['frame_features'].to(self.device)
        text_input_ids = data['id'].to(self.device)
        text_mask = data['mask'].to(self.device)
        video_mask = data['frame_mask'].to(self.device)
        target = data['target'].to(self.device)

        if 'mlm' in tasks:
            input_ids, lm_label = self.lm.torch_mask_tokens(text_input_ids.cpu())
            text_input_ids = input_ids.to(text_input_ids.device)
            lm_label = lm_label[:, 1:].to(text_input_ids.device)
        
        if 'mfm' in tasks:
            vm_input = video_feature
            input_feature, video_label = self.vm.torch_mask_frames(video_feature.cpu(), video_mask.cpu())
            video_feature = input_feature.to(video_feature.device)
            video_label = video_label.to(video_feature.device)

        outputs = self.model_fn(
            text_input_ids = text_input_ids,
            text_mask = text_mask,
            video_feature = video_feature, 
            video_mask = video_mask,
            tasks = tasks,
            step = step
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
            }
        else:
            log_dict = {
                f"{step}_acc": accuracy,
                f"{step}_cls_loss": cls_loss,
            }

        # wandb.log(log_dict)

        return {'pred': pred, "target": target}, loss, log_dict

    def train_one_step(self, data):
        if self.accumulation_steps == 1 and self.batch_index == 0:
            self.zero_grad()
        _, loss, metrics = self._step(data, "train")
        loss = loss / self.accumulation_steps
        if self.fp16:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()
        if self.clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)

        if self.config.adv_mode == 'fgm' or self.config.adv_mode == 'adv_all':
            # print("FGM Attack...")
            fgm = FGM(self.model,
                    self.config.epsilon, 
                    self.config.emb_name, 
            )
            fgm.attack()
            _, loss, metrics = self._step(data, "train")
            adv_loss = loss
            if self.fp16:
                self.scaler.scale(adv_loss).backward()
            else:
                adv_loss.backward()
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
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
                    self.zero_grad()
                else:
                    pgd.restore_grad()
                _, loss, metrics = self._step(data, "train")
                adv_loss = loss
                if self.fp16:
                    self.scaler.scale(adv_loss).backward()
                else:
                    adv_loss.backward()
                if self.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.parameters(), self.clip_grad_norm)
            pgd.restore()

        
        if (self.batch_index + 1) % self.accumulation_steps == 0:
            if self.fp16:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                if self.using_tpu:
                    xm.optimizer_step(self.optimizer, barrier=True)
                else:
                    self.optimizer.step()
            if self.scheduler:
                if self.step_scheduler_after == "batch":
                    # for i, pg in enumerate(self.optimizer.param_groups, start=1):
                    #     wandb.log({f"lr-AdamW/pg{i}" : pg['lr']})
                    #     wandb.log({f"lr-AdamW/pg{i}-momentum" : pg["betas"][0]})
                    if self.step_scheduler_metric is None:
                        self.scheduler.step()
                    else:
                        step_metric = self.name_to_metric(self.step_scheduler_metric)
                        self.scheduler.step(step_metric)
            if self.batch_index > 0:
                self.zero_grad()
        return loss, metrics

    def train_one_epoch(self, data_loader):
        self.train()
        self.model_state = enums.ModelState.TRAIN
        losses = AverageMeter()
        if self.accumulation_steps > 1:
            self.optimizer.zero_grad()
        if self.using_tpu:
            tk0 = data_loader
        else:
            tk0 = tqdm(data_loader, total=len(data_loader))
        for b_idx, data in enumerate(tk0):
            self.batch_index = b_idx
            self.train_state = enums.TrainingState.TRAIN_STEP_START
            loss, metrics = self.train_one_step(data)
            self.train_state = enums.TrainingState.TRAIN_STEP_END
            losses.update(loss.item() * self.accumulation_steps, data_loader.batch_size)
            if b_idx == 0:
                metrics_meter = {k: AverageMeter() for k in metrics}
            monitor = {}
            for m_m in metrics_meter:
                metrics_meter[m_m].update(metrics[m_m], data_loader.batch_size)
                monitor[m_m] = metrics_meter[m_m].avg.item()
            self.current_train_step += 1
            if not self.using_tpu:
                tk0.set_postfix(loss=losses.avg, stage="train", **monitor)
            if self.using_tpu:
                print(f"train step: {self.current_train_step} loss: {losses.avg}")
        if not self.using_tpu:
            tk0.close()
        self.update_metrics(losses=losses, monitor=monitor)

        return losses.avg


    def validate_one_step(self, data):
        outputs, loss, metrics = self._step(data, "val")
        return outputs, loss, metrics

    def validate_one_epoch(self, data_loader):
        self.eval()
        self.model_state = enums.ModelState.VALID
        losses = AverageMeter()
        if self.using_tpu:
            tk0 = data_loader
        else:
            tk0 = tqdm(data_loader, total=len(data_loader))
        outputs_list = []
        for b_idx, data in enumerate(tk0):
            self.train_state = enums.TrainingState.VALID_STEP_START
            with torch.no_grad():
                outputs, loss, metrics = self.validate_one_step(data)
            outputs_list.append(outputs)
            self.train_state = enums.TrainingState.VALID_STEP_END
            losses.update(loss.item(), data_loader.batch_size)
            if b_idx == 0:
                metrics_meter = {k: AverageMeter() for k in metrics}
            monitor = {}
            for m_m in metrics_meter:
                metrics_meter[m_m].update(metrics[m_m], data_loader.batch_size)
                monitor[m_m] = metrics_meter[m_m].avg.item()
            if not self.using_tpu:
                tk0.set_postfix(loss=losses.avg, stage="valid", **monitor)
            self.current_valid_step += 1
        if not self.using_tpu:
            tk0.close()
        self.update_metrics(losses=losses, monitor=monitor)

        preds = torch.cat([x['pred'] for x in outputs_list])
        targets = torch.cat([x['target'] for x in outputs_list])

        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()

        results = evaluate(preds, targets)

        pprint(results)
        # wandb.log(results)
        

        lv1_predictions = [category_id_to_lv1id(lv2id_to_category_id(lv2id)) for lv2id in preds]
        lv1_labels = [category_id_to_lv1id(lv2id_to_category_id(lv2id)) for lv2id in targets]

        lv1_conf_mx = confusion_matrix(lv1_labels, lv1_predictions)

        plt.matshow(lv1_conf_mx, cmap=plt.cm.gray)
        plt.savefig(os.path.join(self.config.checkpoints_dir, "lv1_conf_mx.png"))
        # lv1_conf_mx_image = wandb.Image(lv1_conf_mx, caption=f"lv1_conf_mx")

        

        lv1_row_sums = lv1_conf_mx.sum(axis=1, keepdims=True)
        lv1_norm_conf_mx = lv1_conf_mx / lv1_row_sums
        np.fill_diagonal(lv1_norm_conf_mx, 0)
        plt.matshow(lv1_norm_conf_mx, cmap=plt.cm.gray)
        plt.savefig(os.path.join(self.config.checkpoints_dir, "lv1_norm_conf_mx.png"))

        # lv1_norm_conf_mx_image = wandb.Image(lv1_norm_conf_mx, caption=f"lv1_norm_conf_mx")
        # wandb.log({"confusion matrix" : [lv1_conf_mx_image, lv1_norm_conf_mx_image]})

        return results['mean_f1']

    def fit(
        self,
        train_dataset,
        valid_dataset=None,
        train_sampler=None,
        valid_sampler=None,
        device="cuda",
        n_gpus=1,
        epochs=10,
        train_bs=16,
        valid_bs=16,
        n_jobs=8,
        callbacks=None,
        fp16=False,
        train_collate_fn=None,
        valid_collate_fn=None,
        train_shuffle=True,
        valid_shuffle=False,
        accumulation_steps=1,
        clip_grad_norm=None,
        step_scheduler_after='step'
    ):
        """
        The model fit function. Heavily inspired by tf/keras, this function is the core of Tez and this is the only
        function you need to train your models.

        """
        if device == "tpu":
            if XLA_AVAILABLE is False:
                raise RuntimeError("XLA is not available. Please install pytorch_xla")
            else:
                self.using_tpu = True
                fp16 = False
                device = xm.xla_device()
        self._init_model(
            device=device,
            n_gpus=n_gpus,
            train_dataset=train_dataset,
            valid_dataset=valid_dataset,
            train_sampler=train_sampler,
            valid_sampler=valid_sampler,
            train_bs=train_bs,
            valid_bs=valid_bs,
            n_jobs=n_jobs,
            callbacks=callbacks,
            fp16=fp16,
            train_collate_fn=train_collate_fn,
            valid_collate_fn=valid_collate_fn,
            train_shuffle=train_shuffle,
            valid_shuffle=valid_shuffle,
            accumulation_steps=accumulation_steps,
            clip_grad_norm=clip_grad_norm,
            step_scheduler_after=step_scheduler_after,
        )

        for _ in range(epochs):
            self.train_state = enums.TrainingState.EPOCH_START
            self.train_state = enums.TrainingState.TRAIN_EPOCH_START
            train_loss = self.train_one_epoch(self.train_loader)
            self.train_state = enums.TrainingState.TRAIN_EPOCH_END
            if self.config.swa: 
                self.swa_model.update_parameters(self.model)
            if self.valid_loader:
                self.train_state = enums.TrainingState.VALID_EPOCH_START
                if self.config.swa: 
                    torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model)
                valid_mean_f1 = self.validate_one_epoch(self.valid_loader)
                if self.config.swa:
                    self.save_swa(os.path.join(self.config.checkpoints_dir, f"swa_best_pth.ckpt"), weights_only=False)
                else:
                    self.save(os.path.join(self.config.checkpoints_dir, f"best_pth.ckpt"), weights_only=False)
                self.train_state = enums.TrainingState.VALID_EPOCH_END
            if self.scheduler:
                if self.step_scheduler_after == "epoch":
                    if self.step_scheduler_metric is None:
                        self.scheduler.step()
                    else:
                        step_metric = self.name_to_metric(self.step_scheduler_metric)
                        self.scheduler.step(step_metric)
            self.train_state = enums.TrainingState.EPOCH_END
            if self._model_state.value == "end":
                break
            self.current_epoch += 1
        self.train_state = enums.TrainingState.TRAIN_END


    def save_swa(self, model_path, weights_only=False):
        print("SWA Model Saving...")
        model_state_dict = self.swa_model.state_dict() if not self.n_gpus > 1 else self.module.state_dict()
        if weights_only:
            if self.using_tpu:
                xm.save(model_state_dict, model_path)
            else:
                torch.save(model_state_dict, model_path)
            return
        if self.optimizer is not None:
            opt_state_dict = self.optimizer.state_dict()
        else:
            opt_state_dict = None
        if self.scheduler is not None:
            sch_state_dict = self.scheduler.state_dict()
        else:
            sch_state_dict = None
        model_dict = {}
        model_dict["state_dict"] = model_state_dict
        model_dict["optimizer"] = opt_state_dict
        model_dict["scheduler"] = sch_state_dict
        model_dict["epoch"] = self.current_epoch
        model_dict["fp16"] = self.fp16
        if self.using_tpu:
            xm.save(model_dict, model_path)
        else:
            torch.save(model_dict, model_path)


def main():
    label = pd.read_csv(args.label_csv)

    dataset = WeChatDataset(
        model_path = args.model_path,
        data_dir = args.train_dir,
        get_title=True,
        get_asr=True,
        get_ocr=True,
        get_frame=True,
        label_json=args.train_json, 
        get_vid=True,
        get_category=True,
        text_maxlen=args.text_maxlen, 
        frame_maxlen=args.frame_maxlen,
    )
    
    train_ix = label[label['fold']!=args.fold].index 
    val_ix = label[label['fold']==args.fold].index
    train_dataset = torch.utils.data.Subset(dataset, train_ix if not args.DEBUG else train_ix[:100])
    val_dataset = torch.utils.data.Subset(dataset, val_ix if not args.DEBUG else val_ix[:100])

    args.len_train_dl = len(train_dataset) // args.batch_size + 1

    test_dataset = WeChatDataset(
        model_path = args.model_path,
        data_dir = args.test_dir,
        get_title=True,
        get_asr=True,
        get_ocr=True,
        get_frame=True,
        label_json=args.test_json, 
        get_vid=True,
        get_category=False,
        text_maxlen=args.text_maxlen, 
        frame_maxlen=args.frame_maxlen,
    )

    module = WCModel(args)
    module.load_state_dict(torch.load(args.pretrained_path)['state_dict'], strict=False)

    # wandb.watch(module, log_freq=100)

    
    if not os.path.exists(args.checkpoints_dir):
        os.mkdir(args.checkpoints_dir)

    module.fit(
        train_dataset = train_dataset,
        valid_dataset = val_dataset,
        train_sampler = None,
        valid_sampler = None,
        device="cuda",
        n_gpus=1,
        epochs=args.epochs,
        train_bs=args.batch_size,
        valid_bs=args.batch_size * 4,
        n_jobs=4,
        callbacks=None,
        fp16=True,
        train_collate_fn=None,
        valid_collate_fn=None,
        train_shuffle=True,
        valid_shuffle=False,
        accumulation_steps=1,
        clip_grad_norm=None,
        step_scheduler_after="batch",
    )


if __name__ == "__main__":
    main()