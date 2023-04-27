import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import CFG
from models import CLIP
from data import ImageCaptionDataset, CNTokenizer, make_train_val_split_dataframe
from utils import AvgMeter, load_checkpoint, save_checkpoint, set_seed, MyScheduler

import os
import time
import multiprocessing

import evaluation
from evaluation import PTBTokenizer, Cider

import logging

logger = logging.getLogger(__name__)

def train_xe(
    model,
    train_loader,
    optimizer,
    writer,
    epoch,
    device
):
    model.train()
    train_loss_meter = AvgMeter(name = "train_loss_meter")
    train_step = 0
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        tokens, filename = batch['tokens'], batch['filename']
        batch = {
            key: value.to(device) 
            for key, value in batch.items() if (key != "filename" and key != "tokens")
        }
        
        loss, logits = model(
            batch['image'], 
            batch['input_ids'],
            mode = "caption",
            train_image_model = True
        )

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        count = batch['input_ids'].shape[0]
        train_loss_meter.update(loss.mean().item(), count)

        writer.add_scalar("Training loss: ", loss.mean().item(), global_step=(epoch * len(train_loader) + train_step))

        tqdm_object.set_postfix(train_loss=train_loss_meter.avg)

        train_step += 1

    return train_loss_meter


@torch.no_grad()
def val_epoch(
    model,
    val_loader,
    tokenizer,
    writer,
    epoch,
    device
):
    model.eval()
    val_loss_meter = AvgMeter(name = "val_loss_meter")
    val_step = 0
    tqdm_object = tqdm(val_loader, total=len(val_loader))

    gts = {}
    gens = {}

    for it, batch in enumerate(tqdm_object):
        tokens, filename = batch['tokens'], batch['filename']
        batch = {
            key: value.to(device) 
            for key, value in batch.items() if (key != "filename" and key != "tokens")
        }
        
        loss, logits = model(
            batch['image'], 
            batch['input_ids'],
            mode = "caption",
            train_image_model = True
        )
        cap_gens = [' '.join(list(filter(lambda x: (x != '[CLS]' and x != '[SEP]' and x != '[PAD]'), tokenizer.decode(cap.cpu().numpy())))) for cap in logits.argmax(-1)]
        logger.debug(cap_gens[0])
        logger.debug(tokens[0])

        for i, (gen_i, gt_i) in enumerate(zip(cap_gens, tokens)):
            gens['%s_%s' % (it, i)] = [gen_i, ]
            gts['%s_%s' % (it, i)] = [' '.join(gt_i), ]
        

        count = batch['input_ids'].shape[0]
        val_loss_meter.update(loss.mean().item(), count)

        writer.add_scalar("Validing loss: ", loss.mean().item(), global_step=(epoch * len(val_loader) + val_step))

        tqdm_object.set_postfix(val_loss=val_loss_meter.avg)

        val_step += 1

    gts = evaluation.PTBTokenizer.tokenize(gts)
    gens = evaluation.PTBTokenizer.tokenize(gens)

    for gt_k, gt_v in gts.items():
        print(gt_k, gt_v)
        break
    for gen_k, gen_v in gens.items():
        print(gen_k, gen_v)
        break

    scores, _ = evaluation.compute_scores(gts, gens)

    print('Epoch %s, val_cider: %s' % (epoch, scores['CIDEr']))
    print('Epoch %s, val_bleu1: %s' % (epoch, scores['BLEU'][0]))
    print('Epoch %s, val_bleu4: %s' % (epoch, scores['BLEU'][3]))
    print('Epoch %s, val_meteor: %s' % (epoch, scores['METEOR']))
    print('Epoch %s, val_rouge: %s' % (epoch, scores['ROUGE']))

    return val_loss_meter, scores





def main(args):
    set_seed(args.seed)
    CFG.Debug = args.debug
    CFG.clip_load_checkpoint_path = args.clip_save_checkpoint_path
    CFG.cap_save_checkpoint_path = args.cap_save_checkpoint_path

    tokenizer = CNTokenizer(
        max_length = CFG.block_size
    )
    tokenizer.load_vocab(CFG.CN_vocab_file)

    model = CLIP(
        image_model = CFG.image_model,
        pretrained = CFG.pretrained,
        trainable = CFG.trainable,
        vocab_size = tokenizer.vocab_size(), 
        embed_dim = CFG.embed_dim,
        n_head = CFG.n_head,
        block_size = CFG.block_size, 
        num_layers = CFG.num_layers,
        attn_pdrop = CFG.attn_pdrop, 
        resid_pdrop = CFG.resid_pdrop,
        embd_pdrop = CFG.embd_pdrop,
        image_features_dim = CFG.image_features_dim, 
        text_features_dim = CFG.text_features_dim,
        proj_dim = CFG.proj_dim, 
    )

    model.load_state_dict(torch.load(CFG.clip_load_checkpoint_path)['model'])

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)



    train_df = pd.read_csv(CFG.train_AiO_filepath)
    val_df = pd.read_csv(CFG.val_AiO_filepath)


    trainset = ImageCaptionDataset(
        root_dir = CFG.train_AiO_root, 
        filenames = train_df['images'].tolist()[:1000] if CFG.Debug else train_df['images'].tolist(), 
        captions = train_df['captions'].tolist()[:1000] if CFG.Debug else train_df['captions'].tolist(), 
        tokenizer = tokenizer, 
        max_length = CFG.block_size,
        trans = transform
    ) 

    valset = ImageCaptionDataset(
        root_dir = CFG.val_AiO_root, 
        filenames = val_df['images'].tolist()[:100] if CFG.Debug else val_df['images'].tolist(), 
        captions = val_df['captions'].tolist()[:100] if CFG.Debug else val_df['captions'].tolist(), 
        tokenizer = tokenizer, 
        max_length = CFG.block_size,
        trans = transform
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=CFG.lr)
    scheduler = MyScheduler(total_epochs=CFG.cap_num_epochs, base_lr=CFG.lr)

    device = "cpu"
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model = nn.DataParallel(model).to(device)


    writer = SummaryWriter(os.path.join('./logs', CFG.cap_save_checkpoint_path.split('/')[-1].split('.')[0]))
    with open(os.path.join('./logs', CFG.cap_save_checkpoint_path.split('/')[-1].split('.')[0]+'.log'), "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)

    best_cider = 0
    for epoch in range(CFG.cap_num_epochs):
        print(f"Epoch: {epoch}")
        train_loader = DataLoader(
            trainset, 
            batch_size = CFG.batch_size,
            num_workers = CFG.num_workers,
            pin_memory = True,
            shuffle = True,
            drop_last = True,
        )
        val_loader = DataLoader(
            valset,
            batch_size = CFG.batch_size,
            num_workers = CFG.num_workers,
            pin_memory = True,
            shuffle = False,
            drop_last = False,
        )

        train_loss = train_xe(
            model = model,
            train_loader = train_loader,
            optimizer = optimizer,
            writer = writer,
            epoch = epoch,
            device = device,
        )
        
        with open(os.path.join('./logs', CFG.cap_save_checkpoint_path.split('/')[-1].split('.')[0]+'.log'), "a") as log_file:
            log_file.write('epoch: %s,  train loss: %s\n' % (epoch, train_loss.avg))

        val_loss, scores = val_epoch(
            model = model,
            val_loader = val_loader,
            tokenizer = tokenizer,
            writer = writer,
            epoch = epoch,
            device = device,
        )

        writer.add_scalar('data/val_cider', scores['CIDEr'], epoch)
        writer.add_scalar('data/val_bleu1', scores['BLEU'][0], epoch)
        writer.add_scalar('data/val_bleu4', scores['BLEU'][3], epoch)
        writer.add_scalar('data/val_meteor', scores['METEOR'], epoch)
        writer.add_scalar('data/val_rouge', scores['ROUGE'], epoch)

        with open(os.path.join('./logs', CFG.cap_save_checkpoint_path.split('/')[-1].split('.')[0]+'.log'), "a") as log_file:
            log_file.write('epoch: %s,  val loss: %s\n' % (epoch, val_loss.avg))
            log_file.write('Epoch %s, val_cider: %s\n' % (epoch, scores['CIDEr']))
            log_file.write('Epoch %s, val_bleu1: %s\n' % (epoch, scores['BLEU'][0]))
            log_file.write('Epoch %s, val_bleu4: %s\n' % (epoch, scores['BLEU'][3]))
            log_file.write('Epoch %s, val_meteor: %s\n' % (epoch, scores['METEOR']))
            log_file.write('Epoch %s, val_rouge: %s\n' % (epoch, scores['ROUGE']))

        for param_group in optimizer.param_groups:
            param_group['lr'] = scheduler()

        if scores['CIDEr'] >= best_cider:
            best_cider = scores['CIDEr']
            checkpoint = {
                "model": model.module.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, CFG.cap_save_checkpoint_path)


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s \
        -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    main()



    




