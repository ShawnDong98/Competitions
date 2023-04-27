import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms

import os
import time
import pandas as pd

import logging
from tqdm import tqdm

from data import ImageCaptionDataset, CNTokenizer
from models import CLIP
from utils import set_seed, load_checkpoint, save_checkpoint, AvgMeter, MyScheduler
from config import CFG



logger = logging.getLogger(__name__)

def train_epoch(
    model,
    train_loader,
    optimizer,
    writer,
    epoch,
    device
):
    model.train()
    train_loss_meter = AvgMeter(name='train_loss')
    train_step = 0
    tqdm_object = tqdm(train_loader, total=len(train_loader))
    for batch in tqdm_object:
        batch = {
            key: value.to(device)
            for key, value in batch.items() if (key != 'filename' and key != 'tokens')
        }

        loss = model(batch['image'], batch['input_ids'])

        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()

        count = batch['image'].shape[0]
        train_loss_meter.update(loss.mean().item(), count)

        writer.add_scalar("training loss: ", loss.mean().item(), global_step=int(epoch * len(train_loader) + train_step))

        tqdm_object.set_postfix(train_loss = train_loss_meter.avg)

        train_step += 1


    return train_loss_meter

@torch.no_grad()
def val_epoch(
    model, 
    val_loader,
    writer,
    epoch,
    device 
):
    model.eval()
    val_loss_meter = AvgMeter(name='val_loss')
    val_step = 0
    tqdm_object = tqdm(val_loader, total=len(val_loader))
    for batch in tqdm_object:
        batch = {
            key: value.to(device)
            for key, value in batch.items() if (key != 'filename' and key != 'tokens')
        }

        loss = model(batch['image'], batch['input_ids'])

        count = batch['image'].shape[0]
        val_loss_meter.update(loss.mean().item(), count)

        writer.add_scalar("val loss: ", loss.mean().item(), global_step=int(epoch * len(val_loader) + val_step))

        tqdm_object.set_postfix(val_loss = val_loss_meter.avg)

        val_step += 1

    return val_loss_meter

def main(args):
    set_seed(args.seed)
    CFG.Debug = args.debug
    CFG.clip_save_checkpoint_path = args.clip_save_checkpoint_path

    tokenizer = CNTokenizer(max_length=CFG.block_size)
    tokenizer.load_vocab(vocab_file=CFG.CN_vocab_file)
    train_trans = transforms.Compose(
        [   
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    val_trans = transforms.Compose(
        [   
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    train_df = pd.read_csv(CFG.train_AiO_filepath)
    val_df = pd.read_csv(CFG.val_AiO_filepath)


    trainset = ImageCaptionDataset(
        root_dir = CFG.train_AiO_root,
        filenames = train_df['images'][:1000].tolist() if CFG.Debug else train_df['images'].tolist(), 
        captions = train_df['captions'][:1000].tolist() if CFG.Debug else train_df['captions'].tolist(), 
        tokenizer = tokenizer,
        max_length = CFG.block_size,
        trans = train_trans
    )

    valset = ImageCaptionDataset(
        root_dir = CFG.val_AiO_root,
        filenames = val_df['images'][:100].tolist() if CFG.Debug else val_df['images'].tolist(), 
        captions = val_df['captions'][:100].tolist() if CFG.Debug else val_df['captions'].tolist(), 
        tokenizer = tokenizer,
        max_length = CFG.block_size,
        trans = val_trans
    )


    writer = SummaryWriter(os.path.join('./logs', CFG.clip_save_checkpoint_path.split('/')[-1].split('.')[0]))
    with open(os.path.join('./logs', CFG.clip_save_checkpoint_path.split('/')[-1].split('.')[0]+'.log'), "a") as log_file:
        now = time.strftime("%c")
        log_file.write('================ Training Loss (%s) ================\n' % now)


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

    optimizer = AdamW(model.parameters(), lr=CFG.lr)

    device = "cpu"
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model = nn.DataParallel(model).to(device)

    best_loss = float("inf")
    for epoch in range(CFG.clip_num_epochs):
        logging.info(f'===>epoch: {epoch}')
        train_loader = DataLoader(
            dataset = trainset, 
            batch_size = CFG.batch_size, 
            shuffle = True,
            num_workers = CFG.num_workers,
            pin_memory = True,
            drop_last = True
        )

        val_loader =  DataLoader(
            dataset = valset, 
            batch_size = CFG.batch_size, 
            shuffle = False,
            num_workers = CFG.num_workers,
            pin_memory = True,
            drop_last = False
        )

        train_loss = train_epoch(
            model = model,
            train_loader = train_loader,
            optimizer = optimizer,
            writer = writer,
            epoch = epoch,
            device = device,
        )

        val_loss = val_epoch(
            model = model,
            val_loader = val_loader,
            writer = writer,
            epoch = epoch,
            device = device,
        )
        with open(os.path.join('./logs', CFG.clip_save_checkpoint_path.split('/')[-1].split('.')[0]+'.log'), "a") as log_file:
            log_file.write('epoch: %s,  train loss: %s\n' % (epoch, train_loss.avg))
            log_file.write('epoch: %s,  val loss: %s\n' % (epoch, val_loss.avg))
       

        if val_loss.avg < best_loss:
            best_loss = val_loss.avg
            checkpoint = {
                'model': model.module.state_dict(), 
                'optimizer': optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, CFG.clip_save_checkpoint_path)


if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s \
        -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    main()