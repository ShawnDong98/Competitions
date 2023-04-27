import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

import pandas as pd
from tqdm import tqdm
import json

from data import HLBCaptionInferenceDataset, CNTokenizer
from models import CLIP
from config import CFG
import time


@torch.no_grad()
def coco_inference(
    model,
    loader,
    tokenizer,
    device,
    save_file = "coco_test_result.json"
):
    model.eval()
    result = []
    for batch in tqdm(loader):
        result_ = {}
        batch['image'] = batch['image'].to(device)

        caption = model.caption_image(
            batch['image'],
            tokenizer = tokenizer,
            max_length = CFG.block_size,
        )

        caption = (" ".join(caption)).replace('[CLS]', '').replace('[SEP]', '')
        

        result_['image_id'] = int(batch['image_id'][0])
        result_['caption'] = caption
    
        print(f"{batch['filename']}: {caption}")
        result.append(result_)

    with open(save_file, "w", encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)

@torch.no_grad()
def CN_inference(
    model,
    loader,
    tokenizer,
    device,
    save_file = "aic_test_a_result.json"
):
    model.eval()
    result = []
    for batch in tqdm(loader):
        result_ = {}
        batch['image'] = batch['image'].to(device)

        caption = model.caption_image(
            batch['image'],
            tokenizer = tokenizer,
            max_length = CFG.block_size,
        )

        caption = ("".join(caption)).replace('[CLS]', '').replace('[SEP]', '')
        

        result_['image_id'] = int(batch['image_id'][0])
        result_['caption'] = caption
    
        print(f"{batch['filename']}: {caption}")
        result.append(result_)

    with open(save_file, "w", encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False)

@torch.no_grad()
def inference(
    model,
    loader,
    tokenizer,
    device,
    save_file = "flcikr8k_test_result.json"
):
    model.eval()
    res = {}
    all_start = time.time()
    for batch in tqdm(loader):
        start = time.time()
        batch['image'] = batch['image'].to(device)

        caption = model.caption_image(
            batch['image'],
            tokenizer = tokenizer,
            max_length = CFG.block_size,
        )

        caption = ("".join(caption)).replace('[CLS]', '').replace('[SEP]', '')
        cost_time = time.time() - start
        print(f"cost time: {cost_time:4f}s\n")

        res[batch['filename'][0]] = [caption]

        print(f"{batch['filename']}: {caption}\n")

    all_cost_time = time.time() - all_start
    print(f"Average time: {all_cost_time / len(loader):.4f}\n")


    with open(save_file, "w", encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False)



if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    tokenizer = CNTokenizer(
        max_length = CFG.block_size,
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
        proj_dim = CFG.proj_dim
    ).to(device)

    config = resolve_data_config({}, model=model)
    transform = create_transform(**config)

    model.load_state_dict(torch.load(CFG.cap_load_checkpoint_path)['model'])


    testset = HLBCaptionInferenceDataset(
        root_dir = CFG.test_b_hlb0830_root,
        trans = transform
    )


    testloader = DataLoader(
        testset,
        batch_size = 1,
        num_workers = CFG.num_workers,
        pin_memory = True,
        shuffle = False,
        drop_last = False 
    )


    save_file = "res.json"


    inference(
        model = model,
        loader = testloader,
        tokenizer = tokenizer,
        device = device,
        save_file = save_file,
    )

    