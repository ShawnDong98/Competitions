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
from data import ImageCaptionDataset, CNTokenizer, make_train_val_split_dataframe, HLBCaptionInferenceDataset
from utils import AvgMeter, load_checkpoint, save_checkpoint, set_seed, MyScheduler

set_seed(42)

import time
import evaluation
import logging
import json

logger = logging.getLogger(__name__)

@torch.no_grad()
def val_epoch(
    model1,
    model2,
    model3,
    val_loader,
    tokenizer,
    device
):
    model1.eval()
    model2.eval()
    model3.eval()
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
        
        loss1, logits1 = model1(
            batch['image'], 
            batch['input_ids'],
            mode = "caption",
            train_image_model = True
        )
        loss2, logits2 = model2(
            batch['image'], 
            batch['input_ids'],
            mode = "caption",
            train_image_model = True
        )
        loss3, logits3 = model3(
            batch['image'], 
            batch['input_ids'],
            mode = "caption",
            train_image_model = True
        )

        loss = (loss1 + loss2 + loss3) / 3
        logits = 0.5 * logits1 + 0.3 * logits2 + 0.2 * logits3
      
        cap_gens = [' '.join(list(filter(lambda x: (x != '[CLS]' and x != '[SEP]' and x != '[PAD]'), tokenizer.decode(cap.cpu().numpy())))) for cap in logits.argmax(-1)]
        logger.debug(cap_gens[0])
        logger.debug(tokens[0])

        for i, (gen_i, gt_i) in enumerate(zip(cap_gens, tokens)):
            gens['%s_%s' % (it, i)] = [gen_i, ]
            gts['%s_%s' % (it, i)] = [' '.join(gt_i), ]
        

        count = batch['input_ids'].shape[0]
        val_loss_meter.update(loss.mean().item(), count)


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

    print('val_cider: %s' % scores['CIDEr'])
    print('val_bleu1: %s' % scores['BLEU'][0])
    print('val_bleu4: %s' % scores['BLEU'][3])
    print('val_meteor: %s' % scores['METEOR'])
    print('val_rouge: %s' % scores['ROUGE'])

    return val_loss_meter, scores

def ensemble_cap(model1, model2, model3, image):
    text = torch.tensor([2]).unsqueeze(0).to(image.device)
    image_embedding1 = model1.ImageProjectionHead(model1.ImageEncoder(image))
    image_embedding2 = model2.ImageProjectionHead(model2.ImageEncoder(image))
    image_embedding3 = model3.ImageProjectionHead(model3.ImageEncoder(image))
    

    for step in range(30):
        N, T = text.shape
        word_embedding1 = model1.TextEncoder.word_embedding(text)
        pos_embedding1 = model1.TextEncoder.pos_embedding[:, :T, :]
        text_embedding1 = model1.TextEncoder.embd_drop(word_embedding1 + pos_embedding1)
        embedding1 = torch.cat((image_embedding1.unsqueeze(1),text_embedding1), dim=1)

        last_hidden1 = model1.TextEncoder.blocks(embedding1)
        logits1 = model1.TextEncoder.head(model1.TextEncoder.ln_f(last_hidden1))

        word_embedding2 = model2.TextEncoder.word_embedding(text)
        pos_embedding2 = model2.TextEncoder.pos_embedding[:, :T, :]
        text_embedding2 = model2.TextEncoder.embd_drop(word_embedding2 + pos_embedding2)
        embedding2 = torch.cat((image_embedding2.unsqueeze(1),text_embedding2), dim=1)

        last_hidden2 = model2.TextEncoder.blocks(embedding2)
        logits2 = model2.TextEncoder.head(model2.TextEncoder.ln_f(last_hidden2))

        word_embedding3 = model3.TextEncoder.word_embedding(text)
        pos_embedding3 = model3.TextEncoder.pos_embedding[:, :T, :]
        text_embedding3 = model3.TextEncoder.embd_drop(word_embedding3 + pos_embedding3)
        embedding3 = torch.cat((image_embedding3.unsqueeze(1),text_embedding3), dim=1)

        last_hidden3 = model3.TextEncoder.blocks(embedding3)
        logits3 = model3.TextEncoder.head(model3.TextEncoder.ln_f(last_hidden3))

        logits = 0.5 * logits1 + 0.3 * logits2 + 0.2 * logits3

        token = logits.argmax(-1)[:, -1]
        text = torch.cat((text, token.unsqueeze(0)), dim=1)
        # 如果新生成的token为 [SEP]
        if token == 3:
            return text.cpu().numpy().squeeze()
    return text.cpu().numpy().squeeze()

@torch.no_grad()
def inference(
    model1,
    model2,
    model3,
    loader,
    tokenizer,
    device,
    save_file = "res_ensemble.json"
):
    model1.eval()
    model2.eval()
    model3.eval()
    res = {}
    all_start = time.time()
    for batch in tqdm(loader):
        start = time.time()
        batch['image'] = batch['image'].to(device)

        caption = tokenizer.decode(ensemble_cap(model1, model2, model3, batch['image']))


        caption = ("".join(caption)).replace('[CLS]', '').replace('[SEP]', '')
        cost_time = time.time() - start
        print(f"cost time: {cost_time:4f}s\n")

        res[batch['filename'][0]] = [caption]

        print(f"{batch['filename']}: {caption}\n")

    all_cost_time = time.time() - all_start
    print(f"Average time: {all_cost_time / len(loader):.4f}\n")


    with open(save_file, "w", encoding='utf-8') as f:
        json.dump(res, f, ensure_ascii=False)

def main_inf(args):
    CFG.ensemble_model1 = args.model1
    CFG.ensemble_model2 = args.model2
    CFG.ensemble_model3 = args.model3

    tokenizer = CNTokenizer(
        max_length = CFG.block_size
    )
    tokenizer.load_vocab(CFG.CN_vocab_file)

    model1 = CLIP(
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

    model2 = CLIP(
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

    model3 = CLIP(
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

    model1.load_state_dict(torch.load(CFG.ensemble_model1)['model'])

    model2.load_state_dict(torch.load(CFG.ensemble_model2)['model'])

    model3.load_state_dict(torch.load(CFG.ensemble_model3)['model'])

    config = resolve_data_config({}, model=model1)
    transform = create_transform(**config)


    device = "cpu"
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model1 = nn.DataParallel(model1).to(device)
        model2 = nn.DataParallel(model2).to(device)
        model3 = nn.DataParallel(model3).to(device)

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


    save_file = "res_ensemble.json"

    inference(
        model1 = model1.module,
        model2 = model2.module,
        model3 = model3.module,
        loader = testloader,
        tokenizer = tokenizer,
        device = device,
        save_file = save_file,
    )



def main():
    tokenizer = CNTokenizer(
        max_length = CFG.block_size
    )
    tokenizer.load_vocab(CFG.CN_vocab_file)

    model1 = CLIP(
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

    model2 = CLIP(
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

    model3 = CLIP(
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

    model1.load_state_dict(torch.load(CFG.ensemble_model1)['model'])

    model2.load_state_dict(torch.load(CFG.ensemble_model2)['model'])

    model3.load_state_dict(torch.load(CFG.ensemble_model3)['model'])

    config = resolve_data_config({}, model=model1)
    transform = create_transform(**config)

    val_df = pd.read_csv(CFG.val_AiO_filepath)


    valset = ImageCaptionDataset(
        root_dir = CFG.val_AiO_root, 
        filenames = val_df['images'].tolist()[:100] if CFG.Debug else val_df['images'].tolist(), 
        captions = val_df['captions'].tolist()[:100] if CFG.Debug else val_df['captions'].tolist(), 
        tokenizer = tokenizer, 
        max_length = CFG.block_size,
        trans = transform
    )

    device = "cpu"
    if torch.cuda.is_available():
        device = torch.cuda.current_device()
        model1 = nn.DataParallel(model1).to(device)
        model2 = nn.DataParallel(model2).to(device)
        model3 = nn.DataParallel(model3).to(device)


    val_loader = DataLoader(
        valset,
        batch_size = CFG.batch_size,
        num_workers = CFG.num_workers,
        pin_memory = True,
        shuffle = False,
        drop_last = False,
    )

        
    val_loss, scores = val_epoch(
        model1 = model1,
        model2 = model2,
        model3 = model3,
        val_loader = val_loader,
        tokenizer = tokenizer,
        device = device,
    )



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='ensemble')
    parser.add_argument('--model1', type=str, default="./checkpoints/AiO_cap_best_model_0923.pth.tar")
    parser.add_argument('--model2', type=str, default="./checkpoints/AiO_cap_best_model_0920.pth.tar")
    parser.add_argument('--model3', type=str, default="./checkpoints/AiO_cap_scst_best_model_0913.pth.tar")

    args = parser.parse_args()

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s \
        -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    main_inf(args)
    # main()



    




