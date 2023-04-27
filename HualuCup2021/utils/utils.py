import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

import requests


class AvgMeter:
    def __init__(self, name="Metric"):
        self.name = name
        self.reset()

    def reset(self):
        self.sum, self.avg, self.counts = [0] * 3

    def update(self, value, count):
        self.counts += count
        self.sum += value * count
        self.avg = self.sum / self.counts

    def __repr__(self):
        text = f"{self.name}: {self.avg}"
        return text


class MyScheduler:
    def __init__(self, total_epochs, base_lr=3e-4,  reduce_factor=0.5):
        self.total_epochs = total_epochs
        self.base_lr_orig = base_lr
        self.first_reduce = int(total_epochs * 0.7)
        self.second_reduce = int(total_epochs * 0.9)
        self.warmup_epochs = int(total_epochs * 0.1)
        self.warmup_begin_lr = base_lr * reduce_factor
        self.reduce_factor = reduce_factor
        self.epochs = 0

    def get_warmup_lr(self):
        increase = (self.base_lr_orig - self.warmup_begin_lr) \
                       * float(self.epochs) / float(self.warmup_epochs)
        return self.warmup_begin_lr + increase

    def __call__(self):
        self.epochs += 1
        if self.epochs < self.warmup_epochs:
            return self.get_warmup_lr()
        if self.epochs <= self.first_reduce:
            return self.base_lr_orig
        if self.epochs <= self.second_reduce:
            return self.base_lr_orig *  self.reduce_factor
        return self.base_lr_orig *  self.reduce_factor *  self.reduce_factor

def load_checkpoint(checkpoint, model, optimizer):
    print("=>Loading Checkpoing,,,")
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def save_checkpoint(checkpoint, save_checkpoint_path):
    print("=>Saving Checkpoing...")
    torch.save(checkpoint, save_checkpoint_path)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def print_sentence_example(batch, model, tokenizer):
    loss, logits = model(batch['input_ids'][:, :-1])
    preds = logits.argmax(-1).cpu().numpy()

    for i in range(5):
        print(batch["tokens"][i].replace("[PAD]", ""))
        print((" ".join(tokenizer.decode(preds[i]))).replace("[PAD]", ""))


@torch.no_grad()
def print_cap_example(model, transform, tokenizer):
    from PIL import Image
    img1 = transform(Image.open("../datasets/MM/COCO2014/print_example/COCO_val2014_000000184613.jpg").convert("RGB"))

    
def download_from_url(url, path):
    """Download file, with logic (from tensor2tensor) for Google Drive"""
    if 'drive.google.com' not in url:
        print('Downloading %s; may take a few minutes' % url)
        r = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'})
        with open(path, "wb") as file:
            file.write(r.content)
        return
    print('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    chunk_size = 16 * 1024
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


    


if __name__ == "__main__":
    logits = torch.tensor(
        [[1, 2, 3, 4],
        [5, 7, 6, 8]]
    )

    v, ix = torch.topk(logits, 2)
    print(v)
    print(v[:, [-1]])

    print(logits < v[:, [-1]])