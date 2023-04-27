import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

import os
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm
import pandas as pd

from models.clip import CLIP
from data import ImageCaptionDataset, CNTokenizer
from config.cfg import CFG

@torch.no_grad()
def get_image_embeddings(
    model,
    loader,
    device
):
    image_embeddings = []
    for batch in tqdm(loader):
        image = batch['image'].to(device)
        image_features = model.ImageEncoder(image)
        image_embedding = model.ImageProjectionHead(image_features)
        image_embeddings.append(image_embedding)

    return torch.cat(image_embeddings)

@torch.no_grad()
def find_matches(
    model,
    tokenizer,
    image_embeddings,
    query,
    image_filenames,
    device,
    n=9,
):
    print(tokenizer(query))
    encode_query = torch.tensor(tokenizer(query, return_tokens=False)).unsqueeze(0).to(device)

    text_feature = model.TextEncoder.get_trg_idx_hidden(encode_query)
    text_embedding = model.TextProjectionHead(text_feature)

    image_embeddings_n = F.normalize(image_embeddings, p=2, dim=-1)
    text_embedding_n = F.normalize(text_embedding, p=2, dim=-1)
    dot_similarity = text_embedding_n @ image_embeddings_n.T

    values, indices = torch.topk(dot_similarity.squeeze(0), n * 5)
    matches = [image_filenames[idx] for idx in indices[::5]]

    _, axes = plt.subplots(3, 3, figsize=(10, 10))
    for match, ax in zip(matches, axes.flatten()):
        image = Image.open(os.path.join(CFG.val_AiO_root, match)).convert("RGB")
        ax.imshow(image)
        ax.axis('off')

    plt.show()
    plt.savefig("test.png")


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

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
    ).to(device)

    model.load_state_dict(torch.load(CFG.clip_save_checkpoint_path)['model'])


    valid_df = pd.read_csv(CFG.val_flickr8kcn_filepath)

    transform = transforms.Compose(
        [   
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    valset = ImageCaptionDataset(
        root_dir = CFG.val_flickr8kcn_root,
        filenames = valid_df['images'].tolist()[:5000] if CFG.Debug else valid_df['images'].tolist(),
        captions = valid_df['captions'].tolist()[:5000] if CFG.Debug else valid_df['captions'].tolist(),
        tokenizer = tokenizer,
        max_length = CFG.block_size,
        trans =  transform
    )

    val_loader = DataLoader(
        valset,
        batch_size = CFG.batch_size,
        num_workers = CFG.num_workers,
        shuffle = False,
        drop_last = False,
        pin_memory = True
    )

    image_embeddings = get_image_embeddings(model, val_loader, device)

    find_matches( 
        model = model,
        tokenizer = tokenizer,
        image_embeddings = image_embeddings,
        query = "一只狗在草地上奔跑。",
        image_filenames = valid_df['images'].tolist(),
        device = device,
        n=9
    )

  
