import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np

class CFG:
    Debug = False
    pretrained = True
    trainable = True
    block_size = 100

    # Train
    clip_num_epochs = 5
    cap_num_epochs = 20
    batch_size = 160
    num_workers = 4
    lr = 3e-4

    # Model 
    image_model = "res2next50"

    num_layers = 12
    embed_dim = 768
    n_head = 12
    
    attn_pdrop = 0.1
    resid_pdrop = 0.1
    embd_pdrop = 0.1

    image_features_dim = 2048
    text_features_dim = 768
    proj_dim = 768


    # HLB0830
    test_a_hlb0830_root = "../datasets/MM/HLB0830/testA/images"
    test_b_hlb0830_root = "../datasets/MM/HLB0830/testB/images"

    # AllInOne 
    # train_AiO_root = "../datasets/MM/AllInOne/Images"
    # val_AiO_root = "../datasets/MM/AllInOne/Images"

    # train_AiO_filepath = "../datasets/MM/AllInOne/train.csv"
    # val_AiO_filepath = "../datasets/MM/AllInOne/val.csv"
    train_AiO_root = "../datasets/MM/AllInOne0923/Images"
    val_AiO_root = "../datasets/MM/AllInOne0923/Images"

    train_AiO_filepath = "../datasets/MM/AllInOne0923/train.csv"
    val_AiO_filepath = "../datasets/MM/AllInOne0923/val.csv"

    # save
    clip_load_checkpoint_path = "./checkpoints/AiO_clip_best_model_0923.pth.tar"
    cap_load_checkpoint_path = "./checkpoints/AiO_cap_best_model_0923.pth.tar"

    
    clip_save_checkpoint_path = "./checkpoints/AiO_clip_best_model_0923.pth.tar"
    cap_save_checkpoint_path = "./checkpoints/AiO_cap_best_model_0923.pth.tar"

    ensemble_model1 = "./checkpoints/AiO_cap_best_model_0913.pth.tar"
    ensemble_model2 = "./checkpoints/AiO_cap_best_model_0920.pth.tar"
    ensemble_model3 = "./checkpoints/AiO_cap_scst_best_model_0909.pth.tar"
    

    CN_vocab_file = "./config/vocab_cn.json"




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


if __name__ == "__main__":
    logits = torch.tensor(
        [[1, 2, 3, 4],
        [5, 7, 6, 8]]
    )

    v, ix = torch.topk(logits, 2)
    print(v)
    print(v[:, [-1]])

    print(logits < v[:, [-1]])