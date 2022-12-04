import collections
import os
import sys
import random

from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from pprint import pprint
from pretrain.lxmert_data import InputExample, LXMERTDataset, LXMERTTorchDataset
from lxrt.entry import set_visual_config
from lxrt.tokenization import BertTokenizer
from lxrt.modeling import LXRTPretraining

import argparse


parser = argparse.ArgumentParser(description="LXMERT Pretrain")
parser.add_argument("--tiny", type=bool, default=True)
parser.add_argument('--fast', type=bool, default=True)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--train', type=str, default="labeled,test_b")
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--valid', type=str, default="test_b")
parser.add_argument('--valid_batch_size', type=int, default=32 * 4)
parser.add_argument('--word_mask_rate', type=float, default=0.15)
parser.add_argument('--frame_mask_rate', type=float, default=0.15)
parser.add_argument('--llayers', type=int, default=9)
parser.add_argument('--xlayers', type=int, default=5)
parser.add_argument('--rlayers', type=int, default=5)
parser.add_argument('--task_mask_lm', type=bool, default=True)
parser.add_argument('--task_obj_predict', type=bool, default=True)
parser.add_argument('--task_matched', type=bool, default=True)
parser.add_argument('--task_qa', type=bool, default=False)
parser.add_argument('--visual_losses', type=str, default="feat")
parser.add_argument('--num_answers', type=int, default=2)
parser.add_argument('--from_scratch', type=bool, default=False)
parser.add_argument('--load', type=str, default=None)
parser.add_argument('--load_lxmert', type=str, default=None)
parser.add_argument('--multiGPU', type=bool, default=True)
parser.add_argument('--max_seq_length', type=int, default=300)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument("--model_path", type=str, default="/home/2022Wechat_COMMIT/src/input/pretrain-model/chinese-macbert-base")
parser.add_argument('--output', type=str, default="./checkpoints_pretrain/")
parser.add_argument('--root_path', type=str, default="/home/datasets/2022WeChat/data/annotations")
parser.add_argument('--DEBUG', type=bool, default=False)

config = parser.parse_known_args()[0]

DataTuple = collections.namedtuple("DataTuple", 'dataset torchdset loader')


def get_tuple(splits: str, bs: int, shuffle=False, drop_last=False, topk=-1) -> DataTuple:
    # Decide which QA datasets would be used in pre-training.
    # Options: vqa, gqa, visual7w
    # Note: visual7w is a part of vgqa, we take the name here.

    # Build dataset, data loader, and evaluator.
   
    dset = LXMERTDataset(
        root_path = config.root_path,
        splits = splits)
    
    tset = LXMERTTorchDataset(
        config, 
        dset, 
        topk
    )
   
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=config.num_workers,
        collate_fn=lambda x: x,
        drop_last=drop_last, pin_memory=True
    )
    print()

    return DataTuple(dataset=dset, torchdset=tset, loader=data_loader)

train_tuple = get_tuple(
    config.train, 
    config.batch_size, 
    shuffle=True, 
    drop_last=True
)
valid_tuple = get_tuple(
    config.valid, 
    config.valid_batch_size, 
    shuffle=False, 
    drop_last=False, 
    topk=5000
)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids, 
                 input_mask, 
                 segment_ids, 
                 lm_label_ids,
                 visual_feats, 
                 obj_labels,
                 is_matched,
                 ans
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.lm_label_ids = lm_label_ids

        self.visual_feats = visual_feats
        self.obj_labels = obj_labels

        self.is_matched = is_matched

        self.ans = ans


def random_word(tokens, tokenizer):
    """
    Masking some random tokens for Language Model task with probabilities as in the original BERT paper.
    :param tokens: list of str, tokenized sentence.
    :param tokenizer: Tokenizer, object used for tokenization (we need it's vocab here)
    :return: (list of str, list of int), masked tokens and related labels for LM prediction
    """
    output_label = []

    for i, token in enumerate(tokens):
        prob = random.random()
        # mask token with probability
        ratio = config.word_mask_rate
        if prob < ratio:
            prob /= ratio

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[i] = "[MASK]"

            # 10% randomly change token to random token
            elif prob < 0.9:
                tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

            # -> rest 10% randomly keep current token

            # append current token to output (we will predict these later)
            try:
                output_label.append(tokenizer.vocab[token])
            except KeyError:
                # For unknown words (should not occur with BPE vocab)
                output_label.append(tokenizer.vocab["[UNK]"])
        else:
            # no masking token (will be ignored by loss function later)
            output_label.append(-1)

    return tokens, output_label


def random_feat(feats, feat_mask):
    mask_feats = feats.copy()
    for i in range(len(feats)):
        prob = random.random()
        # mask token with probability
        if prob < config.frame_mask_rate:
            prob /= config.frame_mask_rate

            # 80% randomly change token to zero feat
            if prob < 0.8:
                mask_feats[i, :] = 0
                feat_mask[i] = 0

            # 10% randomly change token to random feat
            elif prob < 0.9:
                mask_feats[i, :] = train_tuple.torchdset.random_feat()
                feat_mask[i] = 0
            # -> rest 10% randomly keep current feat

            # Need to predict this feat
            
    return mask_feats, feat_mask


def convert_example_to_features(example: InputExample, max_seq_length, tokenizer)->InputFeatures:
    """
    Convert a raw sample (pair of sentences as tokenized strings) into a proper training sample with
    IDs, LM labels, input_mask, CLS and SEP tokens etc.
    :param example: InputExample, containing sentence input as strings and is_next label
    :param max_seq_length: int, maximum length of sequence.
    :param tokenizer: Tokenizer
    :return: InputFeatures, containing all inputs and labels of one sample as IDs (as used for model training)
    """
    tokens = tokenizer.tokenize(example.sent.strip())

    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[:(max_seq_length - 2)]

    # Ge random words
    masked_tokens, masked_label = random_word(tokens, tokenizer)

    # concatenate lm labels and account for CLS, SEP, SEP
    masked_tokens = ['[CLS]'] + masked_tokens + ['[SEP]']
    input_ids = tokenizer.convert_tokens_to_ids(masked_tokens)

    # Mask & Segment Word
    lm_label_ids = ([-1] + masked_label + [-1])
    input_mask = [1] * len(input_ids)
    segment_ids = [0] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        lm_label_ids.append(-1)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(lm_label_ids) == max_seq_length

    feat, pos = example.visual_feats
    feat_mask = example.visual_mask
    

    # Mask Image Features:
    masked_feat, feat_mask = random_feat(feat, feat_mask)

    # QA answer label
    if example.label is None or len(example.label) == 0 or example.is_matched != 1:
        # 1. No label 2. Label is pruned 3. unmatched visual + language pair
        ans = -1
    else:
        keys, values = zip(*example.label.items())
        if len(keys) == 1:
            ans = keys[0]
        else:
            value_sum = sum(values)
            prob = [value / value_sum for value in values]
            choice = np.random.multinomial(1, prob).argmax()
            ans = keys[choice]

    features = InputFeatures(
        input_ids=input_ids,
        input_mask=input_mask,
        segment_ids=segment_ids,
        lm_label_ids=lm_label_ids,
        visual_feats=(masked_feat, pos),
        obj_labels={
            'feat': (feat, feat_mask),
        },
        is_matched=example.is_matched,
        ans=ans,
    )
    return features


LOSSES_NAME = ('Mask_LM', 'Matched', 'Obj', 'Attr', 'Feat', 'QA')

class LXMERT:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.max_seq_length = config.max_seq_length

        self.tokenizer = BertTokenizer.from_pretrained(
            config.model_path,
            do_lower_case=True
        )

        # Build model
        set_visual_config(config)
        self.model = LXRTPretraining.from_pretrained(
            config.model_path,
            task_mask_lm=config.task_mask_lm,
            task_obj_predict=config.task_obj_predict,
            task_matched=config.task_matched,
            task_qa=config.task_qa,
            visual_losses=config.visual_losses,
            num_answers=config.num_answers
        )

        # Weight initialization and loading
        if config.from_scratch:
            print("Train from Scratch: re-initialize all BERT weights.")
            self.model.apply(self.model.init_bert_weights)
        if config.load is not None:
            self.load(config.load)
        if config.load_lxmert is not None:
            # Load lxmert would not load the answer head.
            self.load_lxmert(config.load_lxmert)

        # GPU Options
        self.model = self.model.cuda()
        if config.multiGPU:
            self.model = nn.DataParallel(self.model)

    def forward(self, examples):
        train_features = [convert_example_to_features(example, self.max_seq_length, self.tokenizer)
                          for example in examples]

        # language Inputs
        input_ids = torch.tensor([f.input_ids for f in train_features], dtype=torch.long).cuda()
        input_mask = torch.tensor([f.input_mask for f in train_features], dtype=torch.long).cuda()
        segment_ids = torch.tensor([f.segment_ids for f in train_features], dtype=torch.long).cuda()

        # Visual Inputs
        feats = torch.from_numpy(np.stack([f.visual_feats[0] for f in train_features]).astype(np.float32)).cuda()
        pos = torch.from_numpy(np.stack([f.visual_feats[1] for f in train_features]).astype(np.int32)).cuda()

        # Language Prediction
        lm_labels = torch.tensor([f.lm_label_ids for f in train_features], dtype=torch.long).cuda()

        # Visual Prediction
        obj_labels = {}
        for key in (['feat']):
            visn_labels = torch.from_numpy(np.stack([f.obj_labels[key][0] for f in train_features]).astype(np.float32)).cuda()
            visn_mask = torch.from_numpy(np.stack([f.obj_labels[key][1] for f in train_features]).astype(np.float32)).cuda()
            assert visn_labels.size(0) == visn_mask.size(0) and visn_labels.size(1) == visn_mask.size(1)
            obj_labels[key] = (visn_labels, visn_mask)

        # Joint Prediction
        matched_labels = torch.tensor([f.is_matched for f in train_features], dtype=torch.long).cuda()
        ans = torch.from_numpy(np.stack([f.ans for f in train_features]).astype(np.float32)).cuda()

        loss, losses, ans_logit = self.model(
            input_ids, 
            segment_ids, 
            input_mask, 
            lm_labels,
            feats, 
            pos, 
            obj_labels, 
            matched_labels, 
            ans
        )

        return loss, losses.detach().cpu(), ans_logit

    def train_batch(self, optim, batch):
        optim.zero_grad()
        loss, losses, ans_logit = self.forward(batch)
        if self.config.multiGPU:
            loss = loss.mean()
            losses = losses.mean(0)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), 1.)
        optim.step()

        return loss.item(), losses.cpu().numpy(), ans_logit

    def valid_batch(self, batch):
        with torch.no_grad():
            loss, losses, ans_logit = self.forward(batch)
            if self.config.multiGPU:
                loss = loss.mean()
                losses = losses.mean(0)
        return loss.item(), losses.cpu().numpy(), ans_logit

    def train(self, train_tuple: DataTuple, eval_tuple: DataTuple):
        train_ld = train_tuple.loader

        # Optimizer
        from lxrt.optimization import BertAdam
        batch_per_epoch = len(train_ld)
        t_total = int(batch_per_epoch * self.config.epochs)
        warmup_ratio = 0.05
        warmup_iters = int(t_total * warmup_ratio)
        print("Batch per epoch: %d" % batch_per_epoch)
        print("Total Iters: %d" % t_total)
        print("Warm up Iters: %d" % warmup_iters)
        # optim = BertAdam(
        #     self.model.parameters(), 
        #     lr=self.config.lr, 
        #     warmup=warmup_ratio, 
        #     t_total=t_total)
        optim = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.config.lr)

        # Train
        best_eval_loss = 9595.
        for epoch in range(1 if config.DEBUG else config.epochs):
            # Train
            self.model.train()
            total_loss = 0.
            total_losses = 0.
    
            for batch in tqdm(train_ld, total=len(train_ld)):
                loss, losses, logit = self.train_batch(optim, batch)
                total_loss += loss
                total_losses += losses

            print("The training loss for Epoch %d is %0.4f" % (epoch, total_loss / batch_per_epoch))
            losses_str = "The losses are "
            for name, loss in zip(LOSSES_NAME, total_losses):
                losses_str += "%s: %0.4f " % (name, loss / batch_per_epoch)
            print(losses_str)

            # Eval
            avg_eval_loss = self.evaluate_epoch(eval_tuple, iters=-1)

            # Save
            if avg_eval_loss < best_eval_loss:
                best_eval_loss = avg_eval_loss
                self.save("BEST_EVAL_LOSS")
            self.save("Epoch%02d" % (epoch+1))

    def evaluate_epoch(self, eval_tuple: DataTuple, iters: int=-1):
        self.model.eval()
        eval_ld = eval_tuple.loader
        total_loss = 0.
        total_losses = 0.
        
        for i, batch in enumerate(eval_ld):
            loss, losses, logit = self.valid_batch(batch)
            total_loss += loss
            total_losses += losses
            if i == iters:
                break

        print("The valid loss is %0.4f" % (total_loss / len(eval_ld)))
        losses_str = "The losses are "
        for name, loss in zip(LOSSES_NAME, total_losses / len(eval_ld)):
            losses_str += "%s: %0.4f " % (name, loss)
        print(losses_str)

        return total_loss / len(eval_ld)

    def save(self, name):
        if not os.path.exists(self.config.output):
            os.mkdir(self.config.output)
        torch.save(self.model.state_dict(),
                   os.path.join(self.config.output, "%s_LXRT.pth" % name))


    def load(self, path):
        print("Load BERT extractor from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)
        self.model.load_state_dict(state_dict)

    def load_lxmert(self, path):
        print("Load LXMERT model from %s" % path)
        state_dict = torch.load("%s_LXRT.pth" % path)

        # Do not load any answer head
        for key in list(state_dict.keys()):
            if 'answer' in key:
                state_dict.pop(key)

        # Change Multi GPU to single GPU
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("module."):
                new_state_dict[key[len("module."):]] = value
        state_dict = new_state_dict

        load_keys = set(state_dict.keys())
        model_keys = set(self.model.state_dict().keys())
        print()
        print("Keys in loaded but not in model:")
        for key in sorted(load_keys.difference(model_keys)):
            print(key)
        print()
        print("Keys in model but not in loaded:")
        for key in sorted(model_keys.difference(load_keys)):
            print(key)
        print()

        self.model.load_state_dict(state_dict, strict=False)


if __name__ == "__main__":

    lxmert = LXMERT(config=config)


    lxmert.train(train_tuple, valid_tuple)