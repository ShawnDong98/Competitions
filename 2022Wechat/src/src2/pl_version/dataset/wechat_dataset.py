import os
import json

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from transformers import AutoTokenizer
from category_id_map import category_id_to_lv2id

from dataset.tokenization import BertTokenizer

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids

def convert_tokens_to_features(tokens_a, max_seq_length, tokenizer):
    """Loads a data file into a list of `InputBatch`s."""


    # Account for [CLS] and [SEP] with "- 2"
    if len(tokens_a) > max_seq_length - 2:
        tokens_a = tokens_a[:(max_seq_length - 2)]

    # Keep segment id which allows loading BERT-weights.
    tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
    segment_ids = [0] * len(tokens)

    input_ids = tokenizer.convert_tokens_to_ids(tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    padding = [0] * (max_seq_length - len(input_ids))
    input_ids += padding
    input_mask += padding
    segment_ids += padding

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

                
    return InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids)


class WeChatDataset(torch.utils.data.Dataset):
    def __init__(
        self, 
        model_path,
        data_dir,
        get_asr=True,
        get_ocr=True,
        get_title=True, 
        get_frame=True,
        label_json=None, 
        get_vid=True,
        get_category=False,
        text_maxlen=84, 
        frame_maxlen=32,
    ): 
        self.data_dir = data_dir
        self.get_asr = get_asr
        self.get_ocr = get_ocr
        self.get_title = get_title
        self.get_frame = get_frame
        self.get_vid = get_vid
        self.get_asr = get_asr
        self.get_category = get_category
        self.text_maxlen = text_maxlen
        self.frame_maxlen = frame_maxlen

        
        self.output_dict = {}
        with open(label_json, "r") as f:
            self.data = json.load(f)
            
        # video padding frame
        self.zero_frame = np.zeros(768).astype(dtype=np.float32)
        
        # tokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
             model_path,
             do_lower_case=True
        )

       
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        o = {}
        tokens = []
        if self.get_title:
            title = self.data[index]['title']
            title_a = self.tokenizer.tokenize(title.strip())
            title_tokens = title_a + ["[SEP]"]
            tokens.extend(title_tokens)
        if self.get_asr:
            asr = self.data[index]['asr']
            asr_a = self.tokenizer.tokenize(asr.strip())
            asr_tokens = asr_a + ["[SEP]"]
            tokens.extend(asr_tokens)
            
        if self.get_ocr:
            ocr = [ocr['text'] for ocr in self.data[index]['ocr']]
            ocr = ' '.join(ocr)
            ocr_a = self.tokenizer.tokenize(ocr.strip())
            ocr_tokens = ocr_a
            tokens.extend(ocr_tokens)
        
        text_id_mask = convert_tokens_to_features(tokens, self.text_maxlen, self.tokenizer)
        
        o['id'] = torch.tensor(text_id_mask.input_ids, dtype=torch.long)# [CLS][SEP] Text [SEP]
        o['mask'] = torch.tensor(text_id_mask.input_mask, dtype=torch.long)

        
        if self.get_frame:
            o['frame_features'] = [x for x in np.load(os.path.join(self.data_dir, self.data[index]['id'] + '.npy'))]
            o['frame_mask'] = torch.ones(self.frame_maxlen, dtype=torch.long)
            if len(o['frame_features']) != self.frame_maxlen:
                o['frame_mask'][len(o['frame_features']) - self.frame_maxlen:] = 0 

            frame_features_padding = self.padding_frames(o['frame_features'])
            o['frame_features'] = torch.tensor(frame_features_padding, dtype=torch.float32)
            
        if self.get_vid:
            o['vid'] = self.data[index]['id']
        
        if self.get_category:
            o['target'] = torch.tensor(category_id_to_lv2id(self.data[index]['category_id']), dtype=torch.long)
        
        return o

    def padding_frames(self, frame_feature):
        """padding fram features"""
        num_frames = len(frame_feature)
        frame_gap = (num_frames - 1) / self.frame_maxlen
        if frame_gap <= 1:
            res = frame_feature + [self.zero_frame] * (self.frame_maxlen - num_frames)
        else:
            res = [frame_feature[round((i + 0.5) * frame_gap)] for i in range(self.frame_maxlen)]
        return np.c_[res]
            