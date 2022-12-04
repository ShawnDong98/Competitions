from collections import defaultdict
import os
import json
import random
import time

import numpy as np
from torch.utils.data import Dataset


TINY_IMG_NUM = 500
FAST_IMG_NUM = 5000


Split2ImgFeatPath = {
    'unlabeled': './data/zip_feats/unlabeled',
    'labeled': './data/zip_feats/labeled',
    'test_b': './data/zip_feats/test_b',
}

class InputExample(object):
    """A single training/test example for the language model."""
    def __init__(self, 
                 uid, 
                 sent, 
                 visual_feats=None,
                 visual_mask=None,
                 is_matched=None, 
                 label=None
    ):
        self.uid = uid
        self.sent = sent
        self.visual_feats = visual_feats
        self.visual_mask = visual_mask
        self.is_matched = is_matched  # whether the visual and obj matched

        self.label = label


class LXMERTDataset:
    def __init__(self, root_path, splits: str):
        self.name = splits
        self.sources = splits.split(',')

        self.data = {}
        for source in self.sources:
            self.data[source] = json.load(open(os.path.join(root_path, "%s.json" % source)))
        print("Load %d data from %s" % (len(self.data), self.name))

    def __len__(self):
        tmp = []
        for k, v in self.data.items():
            tmp.extend(v)
        return len(tmp)


def padding_frames(frame_maxlen, frame_feature):
    """padding fram features"""
    zero_frame = np.zeros(768).astype(dtype=np.float32)
    num_frames = len(frame_feature)
    frame_gap = (num_frames - 1) / frame_maxlen
    if frame_gap <= 1:
        res = frame_feature + [zero_frame] * (frame_maxlen - num_frames)
    else:
        res = [frame_feature[round((i + 0.5) * frame_gap)] for i in range(frame_maxlen)]
    return np.c_[res]

FIELDNAMES = ["img_id", "features", "pos"]
def load_features(root_dir, ann, topk=None):
    data = []
    start_time = time.time()
    fname = root_dir.split('/')[-1]
    print("Start to load video features from %s" % fname)
    
    for i, _ in enumerate(ann):
        item = {}
        item["img_id"] = _['id']
        
        frame_features = [x for x in np.load(os.path.join(root_dir, item['img_id'] + '.npy'))]
        item['frame_mask'] = np.ones(32, dtype=np.float32)
        if len(frame_features) != 32:
            item['frame_mask'][len(frame_features) - 32:] = 0 

        item['features'] = padding_frames(32, frame_features)
        item['pos'] = np.arange(32, dtype=np.int32)

        data.append(item)
        if topk is not None and len(data) == topk:
            break

    elapsed_time = time.time() - start_time
    print("Loaded %d images in file %s in %d seconds." % (len(data), fname, elapsed_time))
    return data


class LXMERTTorchDataset(Dataset):
    def __init__(self,
                config, 
                dataset: LXMERTDataset,
                topk=-1):
        super().__init__()
        self.raw_dataset = dataset
        self.task_matched = config.task_matched

        if config.tiny:
            topk = TINY_IMG_NUM
        elif config.fast:
            topk = FAST_IMG_NUM

        img_data = []
        for split, ann in self.raw_dataset.data.items():
            img_data.extend(load_features(Split2ImgFeatPath[split], ann, topk))
            
        self.imgid2img = {}
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum.copy()

        
        # Filter out the dataset
        tmp = []
        for k, v in self.raw_dataset.data.items():
            print(f"loading {k}")
            tmp.extend(v)
        self.raw_dataset.data = tmp
        used_data = []
        for datum in self.raw_dataset.data:
            if datum['id'] in self.imgid2img:
                used_data.append(datum)

        # Flatten the dataset (into one sent + one image entries)
        self.data = []
        for i, datum in enumerate(used_data):
            sent = datum['title'] + datum['asr'] + ' '.join([ocr['text'] for ocr in datum['ocr']])
            new_datum = {
                'uid': datum['id'],
                'img_id': datum['id'],
                'sent': sent
            }
            self.data.append(new_datum)

        print("Use %d data in torch dataset" % (len(self.data)))

    def __len__(self):
        return len(self.data)

    def random_feat(self):
        """Get a random obj feat from the dataset."""
        datum = self.data[random.randint(0, len(self.data)-1)]
        img_id = datum['img_id']
        img_info = self.imgid2img[img_id]
        feat = img_info['features'][random.randint(0, 31)]
        return feat


    def __getitem__(self, item: int):
        datum = self.data[item]

        uid = datum['uid']
        img_id = datum['img_id'] 

        # Get image info
        img_info = self.imgid2img[img_id]
        feats = img_info['features'].copy()
        pos = img_info['pos'].copy()
        frame_mask = img_info['frame_mask'].copy()

        # If calculating the matched loss, replace the sentence with an sentence
        # corresponding to other image.
        is_matched = 1
        sent = datum['sent']
        if self.task_matched:
            if random.random() < 0.5:
                is_matched = 0
                other_datum = self.data[random.randint(0, len(self.data)-1)]
                while other_datum['img_id'] == img_id:
                    other_datum = self.data[random.randint(0, len(self.data)-1)]
                sent = other_datum['sent']

        # Create target
        example = InputExample(
            uid, 
            sent, 
            (feats, pos), 
            frame_mask,
            is_matched
        )
        return example
        