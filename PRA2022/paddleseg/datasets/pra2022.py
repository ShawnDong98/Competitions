# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import paddle

from paddleseg.cvlibs import manager
from paddleseg.transforms.transforms import Compose


@manager.DATASETS.add_component
class PRA2022(paddle.io.Dataset):
    """
    Args:
        transforms (list): Transforms for image.
        dataset_root (str): tusimple dataset directory.
        mode (str, optional): Which part of dataset to use. it is one of ('train', 'val', 'test'). Default: 'train'.
        cut_height (int, optional): Whether to cut image height while training. Default: 0
    """
    def __init__(self,
                 transforms=None,
                 dataset_root=None,
                 num_classes=4,
                 mode='train',
        ):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms, to_rgb=False)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = num_classes
        self.ignore_index = 255
        self.test_gt_json = os.path.join(self.dataset_root,
                                         'test_set/test_label.json')

        if mode not in ['train', 'val', 'test']:
            raise ValueError(
                "`mode` should be 'train', 'val' or 'test', but got {}.".format(
                    mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")

        if not os.path.exists(self.dataset_root):
            raise FileNotFoundError('there is not `dataset_root`: {}.'.format(
                self.dataset_root))


        if self.dataset_root is None:
            raise ValueError("`dataset_root` is necessary, but it is None.")

        if mode == 'train':
            file_path = os.path.join(self.dataset_root, 'train_list.txt')
        elif mode == 'val':
            file_path = os.path.join(self.dataset_root, 'val_list.txt')
        else:
            file_path = os.path.join(self.dataset_root, 'mini_val_list.txt')

        with open(file_path, 'r') as f:
            for line in f:
                items = line.strip().split()
                if len(items) != 2:
                    if mode == 'train' or mode == 'val':
                        raise Exception(
                            "File list format incorrect! It should be"
                            " image_name label_name\\n")
                    image_path = os.path.join(self.dataset_root, items[0])
                    label_path = None
                else:
                    image_path = self.dataset_root + items[0]
                    label_path = self.dataset_root + items[1]
                self.file_list.append([image_path, label_path])

    def __getitem__(self, idx):
        data = {}
        data['trans_info'] = []
        image_path, label_path = self.file_list[idx]
        data['img'] = image_path
        data['label'] = label_path

        data['gt_fields'] = []
        if self.mode == 'val':
            data = self.transforms(data)
            data['label'] = data['label'][np.newaxis, :, :]
        else:
            data['gt_fields'].append('label')
            data = self.transforms(data)

        return data

    def __len__(self):
        return len(self.file_list)