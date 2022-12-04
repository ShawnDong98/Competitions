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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time
import json

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')

import paddle

from ppdet.core.workspace import create
from ppdet.core.workspace import load_config, merge_config


from ppdet.utils.logger import setup_logger
logger = setup_logger('eval')

import cv2

config = "./configs/yolov3/yolov3_res50.yml"
cfg = load_config(config)

# load dataset
capital_mode = "Train"
dataset = create('{}Dataset'.format(capital_mode))()
dataset.check_or_download_dataset()
anno_file = dataset.get_anno()
loader = create('TrainReader')(
                    dataset,
                    0,
                    batch_sampler=None)
data = next(loader)

for data in loader:
    for img, bbox in zip(data['image'], data['gt_bbox']):
        img = img.numpy().transpose(1, 2, 0)
        for box in bbox:
            box = box.numpy()
            img = img.copy()
            img = cv2.rectangle(img, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color=(0, 255, 0), thickness=2)
        cv2.imshow("img", img)
        cv2.waitKey(0)
