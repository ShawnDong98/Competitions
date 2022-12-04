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

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

# ignore warning log
import warnings
warnings.filterwarnings('ignore')

import paddle

from ppdet.core.workspace import create
from ppdet.core.workspace import load_config, merge_config
from ppdet.utils.check import check_gpu, check_npu, check_xpu, check_version, check_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.engine import Trainer, init_parallel_env
from ppdet.metrics.coco_utils import json_eval_results

from ppdet.utils.logger import setup_logger
logger = setup_logger('eval')

class parse_args:
    config = "./configs/yolov3/yolov3_pra2022.yml"


def create_model():
    pass

def load_weights():
    pass

def load_loader():
    pass


def main():
    pass


if __name__ == '__main__':
    FLAGS = parse_args()
    cfg = load_config(FLAGS.config)
    print(cfg)

    capital_mode = "Test"
    dataset = cfg['{}Dataset'.format(capital_mode)] = create('{}Dataset'.format(capital_mode))()
    print(dataset)
    
    with open("data_val.txt", "r") as f: images = f.readlines()
    dataset.set_images(images)
    loader = create('TestReader')(dataset, cfg.worker_num)
    print(loader)


    main()