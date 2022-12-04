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

# ignore warning log
import warnings
warnings.filterwarnings('ignore')
import paddle
import paddle.distributed as dist
from paddle.distributed import fleet

from ppdet.core.workspace import create
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight

from ppdet.core.workspace import load_config, merge_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.engine import Trainer, init_parallel_env
from ppdet.metrics.coco_utils import json_eval_results
from ppdet.metrics.json_results import get_det_res_pra
from ppdet.metrics import Metric, COCOMetric, VOCMetric, F1Score, get_infer_results


from ppdet.utils.logger import setup_logger
logger = setup_logger('test')

from tqdm import tqdm

class parse_args:
    config = "./configs/yolov5/yolov5_convnext.yml"
    weights = "../model/det/yolov5_convnext.pdparams"


def evaluate(cfg, model, loader, metric, use_amp=True):
    _nranks = dist.get_world_size()
    _local_rank = dist.get_rank()

    model.eval()

    sync_bn = (getattr(cfg, 'norm_type', None) == 'sync_bn' and cfg.use_gpu and _nranks > 1)
    if sync_bn:
        model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    # get distributed model
    if cfg.get('fleet', False):
        model = fleet.distributed_model(model)
    elif _nranks > 1:
        find_unused_parameters = cfg[
            'find_unused_parameters'] if 'find_unused_parameters' in cfg else False
        model = paddle.DataParallel(
            model, find_unused_parameters=find_unused_parameters)

    im_id_l = []
    bbox_l= []
    bbox_num_l = []
    for step_id, data in enumerate(tqdm(loader)):
        # forward
        if use_amp:
            with paddle.amp.auto_cast(
                    enable=cfg.use_gpu,
                    custom_white_list=cfg.get('custom_white_list', None),
                    custom_black_list=cfg.get('custom_black_list', None),
                    level=cfg.get('amp_level', 'O1')):
                outs = model(data)
        else:
            outs = model(data)

        im_id_l.append(data['im_id'])
        bbox_l.append(outs['bbox'])
        bbox_num_l.append(outs['bbox_num'])
    
    im_id_l = paddle.concat(im_id_l, axis=0)
    bbox_l = paddle.concat(bbox_l, axis=0)
    bbox_num_l = paddle.concat(bbox_num_l, axis=0)

    # Gather from all ranks
    if _nranks > 1:
        print("Start gathering from all ranks")
        length_list = []
        im_id_list = []
        bbox_list = []
        bbox_num_list = []
        paddle.distributed.all_gather(length_list, paddle.to_tensor(len(bbox_l)))
        paddle.distributed.all_gather(im_id_list, im_id_l)

        temp_box = paddle.zeros((max(length_list), 6))
        temp_box[:length_list[_local_rank]] = bbox_l

        # gather 的 tensor 形状必须一致， 否则死锁
        paddle.distributed.all_gather(bbox_list, temp_box)
        paddle.distributed.all_gather(bbox_num_list, bbox_num_l)
        print("End gathering from all ranks")

    im_id_dict = {"im_id": paddle.concat(im_id_list, axis=0)}
    outs = {
        "bbox": paddle.concat([v[:l, :] for l, v in zip(length_list, bbox_list)], axis=0),
        "bbox_num": paddle.concat(bbox_num_list, axis=0)
    }

    if _nranks < 2 or _local_rank == 0:
        # update metrics
        metric.update(im_id_dict, outs)
        metric.accumulate()
        metric.log()
    
    num_images =  im_id_dict['im_id'].numpy().shape[0]

    return num_images

def main():
    tic = time.time()

    config_path = sys.argv[1]
    weights_path = sys.argv[2]

    # initalize ddp
    init_parallel_env()

    # load config
    FLAGS = parse_args()
    FLAGS.config = config_path
    FLAGS.weights = weights_path
    cfg = load_config(FLAGS.config)

    # load dataset
    capital_mode = "Eval"
    dataset = cfg['{}Dataset'.format(capital_mode)] = create('{}Dataset'.format(capital_mode))()
    dataset.check_or_download_dataset()
    anno_file = dataset.get_anno()
    loader = create('EvalReader')(
                        dataset,
                        cfg.worker_num,
                        batch_sampler=None)

    clsid2catid = {v: k for k, v in dataset.catid2clsid.items()}
    output_eval = cfg['output_eval'] if 'output_eval' in cfg else None
    bias = 1 if cfg.get('bias', False) else 0
    IouType = cfg['IouType'] if 'IouType' in cfg else 'bbox'
    save_prediction_only = cfg.get('save_prediction_only', False)

    # init metric
    metric = COCOMetric(
                    anno_file=anno_file,
                    clsid2catid=clsid2catid,
                    classwise=True,
                    output_eval=output_eval,
                    bias=bias,
                    IouType=IouType,
                    save_prediction_only=save_prediction_only)
    

    # load model
    model = create(cfg.architecture)
    load_pretrain_weight(model, FLAGS.weights)


    # evaluate
    num_images = evaluate(cfg, model, loader, metric)

    end_time = time.time() - tic
    print("FPS: ", num_images / end_time)
    

if __name__ == '__main__':
    dist.spawn(main, nprocs=4)