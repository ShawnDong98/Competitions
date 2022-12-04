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

import collections
import os
import sys
import time
import json
import math

# ignore warning log
import warnings
warnings.filterwarnings('ignore')
import paddle
import paddle.nn.functional as F
import paddle.distributed as dist
from paddle.distributed import fleet

from ppdet.core.workspace import create
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight

from ppdet.core.workspace import load_config, merge_config
from ppdet.utils.cli import ArgsParser, merge_args
from ppdet.engine import Trainer, init_parallel_env
from ppdet.metrics.coco_utils import json_eval_results
from ppdet.metrics.json_results import get_det_res_pra

from paddleseg.cvlibs import manager, Config
from paddleseg.utils import utils

import cv2
import numpy as np

from ppdet.utils.logger import setup_logger
logger = setup_logger('test')

from tqdm import tqdm

class parse_args:
    det_config = "./PaddleDet/configs/yolov3/yolov3_res34_infer.yml"
    det_weights = "./model/det/yolov3_res34.pdparams"
    seg_config = "./ppseg/configs/pra2022/hardnet_infer.yml"
    seg_weights = "./model/seg/hardnet.pdparams"


def reverse_transform(pred, trans_info, mode='nearest'):
    """recover pred to origin shape"""
    intTypeList = [paddle.int8, paddle.int16, paddle.int32, paddle.int64]
    dtype = pred.dtype
    origin_shape = trans_info['im_shape'] / trans_info['scale_factor']
    origin_shape = paddle.cast(origin_shape, 'int32')
    h, w = origin_shape[:, 0], origin_shape[:, 1]
    if paddle.get_device() == 'cpu' and dtype in intTypeList:
        pred = paddle.cast(pred, 'float32')
        pred = F.interpolate(pred, (h, w), mode=mode)
        pred = paddle.cast(pred, dtype)
    else:
        pred = F.interpolate(pred, [h, w], mode=mode)
        
    return pred

def inference(cfg, det_model, seg_model, loader, amp=True):

    det_model.eval()
    seg_model.eval()

    im_id_l = []
    bbox_l= []
    bbox_num_l = []
    c_results = collections.defaultdict(list)
    print("using amp: ", amp)
    with paddle.no_grad():
        for step_id, data in enumerate(tqdm(loader)):
            # forward
            if amp:
                with paddle.amp.auto_cast(
                                level='O2',
                                enable=True,
                                custom_white_list={
                                    "elementwise_add", "batch_norm",
                                    "sync_batch_norm"
                                },
                                custom_black_list={'bilinear_interp_v2'}):
                    det_outs = det_model(data)
                    seg_time = time.time()
                    logits = seg_model(data['image'])
                    if not isinstance(logits, collections.abc.Sequence):
                        raise TypeError(
                            "The type of logits must be one of collections.abc.Sequence, e.g. list, tuple. But received {}"
                            .format(type(logits)))
                    logit = logits[0]
                    # logit = reverse_transform(logit, data, mode='bilinear')
                    seg_outs = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')
                    seg_outs = seg_outs.numpy().astype('uint8')
                    print("seg infer time: ", time.time() - seg_time)
            else:
                det_outs = det_model(data)
                logits = seg_model(data['image'])
                if not isinstance(logits, collections.abc.Sequence):
                    raise TypeError(
                        "The type of logits must be one of collections.abc.Sequence, e.g. list, tuple. But received {}"
                        .format(type(logits)))
                logit = logits[0]
                # logit = reverse_transform(logit, data, mode='bilinear')
                seg_outs = paddle.argmax(logit, axis=1, keepdim=True, dtype='int32')
                seg_outs = seg_outs.numpy().astype('uint8')

            seg_postprocess_time = time.time()
            for i, (image_id, scale_factor) in enumerate(zip(data['im_id'].numpy(), data['scale_factor'].numpy())):
                # 语义分割后处理
                # seg_result = np.squeeze(seg_results[i, :, :]) * 80
                seg_result = np.squeeze(seg_outs[i, :, :])
                
                max_value = np.max(seg_result)
                if max_value > 0:
                    for iiii in range(1, max_value + 1):
                        gt = np.zeros(seg_result.shape, dtype=np.uint8)
                        gt[seg_result == iiii] = 255
                        label = iiii + 7
                        contours, hierarchy = cv2.findContours(gt.astype('uint8'), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
                        w_ratio = 1.0 / scale_factor[1]
                        h_ratio = 1.0 / scale_factor[0]
                        for contour in contours: 
                            if cv2.contourArea(contour) > 200:
                                rect = cv2.boundingRect(contour)
                                segmentations = []
                                for point in contour:
                                    segmentations.append(round(point[0][0] * w_ratio, 0))
                                    segmentations.append(round(point[0][1] * h_ratio, 0))

                                c_results["result"].append({
                                    "image_id": int(image_id),
                                    "type": label,
                                    "x": round(rect[0] * w_ratio, 0),
                                    "y": round(rect[1] * h_ratio, 0),
                                    "width": round(rect[2] * w_ratio, 0),
                                    "height": round(rect[3] * h_ratio, 0),
                                    "segmentation": [segmentations]}
                                )
            print("seg post process time: ", time.time() - seg_postprocess_time)
            im_id_l.append(data['im_id'])
            bbox_l.append(det_outs['bbox'])
            bbox_num_l.append(det_outs['bbox_num'])
    
    im_id_l = paddle.concat(im_id_l, axis=0)
    bbox_l = paddle.concat(bbox_l, axis=0)
    bbox_num_l = paddle.concat(bbox_num_l, axis=0)


    im_id_dict = {"im_id": im_id_l}
    outs = {
        "bbox": bbox_l,
        "bbox_num": bbox_num_l
    }
    num_images =  im_id_dict['im_id'].numpy().shape[0]
    label_to_cat_id_map = {
        0 : 1,
        1 : 2,
        2 : 3,
        3 : 4,
        4 : 5,
        5 : 6,
        6 : 7,
    }

    res = get_det_res_pra(outs["bbox"], outs["bbox_num"], im_id_dict["im_id"], label_to_cat_id_map)

    c_results['result'].extend(res)

    return c_results, num_images

def main():
    tic = time.time()

    infer_txt = sys.argv[1]
    result_path = sys.argv[2]

    # initalize ddp
    init_parallel_env()

    # load config
    FLAGS = parse_args()
    cfg = load_config(FLAGS.det_config)

    # load dataset
    capital_mode = "Test"
    dataset = cfg['{}Dataset'.format(capital_mode)] = create('{}Dataset'.format(capital_mode))()
    
    # load loader
    with open(infer_txt, "r") as f: 
        images = [line.strip() for line in f.readlines()]
    dataset.set_images(images)
    loader = create('TestReader')(dataset, cfg.worker_num)

    # load model
    det_model = create(cfg.architecture)
    load_pretrain_weight(det_model, FLAGS.det_weights)

    seg_cfg = Config(FLAGS.seg_config)
    seg_model = seg_cfg.model
    utils.load_entire_model(seg_model, FLAGS.seg_weights)
    logger.info('Loaded trained params of model successfully')

    # inference
    res, num_images = inference(cfg, det_model, seg_model, loader)

    with open(result_path, "w") as f:
        json.dump(res, f)

    end_time = time.time() - tic
    print("num predict results: ", len(res['result']))
    print("FPS: ", num_images / end_time)
    

if __name__ == '__main__':
    main()