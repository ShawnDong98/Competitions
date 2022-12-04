#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

import math
import six
import numpy as np
from numbers import Integral

import paddle
import paddle.nn as nn
from paddle import ParamAttr
from paddle import to_tensor
import paddle.nn.functional as F
from paddle.nn.initializer import Normal, Constant, XavierUniform
from paddle.regularizer import L2Decay

from ppdet.core.workspace import register, serializable

from . import ops

class DropBlock(nn.Layer):
    def __init__(self, block_size, keep_prob, name=None, data_format='NCHW'):
        """
        DropBlock layer, see https://arxiv.org/abs/1810.12890
        Args:
            block_size (int): block size
            keep_prob (int): keep probability
            name (str): layer name
            data_format (str): data format, NCHW or NHWC
        """
        super(DropBlock, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.name = name
        self.data_format = data_format

    def forward(self, x):
        if not self.training or self.keep_prob == 1:
            return x
        else:
            gamma = (1. - self.keep_prob) / (self.block_size**2)
            if self.data_format == 'NCHW':
                shape = x.shape[2:]
            else:
                shape = x.shape[1:3]
            for s in shape:
                gamma *= s / (s - self.block_size + 1)

            matrix = paddle.cast(paddle.rand(x.shape) < gamma, x.dtype)
            mask_inv = F.max_pool2d(
                matrix,
                self.block_size,
                stride=1,
                padding=self.block_size // 2,
                data_format=self.data_format)
            mask = 1. - mask_inv
            y = x * mask * (mask.numel() / mask.sum())
            return y



@register
@serializable
class YOLOBox(object):
    __shared__ = ['num_classes']

    def __init__(self,
                 num_classes=80,
                 conf_thresh=0.005,
                 downsample_ratio=32,
                 clip_bbox=True,
                 scale_x_y=1.):
        self.num_classes = num_classes
        self.conf_thresh = conf_thresh
        self.downsample_ratio = downsample_ratio
        self.clip_bbox = clip_bbox
        self.scale_x_y = scale_x_y

    def __call__(self,
                 yolo_head_out,
                 anchors,
                 im_shape,
                 scale_factor,
                 var_weight=None):
        boxes_list = []
        scores_list = []
        origin_shape = im_shape / scale_factor
        origin_shape = paddle.cast(origin_shape, 'int32')
        for i, head_out in enumerate(yolo_head_out):
            boxes, scores = paddle.vision.ops.yolo_box(
                head_out,
                origin_shape,
                anchors[i],
                self.num_classes,
                self.conf_thresh,
                self.downsample_ratio // 2**i,
                self.clip_bbox,
                scale_x_y=self.scale_x_y)
            boxes_list.append(boxes)
            scores_list.append(paddle.transpose(scores, perm=[0, 2, 1]))
        yolo_boxes = paddle.concat(boxes_list, axis=1)
        yolo_scores = paddle.concat(scores_list, axis=2)
        return yolo_boxes, yolo_scores


@register
@serializable
class MultiClassNMS(object):
    def __init__(self,
                 score_threshold=.05,
                 nms_top_k=-1,
                 keep_top_k=100,
                 nms_threshold=.5,
                 normalized=True,
                 nms_eta=1.0,
                 return_index=False,
                 return_rois_num=True,
                 trt=False):
        super(MultiClassNMS, self).__init__()
        self.score_threshold = score_threshold
        self.nms_top_k = nms_top_k
        self.keep_top_k = keep_top_k
        self.nms_threshold = nms_threshold
        self.normalized = normalized
        self.nms_eta = nms_eta
        self.return_index = return_index
        self.return_rois_num = return_rois_num
        self.trt = trt

    def __call__(self, bboxes, score, background_label=-1):
        """
        bboxes (Tensor|List[Tensor]): 
        1. (Tensor) Predicted bboxes with shape [N, M, 4], N is the batch size and M is the number of bboxes
        2. (List[Tensor]) bboxes and bbox_num, bboxes have shape of [M, C, 4], C is the class number and bbox_num means
        the number of bboxes of each batch with shape [N,] 
        score (Tensor): Predicted scores with shape [N, C, M] or [M, C]
        background_label (int): Ignore the background label; For example, RCNN is num_classes and YOLO is -1. 
        """
        kwargs = self.__dict__.copy()
        if isinstance(bboxes, tuple):
            bboxes, bbox_num = bboxes
            kwargs.update({'rois_num': bbox_num})
        if background_label > -1:
            kwargs.update({'background_label': background_label})
        kwargs.pop('trt')
        # TODO(wangxinxin08): paddle version should be develop or 2.3 and above to run nms on tensorrt
        if self.trt and (int(paddle.version.major) == 0 or
                         (int(paddle.version.major) >= 2 and
                          int(paddle.version.minor) >= 3)):
            # TODO(wangxinxin08): tricky switch to run nms on tensorrt
            kwargs.update({'nms_eta': 1.1})
            bbox, bbox_num, _ = ops.multiclass_nms(bboxes, score, **kwargs)
            bbox = bbox.reshape([1, -1, 6])
            idx = paddle.nonzero(bbox[..., 0] != -1)
            bbox = paddle.gather_nd(bbox, idx)
            return bbox, bbox_num, None
        else:
            return ops.multiclass_nms(bboxes, score, **kwargs)
