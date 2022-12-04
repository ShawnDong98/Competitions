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
import copy
import time
from tqdm import tqdm

import numpy as np
import typing
from PIL import Image, ImageOps, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

import paddle
import paddle.nn as nn
import paddle.distributed as dist
from paddle.distributed import fleet
from paddle.static import InputSpec
from ppdet.optimizer import ModelEMA

from ppdet.core.workspace import create
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight
from ppdet.utils.visualizer import visualize_results, save_result
from ppdet.metrics import Metric, COCOMetric, VOCMetric, F1Score, get_infer_results

from ppdet.data.source.category import get_categories
import ppdet.utils.stats as stats
from ppdet.utils import profiler


from .callbacks import Callback, ComposeCallback, LogPrinter, Checkpointer

from paddle.distributed.fleet.utils.hybrid_parallel_util import fused_allreduce_gradients

from ppdet.utils.logger import setup_logger
logger = setup_logger('ppdet.engine')

__all__ = ['Trainer']


class Trainer(object):
    def __init__(self, cfg, mode='train'):
        self.cfg = cfg
        assert mode.lower() in ['train', 'eval', 'test'], \
                "mode should be 'train', 'eval' or 'test'"
        self.mode = mode.lower()
        self.optimizer = None
        self.is_loaded_weights = False
        self.use_amp = self.cfg.get('amp', False)
        self.amp_level = self.cfg.get('amp_level', 'O1')
        self.custom_white_list = self.cfg.get('custom_white_list', None)
        self.custom_black_list = self.cfg.get('custom_black_list', None)

        # build data loader
        capital_mode = self.mode.capitalize()
        self.dataset = self.cfg['{}Dataset'.format(capital_mode)] = create('{}Dataset'.format(capital_mode))()

        if self.mode == 'train':
            self.loader = create('{}Reader'.format(capital_mode))(
                self.dataset, cfg.worker_num)


        # build model
        if 'model' not in self.cfg:
            self.model = create(cfg.architecture)
        else:
            self.model = self.cfg.model
            self.is_loaded_weights = True

        # print(self.model)

        # EvalDataset build with BatchSampler to evaluate in single device
        # TODO: multi-device evaluate
        if self.mode == 'eval':
            # self._eval_batch_sampler = paddle.io.BatchSampler(
            #     self.dataset, batch_size=self.cfg.EvalReader['batch_size'])
            self._eval_batch_sampler = None
            reader_name = '{}Reader'.format(self.mode.capitalize())
            # If metric is VOC, need to be set collate_batch=False.
            if cfg.metric == 'VOC':
                cfg[reader_name]['collate_batch'] = False
            self.loader = create(reader_name)(self.dataset, cfg.worker_num, self._eval_batch_sampler)
        # TestDataset build after user set images, skip loader creation here

        params = sum([
            p.numel() for n, p in self.model.named_parameters()
            if all([x not in n for x in ['_mean', '_variance']])
        ])  # exclude BatchNorm running status
        print('Params: ', params / 1e6)

        # build optimizer in train mode
        if self.mode == 'train':
            steps_per_epoch = len(self.loader)
            if steps_per_epoch < 1:
                logger.warning(
                    "Samples in dataset are less than batch_size, please set smaller batch_size in TrainReader."
                )
            self.lr = create('LearningRate')(steps_per_epoch)
            self.optimizer = create('OptimizerBuilder')(self.lr, self.model)

            print(self.lr)
            print(self.optimizer)

        if self.use_amp and self.amp_level == 'O2':
            self.model, self.optimizer = paddle.amp.decorate(
                models=self.model,
                optimizers=self.optimizer,
                level=self.amp_level)
        self.use_ema = ('use_ema' in cfg and cfg['use_ema'])

        if self.use_ema:
            ema_decay = self.cfg.get('ema_decay', 0.9998)
            ema_decay_type = self.cfg.get('ema_decay_type', 'threshold')
            cycle_epoch = self.cfg.get('cycle_epoch', -1)
            ema_black_list = self.cfg.get('ema_black_list', None)
            self.ema = ModelEMA(
                self.model,
                decay=ema_decay,
                ema_decay_type=ema_decay_type,
                cycle_epoch=cycle_epoch,
                ema_black_list=ema_black_list)

        self._nranks = dist.get_world_size()
        self._local_rank = dist.get_rank()

        print(self._nranks)
        print(self._local_rank)


        self.status = {}

        self.start_epoch = 0
        self.end_epoch = 0 if 'epoch' not in cfg else cfg.epoch

        # initial default callbacks
        self._init_callbacks()

        # initial default metrics
        self._init_metrics()
        self._reset_metrics()

    def _init_callbacks(self):
        if self.mode == 'train':
            self._callbacks = [LogPrinter(self), Checkpointer(self)]
            self._compose_callback = ComposeCallback(self._callbacks)
        elif self.mode == 'eval':
            self._callbacks = [LogPrinter(self)]
            self._compose_callback = ComposeCallback(self._callbacks)
        else:
            self._callbacks = []
            self._compose_callback = None

    def _init_metrics(self, validate=False):
        if self.mode == 'test' or (self.mode == 'train' and not validate):
            self._metrics = []
            return
        classwise = self.cfg['classwise'] if 'classwise' in self.cfg else False
        if self.cfg.metric == 'COCO':
            # TODO: bias should be unified
            bias = 1 if self.cfg.get('bias', False) else 0
            output_eval = self.cfg['output_eval'] \
                if 'output_eval' in self.cfg else None
            save_prediction_only = self.cfg.get('save_prediction_only', False)
    
            # pass clsid2catid info to metric instance to avoid multiple loading
            # annotation file
            clsid2catid = {v: k for k, v in self.dataset.catid2clsid.items()} if self.mode == 'eval' else None

            # when do validation in train, annotation file should be get from
            # EvalReader instead of self.dataset(which is TrainReader)
            if self.mode == 'train' and validate:
                eval_dataset = self.cfg['EvalDataset']
                eval_dataset.check_or_download_dataset()
                anno_file = eval_dataset.get_anno()
                dataset = eval_dataset
            else:
                dataset = self.dataset
                anno_file = dataset.get_anno()

            IouType = self.cfg['IouType'] if 'IouType' in self.cfg else 'bbox'
            self._metrics = [
                COCOMetric(
                    anno_file=anno_file,
                    clsid2catid=clsid2catid,
                    classwise=classwise,
                    output_eval=output_eval,
                    bias=bias,
                    IouType=IouType,
                    save_prediction_only=save_prediction_only)
            ]
        elif self.cfg.metric == 'VOC':
            output_eval = self.cfg['output_eval'] \
                if 'output_eval' in self.cfg else None
            save_prediction_only = self.cfg.get('save_prediction_only', False)

            self._metrics = [
                VOCMetric(
                    label_list=self.dataset.get_label_list(),
                    class_num=self.cfg.num_classes,
                    map_type=self.cfg.map_type,
                    classwise=classwise,
                    output_eval=output_eval,
                    save_prediction_only=save_prediction_only)
            ]
        elif self.cfg.metric == 'F1Score':
            output_eval = self.cfg['output_eval'] \
                if 'output_eval' in self.cfg else None
            if self.mode == 'train' and validate:
                eval_dataset = self.cfg['EvalDataset']
                eval_dataset.check_or_download_dataset()
                anno_file = eval_dataset.get_anno()
                dataset = eval_dataset
            else:
                dataset = self.dataset
                anno_file = dataset.get_anno()
            IouType = self.cfg['IouType'] if 'IouType' in self.cfg else 'bbox'
            self._metrics = [
                F1Score(
                    anno_file=anno_file,
                    output_eval=output_eval,
                    IouType=IouType,
                )
            ]
        else:
            logger.warning("Metric not support for metric type {}".format(self.cfg.metric))
            self._metrics = []

    def _reset_metrics(self):
        for metric in self._metrics:
            metric.reset()

    def register_callbacks(self, callbacks):
        callbacks = [c for c in list(callbacks) if c is not None]
        for c in callbacks:
            assert isinstance(c, Callback), \
                    "metrics shoule be instances of subclass of Metric"
        self._callbacks.extend(callbacks)
        self._compose_callback = ComposeCallback(self._callbacks)

    def register_metrics(self, metrics):
        metrics = [m for m in list(metrics) if m is not None]
        for m in metrics:
            assert isinstance(m, Metric), \
                    "metrics shoule be instances of subclass of Metric"
        self._metrics.extend(metrics)

    def load_weights(self, weights):
        if self.is_loaded_weights:
            return
        self.start_epoch = 0
        load_pretrain_weight(self.model, weights)
        logger.debug("Load weights {} to start training".format(weights))

    def load_weights_sde(self, det_weights, reid_weights):
        if self.model.detector:
            load_weight(self.model.detector, det_weights)
            load_weight(self.model.reid, reid_weights)
        else:
            load_weight(self.model.reid, reid_weights)

    def resume_weights(self, weights):
        # support Distill resume weights
        if hasattr(self.model, 'student_model'):
            self.start_epoch = load_weight(self.model.student_model, weights, self.optimizer)
        else:
            self.start_epoch = load_weight(self.model, weights, self.optimizer, self.ema if self.use_ema else None)
        logger.debug("Resume weights of epoch {}".format(self.start_epoch))

    def train(self, validate=False):
        assert self.mode == 'train', "Model not in 'train' mode"
        Init_mark = False
        if validate:
            self.cfg['EvalDataset'] = self.cfg.EvalDataset = create("EvalDataset")()

        model = self.model
        sync_bn = (getattr(self.cfg, 'norm_type', None) == 'sync_bn' and self.cfg.use_gpu and self._nranks > 1)
        if sync_bn:
            model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # enabel auto mixed precision mode
        if self.use_amp:
            scaler = paddle.amp.GradScaler(
                enable=self.cfg.use_gpu or self.cfg.use_npu,
                init_loss_scaling=self.cfg.get('init_loss_scaling', 1024))
        else:
            scaler = paddle.amp.GradScaler(enable=False)

        # get distributed model
        if self.cfg.get('fleet', False):
            model = fleet.distributed_model(model)
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
        elif self._nranks > 1:
            find_unused_parameters = self.cfg[
                'find_unused_parameters'] if 'find_unused_parameters' in self.cfg else False
            model = paddle.DataParallel(
                model, find_unused_parameters=find_unused_parameters)

        self.status.update({
            'epoch_id': self.start_epoch,
            'step_id': 0,
            'steps_per_epoch': len(self.loader)
        })

        self.status['batch_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['data_time'] = stats.SmoothedValue(
            self.cfg.log_iter, fmt='{avg:.4f}')
        self.status['training_staus'] = stats.TrainingStats(self.cfg.log_iter)

        if self.cfg.get('print_flops', False):
            flops_loader = create('{}Reader'.format(self.mode.capitalize()))(self.dataset, self.cfg.worker_num)
            self._flops(flops_loader)
        profiler_options = self.cfg.get('profiler_options', None)

        self._compose_callback.on_train_begin(self.status)

        use_fused_allreduce_gradients = self.cfg[
            'use_fused_allreduce_gradients'] if 'use_fused_allreduce_gradients' in self.cfg else False

        for epoch_id in range(self.start_epoch, self.cfg.epoch):
            self.status['mode'] = 'train'
            self.status['epoch_id'] = epoch_id
            self._compose_callback.on_epoch_begin(self.status)
            self.loader.dataset.set_epoch(epoch_id)
            model.train()
            iter_tic = time.time()
            for step_id, data in enumerate(self.loader):
                self.status['data_time'].update(time.time() - iter_tic)
                self.status['step_id'] = step_id
                profiler.add_profiler_step(profiler_options)
                self._compose_callback.on_step_begin(self.status)
                data['epoch_id'] = epoch_id
                data['num_gpus'] = self._nranks

                if self.use_amp:
                    if isinstance(model, paddle.DataParallel) and use_fused_allreduce_gradients:
                        with model.no_sync():
                            with paddle.amp.auto_cast(
                                    enable=self.cfg.use_gpu,
                                    custom_white_list=self.custom_white_list,
                                    custom_black_list=self.custom_black_list,
                                    level=self.amp_level):
                                # model forward
                                outputs = model(data)
                                loss = outputs['loss']
                            # model backward
                            scaled_loss = scaler.scale(loss)
                            scaled_loss.backward()
                        fused_allreduce_gradients(list(model.parameters()), None)
                    else:
                        with paddle.amp.auto_cast(
                                enable=self.cfg.use_gpu,
                                custom_white_list=self.custom_white_list,
                                custom_black_list=self.custom_black_list,
                                level=self.amp_level):
                            # model forward
                            outputs = model(data)
                            loss = outputs['loss']
                        # model backward
                        scaled_loss = scaler.scale(loss)
                        scaled_loss.backward()
                    # in dygraph mode, optimizer.minimize is equal to optimizer.step
                    scaler.minimize(self.optimizer, scaled_loss)
                else:
                    if isinstance(model, paddle.DataParallel) and use_fused_allreduce_gradients:
                        with model.no_sync():
                            # model forward
                            outputs = model(data)
                            loss = outputs['loss']
                            # model backward
                            loss.backward()
                        fused_allreduce_gradients(
                            list(model.parameters()), None)
                    else:
                        # model forward
                        outputs = model(data)
                        loss = outputs['loss']
                        # model backward
                        loss.backward()
                    self.optimizer.step()
                curr_lr = self.optimizer.get_lr()
                self.lr.step()
                if self.cfg.get('unstructured_prune'):
                    self.pruner.step()
                self.optimizer.clear_grad()
                self.status['learning_rate'] = curr_lr

                if self._nranks < 2 or self._local_rank == 0:
                    self.status['training_staus'].update(outputs)

                self.status['batch_time'].update(time.time() - iter_tic)
                self._compose_callback.on_step_end(self.status)
                if self.use_ema:
                    self.ema.update()
                iter_tic = time.time()

            is_snapshot = (self._nranks < 2 or self._local_rank == 0)  and ((epoch_id + 1) % self.cfg.snapshot_epoch == 0 or epoch_id == self.end_epoch - 1)

            if is_snapshot and self.use_ema:
                # apply ema weight on model
                weight = copy.deepcopy(self.model.state_dict())
                self.model.set_dict(self.ema.apply())
                self.status['weight'] = weight

            self._compose_callback.on_epoch_end(self.status)

            if validate and is_snapshot:
                if not hasattr(self, '_eval_loader'):
                    # build evaluation dataset and loader
                    self._eval_dataset = self.cfg.EvalDataset
                    self._eval_batch_sampler = \
                        paddle.io.BatchSampler(
                            self._eval_dataset,
                            batch_size=self.cfg.EvalReader['batch_size'])
                    # If metric is VOC, need to be set collate_batch=False.
                    if self.cfg.metric == 'VOC':
                        self.cfg['EvalReader']['collate_batch'] = False
                    self._eval_loader = create('EvalReader')(
                        self._eval_dataset,
                        self.cfg.worker_num,
                        batch_sampler=self._eval_batch_sampler)
                # if validation in training is enabled, metrics should be re-init
                # Init_mark makes sure this code will only execute once
                if validate and Init_mark == False:
                    Init_mark = True
                    self._init_metrics(validate=validate)
                    self._reset_metrics()

                with paddle.no_grad():
                    self.status['save_best_model'] = True
                    self._eval_with_loader(self._eval_loader)

            if is_snapshot and self.use_ema:
                # reset original weight
                self.model.set_dict(weight)
                self.status.pop('weight')

        self._compose_callback.on_train_end(self.status)

    def _eval_with_loader(self, loader):
        sample_num = 0
        tic = time.time()
        self._compose_callback.on_epoch_begin(self.status)
        self.status['mode'] = 'eval'
        self.model.eval()
        if self.cfg.get('print_flops', False):
            flops_loader = create('{}Reader'.format(self.mode.capitalize()))(
                self.dataset, self.cfg.worker_num, self._eval_batch_sampler)
            self._flops(flops_loader)

        # ddp
        model = self.model
        sync_bn = (getattr(self.cfg, 'norm_type', None) == 'sync_bn' and self.cfg.use_gpu and self._nranks > 1)
        if sync_bn:
            model = paddle.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        # get distributed model
        if self.cfg.get('fleet', False):
            model = fleet.distributed_model(model)
            self.optimizer = fleet.distributed_optimizer(self.optimizer)
        elif self._nranks > 1:
            find_unused_parameters = self.cfg[
                'find_unused_parameters'] if 'find_unused_parameters' in self.cfg else False
            model = paddle.DataParallel(
                model, find_unused_parameters=find_unused_parameters)

        im_id_l = []
        bbox_l = []
        bbox_num_l = []
        for step_id, data in enumerate(loader):
            self.status['step_id'] = step_id
            self._compose_callback.on_step_begin(self.status)
            # forward
            # data['image'] = paddle.to_tensor(np.load('x0.npy')) ###
            if self.use_amp:
                with paddle.amp.auto_cast(
                        enable=self.cfg.use_gpu,
                        custom_white_list=self.custom_white_list,
                        custom_black_list=self.custom_black_list,
                        level=self.amp_level):
                    outs = model(data)
            else:
                outs = model(data)
            im_id_l.append(data['im_id'])
            bbox_l.append(outs['bbox'])
            bbox_num_l.append(outs['bbox_num'])
            # Gather from all ranks
            # if self._nranks > 1:
            #     length_list = []
            #     im_id_list = []
            #     bbox_list = []
            #     bbox_num_list = []
            #     paddle.distributed.all_gather(length_list, paddle.to_tensor(len(outs['bbox'])))
            #     paddle.distributed.all_gather(im_id_list, data['im_id'])

            #     temp_box = paddle.zeros((max(length_list), 6))
            #     temp_box[:length_list[self._local_rank]] = outs['bbox']

            #     # gather 的 tensor 形状必须一致， 否则死锁
            #     paddle.distributed.all_gather(bbox_list, temp_box)
            #     paddle.distributed.all_gather(bbox_num_list, outs['bbox_num'])
            

            #     # Some image has been evaluated and should be eliminated in last iter
            #     # if (step_id + 1) * self._nranks > len(loader):
            #     #      valid = len(loader) - iter * self._nranks
            #     #      bbox_list = bbox_list[:valid]
            #     #      bbox_num_list = bbox_num_list[:valid]
            #     print("End gathering from all ranks")
            # im_id_dict = {"im_id": paddle.concat(im_id_list, axis=0)}
            # outs = {
            #     "bbox": paddle.concat([v[:l, :] for l, v in zip(length_list, bbox_list)], axis=0),
            #     "bbox_num": paddle.concat(bbox_num_list, axis=0)
            # }
            # if self._nranks < 2 or self._local_rank == 0:
            #     # update metrics
            #     for metric in self._metrics:
            #         metric.update(im_id_dict, outs)

            # # multi-scale inputs: all inputs have same im_id
            # # if isinstance(data, typing.Sequence):
            # #     sample_num += data[0]['im_id'].numpy().shape[0]
            # # else:
            # #     sample_num += data['im_id'].numpy().shape[0]
            # sample_num += im_id_dict['im_id'].numpy().shape[0]
            self._compose_callback.on_step_end(self.status)

        im_id_l = paddle.concat(im_id_l, axis=0)
        bbox_l = paddle.concat(bbox_l, axis=0)
        bbox_num_l = paddle.concat(bbox_num_l, axis=0)
        # Gather from all ranks
        if self._nranks > 1:
            length_list = []
            im_id_list = []
            bbox_list = []
            bbox_num_list = []
            paddle.distributed.all_gather(length_list, paddle.to_tensor(len(bbox_l)))
            paddle.distributed.all_gather(im_id_list, im_id_l)

            temp_box = paddle.zeros((max(length_list), 6))
            temp_box[:length_list[self._local_rank]] = bbox_l

            # gather 的 tensor 形状必须一致， 否则死锁
            paddle.distributed.all_gather(bbox_list, temp_box)
            paddle.distributed.all_gather(bbox_num_list, bbox_num_l)

        im_id_dict = {"im_id": paddle.concat(im_id_list, axis=0)}
        outs = {
            "bbox": paddle.concat([v[:l, :] for l, v in zip(length_list, bbox_list)], axis=0),
            "bbox_num": paddle.concat(bbox_num_list, axis=0)
        }
        if self._nranks < 2 or self._local_rank == 0:
            # update metrics
            for metric in self._metrics:
                metric.update(im_id_dict, outs)

        sample_num += im_id_dict['im_id'].numpy().shape[0]

        self.status['sample_num'] = sample_num
        self.status['cost_time'] = time.time() - tic
        if self._nranks < 2 or self._local_rank == 0:
            # accumulate metric to log out
            for metric in self._metrics:
                metric.accumulate()
                metric.log()
        self._compose_callback.on_epoch_end(self.status)
        # reset metric states for metric may performed multiple times
        self._reset_metrics()

    def evaluate(self):
        with paddle.no_grad():
            self._eval_with_loader(self.loader)

    def predict(self,
                images,
                draw_threshold=0.5,
                output_dir='output',
            ):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        self.dataset.set_images(images)
        loader = create('TestReader')(self.dataset, 0)

        imid2path = self.dataset.get_imid2path()

        def setup_metrics_for_loader():
            # mem
            metrics = copy.deepcopy(self._metrics)
            mode = self.mode
            save_prediction_only = self.cfg[
                'save_prediction_only'] if 'save_prediction_only' in self.cfg else None
            output_eval = self.cfg[
                'output_eval'] if 'output_eval' in self.cfg else None

            # modify
            self.mode = '_test'
            self.cfg['save_prediction_only'] = True
            self.cfg['output_eval'] = output_dir
            self.cfg['imid2path'] = imid2path
            self._init_metrics()

            # restore
            self.mode = mode
            self.cfg.pop('save_prediction_only')
            if save_prediction_only is not None:
                self.cfg['save_prediction_only'] = save_prediction_only

            self.cfg.pop('output_eval')
            if output_eval is not None:
                self.cfg['output_eval'] = output_eval

            self.cfg.pop('imid2path')

            _metrics = copy.deepcopy(self._metrics)
            self._metrics = metrics

            return _metrics
        
        metrics = setup_metrics_for_loader()