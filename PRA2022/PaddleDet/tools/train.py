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

from ppdet.core.workspace import load_config, merge_config
from ppdet.engine import Trainer, init_parallel_env, set_random_seed, init_fleet_env

from ppdet.utils.cli import ArgsParser, merge_args
import ppdet.utils.check as check
from ppdet.utils.logger import setup_logger
logger = setup_logger('train')

from ppdet.core.workspace import create

def parse_args():
    parser = ArgsParser()
    parser.add_argument(
        "--eval",
        action='store_true',
        default=False,
        help="Whether to perform evaluation in train")
    parser.add_argument(
        "-r", "--resume", default=None, help="weights path for resume")
    parser.add_argument(
        "--enable_ce",
        action='store_true',
        default=False,
        help="If set True, enable continuous evaluation job."
        "This flag is only used for internal test.")
    parser.add_argument(
        "--classwise",
        action="store_true",
        help="whether per-category AP and draw P-R Curve or not.")
    parser.add_argument(
        "--amp",
        action='store_true',
        default=False,
        help="Enable auto mixed precision training.")
    parser.add_argument(
        "--fleet", action='store_true', default=False, help="Use fleet or not")

    args = parser.parse_args()
    return args


def run(FLAGS, cfg):
    # init fleet environment
    if cfg.fleet:
        init_fleet_env(cfg.get('find_unused_parameters', False))
    else:
        # init parallel environment if nranks > 1
        init_parallel_env()

    if FLAGS.enable_ce:
        set_random_seed(0)

    # build trainer
    trainer = Trainer(cfg, mode='train')

    # load weights
    if FLAGS.resume is not None:
        trainer.resume_weights(FLAGS.resume)
    elif 'pretrain_weights' in cfg and cfg.pretrain_weights:
        trainer.load_weights(cfg.pretrain_weights)

    # training
    trainer.train(FLAGS.eval)


def main():
    FLAGS = parse_args()
    print(FLAGS)
    cfg = load_config(FLAGS.config)
    merge_args(cfg, FLAGS)

    merge_config(FLAGS.opt)

    # disable npu in config by default
    if 'use_npu' not in cfg:
        cfg.use_npu = False

    # disable xpu in config by default
    if 'use_xpu' not in cfg:
        cfg.use_xpu = False

    if cfg.use_gpu:
        place = paddle.set_device('gpu')
    elif cfg.use_npu:
        place = paddle.set_device('npu')
    elif cfg.use_xpu:
        place = paddle.set_device('xpu')
    else:
        place = paddle.set_device('cpu')

    # FIXME: Temporarily solve the priority problem of FLAGS.opt
    merge_config(FLAGS.opt)
    check.check_config(cfg)
    check.check_gpu(cfg.use_gpu)
    check.check_npu(cfg.use_npu)
    check.check_version()

    run(FLAGS, cfg)

# trainer = Trainer(cfg, mode='eval')

# # trainer.train(validate=False)
# trainer.evaluate()


# # mode = "train"
# # capital_mode = mode.capitalize()
# # dataset = cfg['{}Dataset'.format(capital_mode)] = create(
# #                 '{}Dataset'.format(capital_mode))()

# # loader = create('{}Reader'.format(capital_mode))(
# #                 dataset, 
# #                 cfg.worker_num
# #             )
# # print(next(loader))
# # # print(next(iter(loader)))

# # model = create(cfg.architecture)

# # print(model)

if __name__ == "__main__":
    main()