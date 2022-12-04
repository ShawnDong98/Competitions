import os
import random

from getpass import getuser
from socket import gethostname

def get_host_info():
    return '{}@{}'.format(getuser(), gethostname())

import torch
import torch.distributed as dist

import numpy as np


def get_dist_info():
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def seed_everything(
    seed = 3407,
    deterministic = True, 
    use_rank_shift = False
):
    """Set random seed.
    Args:
        seed (int): Seed to be used, default seed 3407, from the paper
        Torch. manual_seed (3407) is all you need: On the influence of random seeds in deep learning architectures for computer vision[J]. arXiv preprint arXiv:2109.08203, 2021.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
        rank_shift (bool): Whether to add rank number to the random seed to
            have different random seed in different threads. Default: False.
    """
    if use_rank_shift:
        rank, _ = get_dist_info()
        seed += rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False