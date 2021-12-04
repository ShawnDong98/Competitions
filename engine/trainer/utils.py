import functools
import os
import sys
import time
import random

from getpass import getuser
from socket import gethostname

import torch
import torch.distributed as dist

import numpy as np

from ..utils.misc import is_str

def get_host_info():
    return '{}@{}'.format(getuser(), gethostname())


def get_dist_info():
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def master_only(func):
    """
        only rank == 0, the func activate
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rank, _ = get_dist_info()
        if rank == 0:
            return func(*args, **kwargs)

    return wrapper

def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())

def obj_from_dict(info, parrent=None, default_args=None):
    """Initialize an object from dict.
    The dict must contain the key "type", which indicates the object type, it
    can be either a string or type, such as "list" or ``list``. Remaining
    fields are treated as the arguments for constructing the object.
    Args:
        info (dict): Object types and arguments.
        module (:class:`module`): Module which may containing expected object
            classes.
        default_args (dict, optional): Default arguments for initializing the
            object.
    Returns:
        any type: Object built from the dict.
    """
    assert isinstance(info, dict) and 'name' in info
    assert isinstance(default_args, dict) or default_args is None

    args = info.copy()
    obj_type = args.pop('name')
    if is_str(obj_type):
        if parrent is not None:
            obj_type = getattr(parrent, obj_type)
        else:
            obj_type = sys.modules[obj_type]
    elif not isinstance(obj_type, type):
        raise TypeError('type must be a str or valid type, but got {}'.format(type(obj_type)))

    if default_args is not None:
        for name, value in default_args.items():
            args.setdefault(name, value)

    return obj_type(**args)


def seed_everything(
    seed = 3407,
    deterministic = False, 
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
