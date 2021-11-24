import os
import os.path as osp
import sys
from pathlib import Path

import six

from .misc import is_str

if sys.version_info <= (3, 3):
    FileNotFoundError = IOError
else:
    FileNotFoundError = FileNotFoundError


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    if six.PY3:
        os.makedirs(dir_name, mode=mode, exist_ok=True)
    else:
        if not osp.isdir(dir_name):
            os.makedirs(dir_name, mode=mode)

def symlink(src, dst, overwrite=True, **kwargs):
    if osp.exists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)