import os
import sys

# add python path of PadleDetection to sys.path
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 2)))
sys.path.insert(0, parent_path)

import paddle
from ppdet.utils.checkpoint import load_weight, load_pretrain_weight
from ppdet.modeling.backbones import ResNet

from pprint import pprint

class NameAdapter(object):
    """Fix the backbones variable names for pretrained weight"""

    def __init__(self):
        super(NameAdapter, self).__init__()
        pass

    def fix_conv_norm_name(self, name):
        if name == "conv1":
            bn_name = "bn_" + name
        else:
            bn_name = "bn" + name[3:]

        return bn_name

    def fix_shortcut_name(self, name):
        return name

    def fix_bottleneck_name(self, name):
        conv_name1 = name + "_branch2a"
        conv_name2 = name + "_branch2b"
        conv_name3 = name + "_branch2c"
        shortcut_name = name + "_branch1"
        return conv_name1, conv_name2, conv_name3, shortcut_name

    def fix_basicblock_name(self, name):
        conv_name1 = name + "_branch2a"
        conv_name2 = name + "_branch2b"
        shortcut_name = name + "_branch1"
        return conv_name1, conv_name2, shortcut_name

    def fix_layer_warp_name(self, stage_num, count, i):
        name = 'res' + str(stage_num)
        if count > 10 and stage_num == 4:
            if i == 0:
                conv_name = name + "a"
            else:
                conv_name = name + "b" + str(i)
        else:
            conv_name = name + chr(ord("a") + i)
        return conv_name

    def fix_c1_stage_name(self):
        return "conv1"

param_state_dict = paddle.load("./pretrained_weights/SE_ResNeXt50_vd_32x4d_pretrained.pdparams")

"""
ResNet:
  depth: 50
  variant: d
  return_idx: [1, 2, 3]
  dcn_v2_stages: [3]
  freeze_at: -1
  freeze_norm: false
  norm_decay: 0.
"""
ResNeXT = ResNet(
    depth = 50,
    ch_in = 128,
    groups = 32,
    base_width = 4,
    variant = 'd',
    return_idx = [1, 2, 3],
    freeze_at = -1,
    freeze_norm = False,
    norm_decay = 0,
    std_senet = True
)

print(dir(ResNeXT))
print(ResNeXT._model_type)

new_weights = {}
unpair_keys = {}
cnt = 0
for (name, params), (name_, params_) in zip(ResNeXT.named_parameters(), param_state_dict.items()):
    print(f"{name}: {params.shape}")
    print(f"{name_}: {params_.shape}")
    if params.shape != params_.shape:
        unpair_keys[name] = name_
    if cnt > 0:
        assert params.shape == params_.shape, f"params: {params.shape}, params_: {params_.shape}"
    new_weights[name] = params_
    cnt += 1

print(cnt)

paddle.save(new_weights, os.path.join("./pretrained_weights/", "SE_ResNeXt50_32x4d" + ".pdparams"))
pprint(unpair_keys)