import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class PetfinderModel(nn.Module):
    def __init__(
        self, 
        cfg
    ):
        super(PetfinderModel, self).__init__()
        self._cfg = cfg
        self.backbone = timm.create_model(self._cfg.model.name, pretrained=True, num_classes=self._cfg.model.output_dim)

    def forward(self, x):
        out = self.backbone(x)
        return out



