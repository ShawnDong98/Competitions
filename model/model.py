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
        self.backbone = timm.create_model(self._cfg.model.name, pretrained=True, num_classes=0)
        self.num_features = self.backbone.num_features

        self.dense1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.num_features, self._cfg.model.feature_dim)
        )
        self.dense2 = nn.Linear(self._cfg.model.feature_dim, self._cfg.model.output_dim)

    def forward(self, x):
        x1 = self.backbone(x)
        feature  = self.dense1(x1)
        out = self.dense2(feature)
        return out



