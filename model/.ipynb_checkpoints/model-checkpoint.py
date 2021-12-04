import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class FackFaceDetModel(nn.Module):
    def __init__(
        self, 
        cfg
    ):
        super(FackFaceDetModel, self).__init__()
        self._cfg = cfg
        self.backbone = timm.create_model(self._cfg.model.name, pretrained=True, num_classes=0)
        num_features = self.backbone.num_features
        self.fc = nn.Sequential(
            nn.Linear(num_features, num_features*4),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(num_features*4, self._cfg.model.output_dim)
        )

    def forward(self, x):
        feature = self.backbone(x)
        out = self.fc(feature)
        return out