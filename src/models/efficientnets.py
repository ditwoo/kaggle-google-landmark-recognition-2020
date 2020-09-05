import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet


class EfficientNetEncoder(nn.Module):
    def __init__(self, base: str, num_classes: int = 1, bias: bool = True):
        super().__init__()

        self.base = EfficientNet.from_pretrained(base)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.output_filter = self.base._fc.in_features
        self.classifier = nn.Linear(self.output_filter, num_classes, bias=bias)

    def forward(self, batch):
        x = self.base.extract_features(batch)
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x = self.classifier(x)
        return x
