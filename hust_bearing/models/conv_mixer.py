# pylint: disable=not-callable, too-many-ancestors
from typing import Any

from mlp_mixer_pytorch import MLPMixer
from torch import nn
from torch.nn.functional import gelu

from hust_bearing.models import Classifier


class ConvMixer(Classifier):
    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes)
        self.conv = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        self.batch_norm = nn.BatchNorm2d(num_features=64)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.mixer = MLPMixer(
            image_size=32,
            channels=64,
            patch_size=8,
            dim=256,
            depth=8,
            num_classes=num_classes,
        )

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        inputs, *_ = args
        conv = self.batch_norm(gelu(self.conv(inputs)))
        pool = self.pool(conv)
        return self.mixer(pool)
