# pylint: disable=too-many-ancestors
from abc import ABCMeta
from typing import Any

import lightning as pl
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

from hust_bearing.models import conv_mixer


class Classifier(pl.LightningModule, metaclass=ABCMeta):
    def __init__(self, model, num_classes: int) -> None:
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.accuracy = MulticlassAccuracy(num_classes)

    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        (inputs, targets), *_ = args
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        self.accuracy(outputs, targets)
        self.log_dict({"train_loss": loss, "train_acc": self.accuracy}, prog_bar=True)
        return loss

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        (inputs, targets), *_ = args
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        self.accuracy(outputs, targets)
        self.log_dict({"val_loss": loss, "val_acc": self.accuracy}, prog_bar=True)

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        (inputs, targets), *_ = args
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        self.accuracy(outputs, targets)
        self.log_dict({"test_loss": loss, "test_acc": self.accuracy}, prog_bar=True)

    def predict_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        (inputs, _), *_ = args
        return self.model(inputs).argmax(dim=1)


class ConvMixer(Classifier):
    def __init__(self, num_classes: int) -> None:
        model = conv_mixer.ConvMixer(num_classes)
        super().__init__(model, num_classes)
