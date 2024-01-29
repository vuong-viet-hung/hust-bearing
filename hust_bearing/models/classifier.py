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
        self.train_acc = MulticlassAccuracy(num_classes)
        self.test_acc = MulticlassAccuracy(num_classes)
        self.val_acc = MulticlassAccuracy(num_classes)

    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        (inputs, targets), *_ = args
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        self.train_acc(outputs, targets)
        self.log_dict({"train_loss": loss, "train_acc": self.train_acc}, prog_bar=True)
        return loss

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        (inputs, targets), *_ = args
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        self.val_acc(outputs, targets)
        self.log_dict({"val_loss": loss, "val_acc": self.val_acc}, prog_bar=True)

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        (inputs, targets), *_ = args
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        self.test_acc(outputs, targets)
        self.log_dict({"test_loss": loss, "test_acc": self.test_acc}, prog_bar=True)

    def predict_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        (inputs, _), *_ = args
        return self.model(inputs).argmax(dim=1)


class ConvMixer(Classifier):
    def __init__(self, num_classes: int) -> None:
        model = conv_mixer.ConvMixer(num_classes)
        super().__init__(model, num_classes)
