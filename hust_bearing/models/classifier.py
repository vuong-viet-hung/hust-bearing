from typing import Any

import lightning as pl
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy


class Classifier(pl.LightningModule):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes)
        self.test_acc = self.train_acc.clone()
        self.val_acc = self.train_acc.clone()

    def training_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        (inputs, targets), *_ = args
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        self.train_acc(outputs, targets)
        self.log_dict({"train_loss": loss, "train_acc": self.train_acc}, prog_bar=True)
        return loss

    def validation_step(self, *args: Any, **kwargs: Any) -> None:
        (inputs, targets), *_ = args
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        self.train_acc(outputs, targets)
        self.log_dict({"val_loss": loss, "val_acc": self.train_acc}, prog_bar=True)

    def test_step(self, *args: Any, **kwargs: Any) -> None:
        (inputs, targets), *_ = args
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        self.train_acc(outputs, targets)
        self.log_dict({"test_loss": loss, "test_acc": self.train_acc}, prog_bar=True)

    def predict_step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        (inputs, _), *_ = args
        return self(inputs).argmax(dim=1)
