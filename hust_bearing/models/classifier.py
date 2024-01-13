from abc import ABCMeta, abstractmethod
from typing import Any

import lightning as pl
from lightning.pytorch.utilities.types import STEP_OUTPUT
from torch import nn
from torchmetrics.classification import MulticlassAccuracy


class Classifier(pl.LightningModule, metaclass=ABCMeta):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes)
        self.test_acc = MulticlassAccuracy(num_classes)
        self.val_acc = MulticlassAccuracy(num_classes)

    def training_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        batch, *_ = args
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        self.train_acc(outputs, targets)
        self.log_dict({"train_loss": loss, "train_acc": self.train_acc}, prog_bar=True)
        return loss

    def validation_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        batch, *_ = args
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        self.val_acc(outputs, targets)
        self.log_dict({"val_loss": loss, "val_acc": self.val_acc}, prog_bar=True)
        return None

    def test_step(self, *args: Any, **kwargs: Any) -> STEP_OUTPUT:
        batch, *_ = args
        inputs, targets = batch
        outputs = self(inputs)
        loss = self.loss(outputs, targets)
        self.test_acc(outputs, targets)
        self.log_dict({"test_loss": loss, "test_acc": self.test_acc}, prog_bar=True)
        return None

    def predict_step(self, *args: Any, **kwargs: Any) -> Any:
        batch, *_ = args
        inputs, _ = batch
        return self(inputs).argmax(dim=1)

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass
