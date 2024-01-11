from typing import Literal

import lightning as pl
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy

from hust_bearing.models.lenet5 import LeNet5
from hust_bearing.models.conv_mixer import ConvMixer


ModelName = Literal["lenet5", "conv_mixer"]


MODEL_CLASSES: dict[ModelName, type[nn.Module]] = {
    "lenet5": LeNet5,
    "conv_mixer": ConvMixer,
}


class Classifier(pl.LightningModule):
    def __init__(self, model: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.model = model
        self.loss = nn.CrossEntropyLoss()
        self.train_acc = MulticlassAccuracy(num_classes)
        self.test_acc = MulticlassAccuracy(num_classes)
        self.val_acc = MulticlassAccuracy(num_classes)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        self.train_acc(outputs, targets)
        self.log_dict({"train_loss": loss, "train_acc": self.train_acc}, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        self.val_acc(outputs, targets)
        self.log_dict({"val_loss": loss, "val_acc": self.val_acc}, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self.loss(outputs, targets)
        self.test_acc(outputs, targets)
        self.log_dict({"test_loss": loss, "test_acc": self.test_acc}, prog_bar=True)

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        inputs, _ = batch
        return self.model(inputs).argmax(dim=1)


def classifier(name: ModelName, num_classes: int) -> Classifier:
    model = MODEL_CLASSES[name](num_classes)
    return Classifier(model, num_classes)
