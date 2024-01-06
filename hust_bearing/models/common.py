import lightning as pl
import torch
from torch import nn
from torchmetrics.classification import MulticlassAccuracy


class Classifier(pl.LightningModule):
    def __init__(self, model: nn.Module, num_classes: int) -> None:
        super().__init__()
        self.model = model
        self._loss = nn.CrossEntropyLoss()
        self._train_acc = MulticlassAccuracy(num_classes)
        self._test_acc = MulticlassAccuracy(num_classes)
        self._val_acc = MulticlassAccuracy(num_classes)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self._loss(outputs, targets)
        self._train_acc(outputs, targets)
        self.log_dict({"train_loss": loss, "train_acc": self._train_acc}, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self._loss(outputs, targets)
        self._val_acc(outputs, targets)
        self.log_dict({"val_loss": loss, "val_acc": self._val_acc}, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> None:
        inputs, targets = batch
        outputs = self.model(inputs)
        loss = self._loss(outputs, targets)
        self._test_acc(outputs, targets)
        self.log_dict({"test_loss": loss, "test_acc": self._test_acc}, prog_bar=True)

    def predict_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        inputs, _ = batch
        return self.model(inputs).argmax(dim=1)
