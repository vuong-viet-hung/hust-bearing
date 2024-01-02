import torch
import lightning as pl
from torch import nn
from mlp_mixer_pytorch import MLPMixer
from torch.nn.functional import gelu
from torchmetrics.classification import MulticlassAccuracy


class ConvMixerClf(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
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

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        conv = self.batch_norm(gelu(self.conv(inputs)))
        pool = self.pool(conv)
        return self.mixer(pool)


class ConvMixer(pl.LightningModule):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.model = ConvMixerClf(num_classes)
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
