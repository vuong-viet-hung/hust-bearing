import torch
import lightning as pl
from torch import nn
from torch.nn.functional import max_pool2d, relu
from torchmetrics.classification import MulticlassAccuracy


class LeNet5Clf(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.fc1 = nn.Linear(in_features=16 * 13 * 13, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        conv1 = max_pool2d(relu(self.conv1(inputs)), kernel_size=2)
        conv2 = max_pool2d(relu(self.conv2(conv1)), kernel_size=2)
        fc1 = relu(self.fc1(conv2.flatten(start_dim=1)))
        fc2 = relu(self.fc2(fc1))
        return relu(self.fc3(fc2))


class LeNet5(pl.LightningModule):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.model = LeNet5Clf(num_classes)
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
