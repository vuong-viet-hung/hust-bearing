import torch
from torch import nn

from hust_bearing.models.core import register_model


@register_model("lenet5")
class LeNet5(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        # 1st layer
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        # 2nd layer
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        # Flatten layer
        self.flatten = nn.Flatten()
        # 3rd layer
        self.fc3 = nn.Linear(16 * 13 * 13, 120)
        self.tanh3 = nn.Tanh()
        # 4th layer
        self.fc4 = nn.Linear(120, 84)
        self.tanh4 = nn.Tanh()
        # 5th layer
        self.fc5 = nn.Linear(84, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        conv1 = self.pool1(self.tanh1(self.conv1(inputs)))
        conv2 = self.pool2(self.tanh2(self.conv2(conv1)))
        flatten = self.flatten(conv2)
        fc3 = self.tanh3(self.fc3(flatten))
        fc4 = self.tanh4(self.fc4(fc3))
        return self.fc5(fc4)
