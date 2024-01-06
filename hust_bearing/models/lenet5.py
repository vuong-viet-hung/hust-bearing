import torch
from torch import nn
from torch.nn.functional import max_pool2d, relu

from hust_bearing.models.common import Classifier


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


class LeNet5(Classifier):
    def __init__(self, num_classes: int) -> None:
        model = LeNet5Clf(num_classes)
        super().__init__(model, num_classes)
