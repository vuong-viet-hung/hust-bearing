import torch
import torch.nn as nn

from hust_bearing.models.core import register_model


@register_model("lenet5")
class LeNet5(nn.Module):
    def __init__(self, num_classes: int) -> None:
        super(LeNet5, self).__init__()

        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(16 * 13 * 13, 120)
        self.dropout1 = nn.Dropout(0.75)

        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.75)

        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        conv1 = self.relu1(self.pool1(self.conv1(inputs)))
        conv2 = self.relu2(self.pool2(self.conv2(conv1)))
        flatten = self.flatten(conv2)
        fc1 = self.dropout1(self.fc1(flatten))
        fc2 = self.dropout2(self.fc2(fc1))
        return self.fc3(fc2)
