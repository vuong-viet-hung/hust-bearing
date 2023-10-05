import torch


class LeNet5(torch.nn.Module):
    def __init__(self, n_classes):
        super(LeNet5, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 6, kernel_size=5, stride=1)
        self.avgpool1 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(6, 16, kernel_size=5, stride=1)
        self.avgpool2 = torch.nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, n_classes)

    def forward(self, x):
        x = self.avgpool1(torch.relu(self.conv1(x)))
        x = self.avgpool2(torch.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x
