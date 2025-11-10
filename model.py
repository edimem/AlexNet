import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.ReLU = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=96, kernel_size=11, stride=4)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, padding=2)
        self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=3, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(6 * 6 * 256, 4096)
        self.fc2 = nn.Linear(4096, 4096)
        self.fc3 = nn.Linear(4096, 10)

    def forward(self, x):
        x = self.ReLU(self.conv1(x))
        x = self.pool1(x)
        x = self.ReLU(self.conv2(x))
        x = self.pool2(x)
        x = self.ReLU(self.conv3(x))
        x = self.ReLU(self.conv4(x))
        x = self.ReLU(self.conv5(x))
        x = self.pool3(x)
        x = self.flatten(x)
        x = self.ReLU(self.fc1(x))
        x = F.dropout(x, 0.5)
        x = self.ReLU(self.fc2(x))
        x = F.dropout(x, 0.5)
        x = self.fc3(x)

        return x


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AlexNet().to(device)
    print(summary(model, (1, 227, 227)))
