import torch
import torch.nn as nn
import torch.nn.functional as F
class custommodel(nn.Module):
    def __init__(self, num_classes=5):
        super(custommodel, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.td_fc = nn.Linear(256, num_classes)
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        b, c, h, t = x.size()
        x = x.permute(0, 3, 1, 2)
        x = x.reshape(b, t, c*h)
        x, _ = self.lstm(x)
        x = self.td_fc(x)
        x = torch.mean(x, dim=1)
        return x
