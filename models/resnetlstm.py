import torch
import torch.nn as nn
import torchvision.models as models

class ResNet_LSTM_TDFC(nn.Module):
    def __init__(self, num_classes=6):
        super(ResNet_LSTM_TDFC, self).__init__()

        self.resnet = models.resnet18(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=False
        )

        self.tdfc = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )

        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.mean(x, dim=2)
        x = x.permute(0, 2, 1)
        x, _ = self.lstm(x)
        x = self.tdfc(x)
        x = x[:, -1, :]
        x = self.fc(x)
        return x