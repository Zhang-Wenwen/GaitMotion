import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

class CNNNet(nn.Module):
    def __init__(self, num_classes, output_len, output_size):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=(2, 1), stride=(2, 1), padding=(0, 0)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16))  # 2000*6 -> 1000*6  output_len/2
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(5, 1), stride=(5, 1), padding=(0, 0)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(32))  # 200*6 output_len/10
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=(5, 1), stride=(5, 1), padding=(0, 0)),
            nn.LeakyReLU(),
            nn.BatchNorm2d(16),  # 40*6  output_len/50
            nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)))  # 20*3  output_len/100

        self.fc1 = nn.Linear(16 * int(output_len/100) * output_size, 64)  # 16:chanel
        self.fc1_bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, num_classes)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x[:, None, :, :]  
        x = self.conv1(x)  
        x = self.conv2(x)
        x = self.conv3(x)
        x = torch.flatten(x, 1)  
        x = self.fc1_bn(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        # x = self.tanh(x)
        # x = F.relu(x)
        x = F.softplus(x)
        # x = self.maxout(x)
        return x
