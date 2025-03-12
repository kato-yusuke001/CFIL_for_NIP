import torch
# import torchvision
import torch.nn as nn
import torch.optim as optim
# from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .base import BaseNetwork

torch.backends.cudnn.benchmark = True
class ABN128(BaseNetwork):
    def __init__(self, output_size=128):
        super().__init__()


        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2), # 64
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # 32
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), #16
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),


        )
        self.regression = nn.Sequential(
            nn.Linear(in_features=128*16*16, out_features=512),
            nn.Linear(in_features=512, out_features=128),
            nn.Linear(in_features=128, out_features=32),
            nn.Linear(in_features=32, out_features=output_size)
            )

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=6, kernel_size=1, padding=0),
        )
        self.bn_att = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

        self.wgp = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=output_size, kernel_size=16, padding=0),
            nn.Tanh()
        )
    

    def forward(self, x):
        x = self.features(x)

        ax = self.attention(x)
        att = torch.sum(ax, dim=1, keepdim=True)
        ax = self.wgp(ax)
    
        rx = x * att
        rx = rx + x
        rx = rx.reshape(rx.size(0), -1)
        rx = self.regression(rx)
        ax = ax.reshape(ax.size(0), -1)
        return rx, ax, att
    

class ABN256(BaseNetwork):
    def __init__(self, output_size = 256):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2), # 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # 64
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), #32
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),


        )
        self.regression = nn.Sequential(
            nn.Linear(in_features=128*32*32, out_features=512),
            nn.Linear(in_features=512, out_features=128),
            nn.Linear(in_features=128, out_features=32),
            nn.Linear(in_features=32, out_features=output_size)
            )

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=6, kernel_size=1, padding=0),
        )
        self.bn_att = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

        self.wgp = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=output_size, kernel_size=32, padding=0),
            nn.Tanh()
        )
    

    def forward(self, x):
        x = self.features(x)

        ax = self.attention(x)
        att = torch.sum(ax, dim=1, keepdim=True)
        ax = self.wgp(ax)
    
        rx = x * att
        rx = rx + x
        rx = rx.reshape(rx.size(0), -1)
        rx = self.regression(rx)
        ax = ax.reshape(ax.size(0), -1)
        return rx, ax, att
    
class CNN256(BaseNetwork):
    def __init__(self, output_size = 6):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(inplace=True), 
            nn.MaxPool2d(kernel_size=2), # 128
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), # 64
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2), #32
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),


        )
        self.regression = nn.Sequential(
            nn.Linear(in_features=128*32*32, out_features=512),
            nn.Linear(in_features=512, out_features=128),
            nn.Linear(in_features=128, out_features=32),
            nn.Linear(in_features=32, out_features=output_size)
            )

        self.attention = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=6, kernel_size=1, padding=0),
        )
        self.bn_att = nn.BatchNorm2d(1)
        self.sigmoid = nn.Sigmoid()

        self.wgp = nn.Sequential(
            nn.Conv2d(in_channels=6, out_channels=output_size, kernel_size=32, padding=0),
            nn.Tanh()
        )
    

    def forward(self, x):
        rx = self.features(x)

        rx = rx.reshape(rx.size(0), -1)
        rx = self.regression(rx)
        return rx


