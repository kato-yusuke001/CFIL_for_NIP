import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from .base import BaseNetwork

torch.backends.cudnn.benchmark = True
class FNN(BaseNetwork):
    def __init__(self):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(in_features=16*2, out_features=128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=128, out_features=64),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=64, out_features=32),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=32, out_features=3)
            )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        # print(x.shape)
        x = self.classifier(x)
        return x
