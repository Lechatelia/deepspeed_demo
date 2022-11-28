import torch
from torch import nn


class ToyModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(10, 13)
        self.linear2 = nn.Linear(13, 512)
        self.linear3 = nn.Linear(512, 10)
        self.relu = nn.ReLU()
        self.temp = nn.Parameter(torch.ones([]) * 0.05)

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        x = self.relu(self.linear3(x))
        return x * self.temp
