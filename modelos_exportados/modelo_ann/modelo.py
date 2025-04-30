# Red neuronal
import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.hidden1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)

        self.hidden2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.hidden3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)

        self.dropout = nn.Dropout(0.2)
        self.output = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.nn.functional.leaky_relu(self.bn1(self.hidden1(x)))
        x = self.dropout(x)
        x = torch.nn.functional.leaky_relu(self.bn2(self.hidden2(x)))
        x = self.dropout(x)
        x = torch.nn.functional.leaky_relu(self.bn3(self.hidden3(x)))
        x = self.output(x)
        return x
