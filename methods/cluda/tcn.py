"""Behavioral reimplementation of the causal TCN used by official CLUDA."""

import torch
from torch import nn


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = int(chomp_size)

    def forward(self, x):
        if self.chomp_size == 0:
            return x.contiguous()
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.padding = int(padding)
        self.dilation = int(dilation)
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2,
        )
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        output = self.net(x)
        residual = x if self.downsample is None else self.downsample(x)
        return self.relu(output + residual)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, stride=1, dilation_factor=2, dropout=0.2):
        super().__init__()
        layers = []
        for level, output_channels in enumerate(num_channels):
            dilation = dilation_factor ** level
            input_channels = num_inputs if level == 0 else num_channels[level - 1]
            layers.append(TemporalBlock(
                input_channels, output_channels, kernel_size, stride, dilation,
                (kernel_size - 1) * dilation, dropout,
            ))
        self.network = nn.Sequential(*layers)
        self.num_channels = tuple(num_channels)

    def forward(self, x):
        return self.network(x)
