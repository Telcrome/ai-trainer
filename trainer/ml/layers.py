from typing import List

import torch
import torch.nn as nn


class ConvGRUCell(nn.Module):
    """
    Adapted from https://github.com/happyjin/ConvGRU-pytorch
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize the ConvLSTM cell
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        """
        super(ConvGRUCell, self).__init__()
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.bias = bias
        self.h = None

        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2 * self.hidden_dim,  # for update_gate,reset_gate respectively
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=self.bias)

        self.conv_can = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                  out_channels=self.hidden_dim,  # for candidate neural memory
                                  kernel_size=kernel_size,
                                  padding=self.padding,
                                  bias=self.bias)

    def reset_hidden(self):
        self.h = None

    def init_hidden(self, input_batch: torch.Tensor):
        self.h = torch.zeros(
            input_batch.size()[0],
            self.hidden_dim,
            input_batch.size()[2],
            input_batch.size()[3],
            dtype=input_batch.dtype,
            device=input_batch.device)

    def forward(self, input_tensor: torch.Tensor):
        if self.h is None:
            self.init_hidden(input_tensor)

        # Hot fix for sequences where every input and output has the same size, but the intra-sequence size differs
        if input_tensor.size()[2] != self.h.size()[2] or input_tensor.size()[3] != self.h.size()[3]:
            self.init_hidden(input_tensor)

        combined = torch.cat([input_tensor, self.h], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate * self.h], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        self.h = (update_gate + (-1)) * self.h + update_gate * cnm
        return self.h


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        # if self.downsample:
        #     residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out
