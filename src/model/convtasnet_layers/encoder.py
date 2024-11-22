import torch
import torch.nn as nn

from model.convtasnet_layers.conv_block import Conv1dBlock

class Encoder(nn.Module):
    def __init__(
            self, 
            in_channels: int = 1,
            out_channels: int = 512,
            kernel_size: int = 8,
            padding: int = 4,
            stride: int = 4,
        ):
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.conv1d = nn.Conv1d(in_channels=self.in_channels, out_channels=1, kernel_size=self.kernel_size, stride=self.stride, padding=self.padding, bias=False)

    def forward(self, x):
        return self.conv1d(x)