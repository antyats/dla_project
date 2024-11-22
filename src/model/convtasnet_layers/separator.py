import torch
import torch.nn as nn

from src.model.convtasnet_layers.conv_block import Conv1dBlock

class TCN(nn.Module):
    def __init__(
            self, 
            in_channels: int = 512, 
            out_channels: int = 512, 
            kernel_size: int = 8,
            padding: int = 4,
            dilation: int = 1,
            separator_size: int = 1,
            separator_count_blocks: int = 1,
        ):

        super(TCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.size = separator_size
        self.count_blocks = separator_count_blocks
        
        self.layer_norm = nn.LayerNorm(self.out_channels)
        self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)        

        self.separator = nn.ModuleList()

        for i in range(self.size - 1):
            for j in range(self.count_blocks - 1):
                conv_block = Conv1dBlock(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, dilation=2 ** j, padding=2 ** j, stride=self.stride, skip=False)
                self.separator.append(conv_block)
        
        conv_block = Conv1dBlock(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, dilation=2 ** (self.self.count_blocks - 1), padding=2 ** (self.self.count_blocks - 1), stride=self.stride, skip=True)
        self.separator.append(conv_block)

        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv1d(in_channels=self.out_channels, out_channels=self.in_channels * 2, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        out = self.layer_norm(x)
        out = self.conv1(out)

        skip_all = 0
        for i in range(len(self.separator) - 1):
            residual, skip = self.separator[i](out)
            out += residual
            skip_all += skip

        out += self.separator[len(self.separator)]
        out += skip_all
            
        out = self.prelu1(out)
        out = self.conv2(out)
        out = self.sigmoid(out)
        
        return out


