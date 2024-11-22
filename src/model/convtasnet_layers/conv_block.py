import torch
import torch.nn as nn

class Conv1dBlock(nn.Module):
    def __init__(
            self, 
            in_channels: int = 512, 
            out_channels: int = 512, 
            kernel_size: int = 8,
            padding: int = 4,
            dilation: int = 1,
            skip: bool = False,
        ):

        super(Conv1dBlock, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.dilation = dilation
        self.skip = skip
        
        self.conv1 = nn.Conv1d(in_channels=self.in_channels, out_channels=self.out_channels, kernel_size=1)        
        self.norm1 = nn.BatchNorm1d(self.out_channels)
        self.prelu1 = nn.PReLU()
        self.dconv = nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=self.kernel_size, 
                                padding=self.padding, dilation=self.dilation, groups=self.out_channels)
        
        self.norm2 = nn.BatchNorm1d(out_channels)
        self.prelu2 = nn.PReLU()
        
        self.conv2 = nn.Conv1d(in_channels=self.out_channels, out_channels=self.out_channels, kernel_size=1)

        self.skip_connection = nn.Conv1d(in_channels=self.out_channels, out_channels=self.in_channels, kernel_size=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.prelu1(out)
        out = self.norm1(out)
        out = self.dconv(out)
        out = self.prelu2(out)
        out = self.norm2(out)

        if self.skip:
            return self.conv2(out), self.skip_connection(out)   
        return self.conv2(out)
            