import torch
from torch import Tensor, nn

from src.model.layers.conv_block import ConvBlock


class InjectionSum(nn.Module):
    def __init__(self, conv_dim: int = 512, kernel_size: int = 5):
        """
        Args:
            conv_dim (int, optional): number of channels.
            kernel_size (int, optional): kernel size of convolutional layers.
        """
        super().__init__()
        self.score_conv = nn.Sequential(
            ConvBlock(
                conv_dim,
                conv_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
                groups=conv_dim,
                activation=nn.Identity,
            ),
            nn.Sigmoid(),
        )
        self.sum_conv = ConvBlock(
            conv_dim,
            conv_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=conv_dim,
            activation=nn.Identity,
        )
        self.x_conv = ConvBlock(
            conv_dim,
            conv_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=conv_dim,
            activation=nn.Identity,
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor. Shape: (batch_size, conv_dim, seq_len_x)
            y (Tensor): input tensor. Shape: (batch_size, conv_dim, seq_len_y)

        Returns:
            out (Tensor): tensor, processed by InjectionSum layer. Shape: (batch_size, conv_dim, seq_len_x)
        """
        y = torch.nn.functional.interpolate(y, x.shape[-1])
        scores = self.score_conv(y)
        additive = self.sum_conv(y)
        return self.x_conv(x) * scores + additive
