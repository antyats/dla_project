from typing import TypeVar

from torch import Tensor, nn

from src.model.layers.global_layer_norm import GlobalLayerNorm

TModule = TypeVar("TModule", bound=nn.Module)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        dilation: int = 1,
        padding: int = 0,
        activation: TModule = nn.ReLU,
        norm: TModule = GlobalLayerNorm,
        groups: int = 1,
        bias=True,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            dilation=dilation,
            bias=bias,
        )
        self.act = activation()
        self.norm = norm(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): Input tensor of shape (batch_size, in_channels, seq_len).

        Returns:
            out (Tensor): Output tensor of shape (batch_size, out_channels, seq_len),
        """
        return self.norm(self.act(self.conv(x)))
