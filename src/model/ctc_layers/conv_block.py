from typing import TypeVar

from torch import Tensor, nn

TModule = TypeVar("TModule", bound=nn.Module)


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        activation: TModule = nn.ReLU,
        norm: TModule = nn.InstanceNorm1d,
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.act = activation()
        self.norm = norm(out_channels)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(self.act(self.conv(x)))
