from torch import Tensor, nn

from src.model.layers import ConvBlock


class TDFMaskGenerator(nn.Module):
    def __init__(self, conv_dim: int = 512):
        super().__init__()
        self.x_conv = nn.Sequential(
            nn.PReLU(),
            ConvBlock(
                in_channels=conv_dim,
                out_channels=conv_dim,
                kernel_size=1,
                groups=conv_dim,
                activation=nn.ReLU,
                norm=nn.Identity,
            ),
        )

        self.y_conv1 = ConvBlock(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=1,
            groups=conv_dim,
            activation=nn.Tanh,
            norm=nn.Identity,
        )

        self.y_conv2 = ConvBlock(
            in_channels=conv_dim,
            out_channels=conv_dim,
            kernel_size=1,
            groups=conv_dim,
            activation=nn.Sigmoid,
            norm=nn.Identity,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.x_conv(x)
        return self.y_conv1(x) * self.y_conv2(x)
