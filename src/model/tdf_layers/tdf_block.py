from typing import TypeVar

import torch
from torch import Tensor, nn

from src.model.layers import ConvBlock, GlobalLayerNorm
from src.model.tdf_layers.injection_sum import InjectionSum
from src.model.tdf_layers.transformer import TransformerBlock

TModule = TypeVar("TModule", bound=nn.Module)


class TDFBlock(nn.Module):
    """
    TDFBlock, used to process audio/video features in.
    Possible replacement for A-FRCNN.

    https://arxiv.org/pdf/2401.14185v1
    """

    def __init__(
        self,
        stage_num: int = 5,
        kernel_size: int = 5,
        in_dim: int = 512,
        conv_dim: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        ffn_dim: int = 1024,
        activation: TModule = nn.PReLU,
        norm: TModule = GlobalLayerNorm,
    ):
        """
        Args:
            stage_num (int): number of stages (nodes) in the block.
            conv_dim (int): number of channels in the block.
        """
        super().__init__()
        assert stage_num > 1, f"stage_num must be greater than 1, got {stage_num=}"
        self.stage_num = stage_num
        self.kernel_size = kernel_size
        self.conv_dim = conv_dim
        self.in_dim = in_dim

        self.encode = ConvBlock(
            self.in_dim,
            self.in_dim,
            kernel_size=1,
            stride=1,
            groups=self.in_dim,
            activation=activation,
            norm=nn.Identity,
        )

        self.in_to_hidden = ConvBlock(
            self.in_dim,
            self.conv_dim,
            kernel_size=1,
            stride=1,
            activation=activation,
            norm=norm,
        )
        self.hidden_to_in = ConvBlock(
            self.conv_dim,
            self.in_dim,
            kernel_size=1,
            stride=1,
            activation=nn.Identity,
            norm=nn.Identity,
        )

        self.bottom_up = nn.ModuleList(
            [
                ConvBlock(
                    self.conv_dim,
                    self.conv_dim,
                    kernel_size=self.kernel_size,
                    stride=2,
                    padding=self.kernel_size // 2,
                    groups=self.conv_dim,
                    activation=nn.Identity,
                    norm=norm,
                )
                for _ in range(stage_num - 1)
            ]
        )

        self.inter_inj_sums = nn.ModuleList(
            [
                InjectionSum(self.conv_dim, kernel_size=kernel_size)
                for _ in range(stage_num)
            ]
        )
        self.last_inj_sums = nn.ModuleList(
            [
                InjectionSum(self.conv_dim, kernel_size=kernel_size)
                for _ in range(stage_num - 1)
            ]
        )

        self.global_attention = TransformerBlock(
            self.conv_dim, n_heads, ffn_dim, dropout
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor. Shape: (batch_size, conv_dim, seq_len)

        Returns:
            out (Tensor): tensor, processed by TDFNet. Shape: (batch_size, conv_dim, seq_len)
        """
        # initial encoding
        x = self.encode(x)
        residual = x

        bottom_ups = [self.in_to_hidden(x)]
        for i in range(self.stage_num - 1):
            bottom_ups.append(self.bottom_up[i](bottom_ups[-1]))

        # transformer block
        global_attention_input = bottom_ups[-1]
        for i in range(self.stage_num - 1):
            global_attention_input = (
                global_attention_input
                + torch.nn.functional.adaptive_avg_pool1d(
                    bottom_ups[i], bottom_ups[-1].shape[-1]
                )
            )

        global_attention_output = self.global_attention(global_attention_input)

        # fusing output of transformer with initial encodings
        for i in range(self.stage_num):
            bottom_ups[i] = self.inter_inj_sums[i](
                bottom_ups[i], global_attention_output
            )

        # final fusion
        prev = bottom_ups[-1]
        for i in range(self.stage_num - 2, -1, -1):
            prev = self.last_inj_sums[i](bottom_ups[i], prev) + bottom_ups[i]

        return self.hidden_to_in(prev) + residual
