import math
from typing import TypeVar

import torch
from torch import Tensor, nn

from src.model.ctc_layers.conv_block import ConvBlock
from src.model.ctc_layers.global_layer_norm import GlobalLayerNorm

TModule = TypeVar("TModule", bound=nn.Module)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x.permute(2, 0, 1)
        x = x + self.pe[: x.size(0)]
        return x.permute(1, 2, 0)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        in_dim: int = 512,
        nhead: int = 8,
        ffn_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.position_encoding = PositionalEncoding(in_dim, 5000)
        self.multihead_attn = nn.MultiheadAttention(
            in_dim, nhead, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            ConvBlock(
                in_dim, ffn_dim, kernel_size=1, bias=False, activation=nn.Identity
            ),
            ConvBlock(
                ffn_dim,
                ffn_dim,
                kernel_size=5,
                groups=ffn_dim,
                bias=True,
                padding=2,
                activation=nn.ReLU,
            ),
            ConvBlock(
                ffn_dim, in_dim, kernel_size=1, bias=False, activation=nn.Identity
            ),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        x = self.position_encoding(x).transpose(1, 2)
        x = self.multihead_attn(x, x, x)[0].transpose(1, 2)
        x = x + residual

        residual = x
        x = self.ffn(x)
        x = x + residual

        return x


class LA(nn.Module):
    def __init__(self, conv_dim: int = 512, kernel_size: int = 5):
        super().__init__()
        self.score_conv = ConvBlock(
            conv_dim,
            conv_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=conv_dim,
            activation=nn.Sigmoid,
        )
        self.sum_conv = ConvBlock(
            conv_dim,
            conv_dim,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=conv_dim,
            activation=nn.Identity,
        )

    def forward(self, cur: Tensor, prev: Tensor) -> Tensor:
        prev = torch.nn.functional.interpolate(prev, cur.shape[-1])
        return cur * self.score_conv(prev) + self.sum_conv(prev)


class TDANet(nn.Module):
    """
    TDANet, used to process audio/video features in.
    Possible replacement for A-FRCNN.

    https://arxiv.org/pdf/2209.15200
    """

    def __init__(
        self,
        stage_num: int = 5,
        enc_kernel_size: int = 5,
        la_kernel_size: int = 5,
        conv_dim: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        ffn_dim: int = 1024,
        norm: TModule = GlobalLayerNorm,
    ):
        """
        Args:
            stage_num (int): number of stages (nodes) in the block.
            conv_dim (int): number of channels in the block.
        """
        super().__init__()
        assert stage_num > 1, f"stage_num must be greater than 2, got {stage_num=}"
        self.stage_num = stage_num
        self.enc_kernel_size = enc_kernel_size
        self.la_kernel_size = la_kernel_size
        self.conv_dim = conv_dim

        self.initial_convs = nn.ModuleList(
            [
                ConvBlock(
                    self.conv_dim,
                    self.conv_dim,
                    kernel_size=self.enc_kernel_size,
                    stride=2,
                    dilation=2,
                    padding=4,
                    groups=self.conv_dim,
                    activation=nn.Identity,
                    norm=norm,
                )
                for _ in range(stage_num - 1)
            ]
        )

        self.avg_pools = nn.ModuleList(
            [
                nn.AvgPool1d(kernel_size=2**i, stride=2**i, ceil_mode=True)
                for i in range(stage_num - 1, -1, -1)
            ]
        )
        self.la_blocks = nn.ModuleList(
            [
                LA(self.conv_dim, kernel_size=self.la_kernel_size)
                for _ in range(stage_num - 1)
            ]
        )

        self.transformer = TransformerBlock(self.conv_dim, n_heads, ffn_dim, dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor. Shape: (batch_size, conv_dim, seq_len)

        Returns:
            out (Tensor): tensor, processed by TADNet. Shape: (batch_size, conv_dim, seq_len)
        """
        residual_x = x
        # initial encoding
        encoded = [x]
        for i in range(self.stage_num - 1):
            encoded.append(self.initial_convs[i](encoded[-1]))

        # transformer block
        transformer_input = encoded[-1]
        for i in range(self.stage_num):
            transformer_input = transformer_input + self.avg_pools[i](encoded[i])

        transformer_output = self.transformer(transformer_input)

        # fusing output of transformer with initial encodings
        for i in range(self.stage_num - 1, -1, -1):
            encoded[i] = encoded[i] * torch.nn.functional.sigmoid(
                torch.nn.functional.interpolate(
                    transformer_output, encoded[i].shape[-1]
                )
            )

        # apply LA blocks
        prev = encoded[-1]
        for i in range(self.stage_num - 2, -1, -1):
            prev = self.la_blocks[i](encoded[i], prev)

        return prev + residual_x
