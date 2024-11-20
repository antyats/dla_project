import math
from typing import TypeVar

import torch
from torch import Tensor, nn

from src.model.layers import ConvBlock

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
        norm: TModule = nn.LayerNorm,
        activation: TModule = nn.ReLU,
    ):
        super().__init__()
        self.norm1 = norm(in_dim)
        self.norm2 = norm(in_dim)
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
                activation=activation,
            ),
            ConvBlock(
                ffn_dim, in_dim, kernel_size=1, bias=False, activation=nn.Identity
            ),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x

        x = self.norm1(x.transpose(1, 2))
        x = self.position_encoding(x.transpose(1, 2)).transpose(1, 2)
        x = self.multihead_attn(x, x, x)[0]
        x = self.norm2(x).transpose(1, 2)
        x = x + residual

        residual = x
        x = self.ffn(x)
        x = x + residual

        return x
