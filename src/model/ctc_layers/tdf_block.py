import math
from typing import TypeVar

import torch
from torch import Tensor, nn

from src.model.ctc_layers.audio_module import AuditoryModule
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
        norm: TModule = nn.LayerNorm,
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
                activation=nn.ReLU,
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


# TODO
# class GRUBlock(nn.Module):
#     def __init__(
#         self,
#         in_dim: int = 512,
#     ):
#         super().__init__()

#         self.gru = nn.GRU()


class InjectionSum(nn.Module):
    def __init__(self, conv_dim: int = 512, kernel_size: int = 5):
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
        y = torch.nn.functional.interpolate(y, x.shape[-1])
        scores = self.score_conv(y)
        additive = self.sum_conv(y)
        return self.x_conv(x) * scores + additive


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
        norm: TModule = GlobalLayerNorm,
        global_attention_type: str = "transformer",
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
        self.global_attention_type = global_attention_type

        self.encode = ConvBlock(
            self.in_dim,
            self.in_dim,
            kernel_size=1,
            stride=1,
            groups=self.in_dim,
            activation=nn.PReLU,
            norm=nn.Identity,
        )

        self.in_to_hidden = ConvBlock(
            self.in_dim,
            self.conv_dim,
            kernel_size=1,
            stride=1,
            activation=nn.PReLU,
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

        assert self.global_attention_type in (
            "transformer",
            "frcnn",
        )
        if self.global_attention_type == "transformer":
            self.global_attention = TransformerBlock(
                self.conv_dim, n_heads, ffn_dim, dropout
            )
        elif self.global_attention_type == "frcnn":
            self.global_attention = AuditoryModule(
                stage_num=3,
                conv_dim=self.conv_dim,
                kernel_size=3,
                activation=nn.PReLU,
            )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor. Shape: (batch_size, conv_dim, seq_len)

        Returns:
            out (Tensor): tensor, processed by TADNet. Shape: (batch_size, conv_dim, seq_len)
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
