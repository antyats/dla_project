from typing import TypeVar

import torch
from torch import Tensor, nn

from src.model.ctc_layers.conv_block import ConvBlock

TModule = TypeVar("TModule", bound=nn.Module)


class FRCNNBlock(nn.Module):
    """
    A-FRCNN block (method B), used to process audio/video features in  CTCNet.

    https://arxiv.org/pdf/2112.02321
    """

    def __init__(
        self,
        stage_num: int = 5,
        kernel_size=5,
        conv_dim: int = 512,
        activation: TModule = nn.ReLU,
        norm: TModule = nn.InstanceNorm1d,
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
        padding_n = kernel_size // 2

        self.initial_cons = nn.ModuleList(
            [
                ConvBlock(
                    self.conv_dim,
                    self.conv_dim,
                    kernel_size=self.kernel_size,
                    stride=2,
                    padding=padding_n,
                    activation=activation,
                    norm=norm,
                )
                for _ in range(stage_num - 1)
            ]
        )
        self.bottom_up_cons = nn.ModuleList(
            [
                ConvBlock(
                    self.conv_dim,
                    self.conv_dim,
                    kernel_size=self.kernel_size,
                    stride=2,
                    padding=padding_n,
                    activation=activation,
                    norm=norm,
                )
                for _ in range(stage_num - 1)
            ]
        )
        self.intermediate_concat = nn.ModuleList(
            [
                ConvBlock(
                    self.conv_dim * 2,
                    self.conv_dim,
                    kernel_size=1,
                    stride=1,
                    activation=activation,
                    norm=norm,
                )
            ]  # first block downsampling - only lateral and top-down cons
            + [
                ConvBlock(
                    self.conv_dim * 3,
                    self.conv_dim,
                    kernel_size=1,
                    stride=1,
                    activation=activation,
                    norm=norm,
                )  # intermediate blocks downsampling - lateral, bottom-up and top-down cons
                for _ in range(stage_num - 2)
            ]
            + [
                ConvBlock(
                    self.conv_dim * 2,
                    self.conv_dim,
                    kernel_size=1,
                    stride=1,
                    activation=activation,
                    norm=norm,
                )
            ]  # last block downsampling - only lateral and bottom-up cons
        )

        self.final_concat = ConvBlock(
            self.conv_dim * self.stage_num,
            self.conv_dim,
            kernel_size=1,
            stride=1,
            activation=activation,
            norm=norm,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor. Shape: (batch_size, conv_dim, seq_len)

        Returns:
            out (Tensor): tensor, processed by FRCNN block. Shape: (batch_size, conv_dim, seq_len)
        """
        left = [x]
        for i in range(self.stage_num - 1):
            left.append(self.initial_cons[i](left[-1]))

        right = []
        for i in range(self.stage_num):
            bottom_up = (
                self.bottom_up_cons[i - 1](left[i - 1])
                if i > 0
                else torch.empty(0, device=left[i].device)
            )
            lateral = left[i]
            top_down = (
                torch.nn.functional.interpolate(left[i + 1], left[i].shape[-1])
                if i + 1 < self.stage_num
                else torch.empty(0, device=left[i].device)
            )
            right.append(
                self.intermediate_concat[i](
                    torch.cat([bottom_up, lateral, top_down], dim=1)
                )
            )

        return self.final_concat(
            torch.cat(
                [nn.functional.interpolate(x, right[0].shape[-1]) for x in right],
                dim=1,
            )
        )
