from typing import TypeVar

import torch
from torch import Tensor, nn

from src.model.ctc_layers.conv_block import ConvBlock

TModule = TypeVar("TModule", bound=nn.Module)


class ThalamicNetwork(nn.Module):
    """
    Thalamic network of CTCNet. It fuses audio and video features
    after they are processed by audio and video modules.
    """

    def __init__(
        self,
        audio_n_channels: int,
        video_n_channels: int,
    ):
        super().__init__()
        self.audio_out = ConvBlock(
            audio_n_channels + video_n_channels,
            audio_n_channels,
            kernel_size=1,
            stride=1,
            activation=nn.Identity,
        )

        self.video_out = ConvBlock(
            audio_n_channels + video_n_channels,
            video_n_channels,
            kernel_size=1,
            stride=1,
            activation=nn.Identity,
        )

    def forward(self, audio: Tensor, video: Tensor) -> Tensor:
        """
        Args:
            audio (Tensor): audio features. Shape: (batch_size, channels_a, seq_len_a)
            video (Tensor): video features. Shape: (batch_size, channels_v, seq_len_v)
        Returns:
            audio (Tensor): audio fused with video. Shape: (batch_size, channels_a, seq_len_a)
            video (Tensor): video fused with audio. Shape: (batch_size, channels_v, seq_len_v)
        """
        # adapted_video = self.video_module_adapter.forward(
        #     nn.functional.interpolate(video, audio.shape[-1])
        # )
        # adapted_audio = self.audio_module_adapter.forward(
        #     nn.functional.interpolate(audio, video.shape[-1])
        # )
        adapted_video = nn.functional.interpolate(video, audio.shape[-1])

        adapted_audio = nn.functional.interpolate(audio, video.shape[-1])
        audio = self.audio_out(torch.cat([audio, adapted_video], dim=1))
        video = self.video_out(torch.cat([adapted_audio, video], dim=1))

        return audio, video


class FusionModule(nn.Module):
    def __init__(
        self,
        audio_n_channels: int = 512,
        video_n_channels: int = 64,
    ):
        super().__init__()
        self.net = ThalamicNetwork(
            audio_n_channels=audio_n_channels,
            video_n_channels=video_n_channels,
        )

    def forward(self, audio, video, **batch):
        new_audio, new_video = self.net(audio, video)
        return new_audio, new_video
