from typing import TypeVar

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
        thalamic_channels: int = 576,
        activation: TModule = nn.ReLU,
    ):
        super().__init__()
        self.audio_module_in = ConvBlock(
            audio_n_channels,
            thalamic_channels,
            kernel_size=1,
            stride=1,
            activation=activation,
        )
        self.video_module_in = ConvBlock(
            video_n_channels,
            thalamic_channels,
            kernel_size=1,
            stride=1,
            activation=activation,
        )
        self.video_module_out = nn.Sequential(
            nn.Linear(
                thalamic_channels,
                video_n_channels,
            ),
            activation(),
        )
        self.audio_module_out = nn.Sequential(
            nn.Linear(
                thalamic_channels,
                audio_n_channels,
            ),
            activation(),
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
        adapted_video = self.video_module_in.forward(
            nn.functional.interpolate(video, audio.shape[-1])
        )
        adapted_audio = self.audio_module_in.forward(
            nn.functional.interpolate(audio, video.shape[-1])
        )

        audio = self.audio_module_out(
            (self.audio_module_in(audio) + adapted_video).transpose(1, 2)
        ).transpose(1, 2)
        video = self.video_module_out(
            (self.video_module_in(video) + adapted_audio).transpose(1, 2)
        ).transpose(1, 2)

        return audio, video


class FusionModule(nn.Module):
    def __init__(
        self,
        audio_n_channels: int = 512,
        video_n_channels: int = 64,
        thalamic_channels: int = 576,
        activation: TModule = nn.ReLU,
    ):
        super().__init__()
        self.net = ThalamicNetwork(
            audio_n_channels=audio_n_channels,
            video_n_channels=video_n_channels,
            thalamic_channels=thalamic_channels,
            activation=activation,
        )

    def forward(self, audio, video, **batch):
        new_audio, new_video = self.net(audio, video)
        audio = audio + new_audio
        video = video + new_video

        return audio, video
