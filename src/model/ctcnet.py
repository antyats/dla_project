from typing import Optional, TypeVar

import torch
from torch import Tensor, nn

from src.model.ctc_layers import AuditoryModule, ConvBlock, FusionModule, VisualModule

TModule = TypeVar("TModule", bound=nn.Module)


class CTCNet(nn.Module):
    """
    AVSS model CTCNet.

    https://arxiv.org/pdf/2212.10744
    """

    def __init__(
        self,
        video_feature_extractor: nn.Module,
        in_video_features: int = 1024,
        path_to_pretrained_video_extractor: Optional[str] = None,
        n_audio_channels: int = 512,
        n_video_channels: int = 64,
        thalamic_channels: int = 576,
        audio_stage_n: int = 5,
        video_stage_n: int = 5,
        audio_kernel_size: int = 5,
        video_kernel_size: int = 3,
        fusion_steps: int = 3,
        audio_only_steps: int = 5,
        activation: TModule = nn.ReLU,
    ):
        """
        Initializes the CTCNet.

        Args:
            video_feature_extractor (nn.Module): feature extractor for video features
            in_video_features (int): number of extracted video features by video_feature_extractor
            n_audio_channels (int, optional): number of feature channels for audio. Defaults to 512.
            n_video_channels (int, optional): number of feature channels for video. Defaults to 64.
            thalamic_channels (int, optional): number of feature channels for thalamic network. Defaults to 576.
            audio_stage_n (int, optional): number of stages for FRCNN in audio module. Defaults to 5.
            video_stage_n (int, optional): number of stages for FRCNN in video module. Defaults to 5.
            audio_kernel_size (int, optional): kernel size for audio module convolutions. Defaults to 5.
            video_kernel_size (int, optional): kernel size for video module convolutions. Defaults to 3.
            fusion_steps (int, optional): number of steps where audio and video are fused. Defaults to 3.
            audio_only_steps (int, optional): number of audio-only steps after fusion. Defaults to 5.
        """
        super().__init__()
        self.fusion_steps = fusion_steps
        self.audio_only_steps = audio_only_steps
        self.in_video_features = in_video_features

        self.video_feature_extractor = video_feature_extractor

        if path_to_pretrained_video_extractor is not None:
            self.video_feature_extractor.load_state_dict(
                torch.load(path_to_pretrained_video_extractor)["model_state_dict"]
            )

        self.video_downsample = ConvBlock(
            in_channels=in_video_features,
            out_channels=n_video_channels,
            kernel_size=1,
            stride=1,
            activation=activation,
            norm=nn.BatchNorm1d,
        )

        self.audio_module = AuditoryModule(
            stage_num=audio_stage_n,
            conv_dim=n_audio_channels,
            kernel_size=audio_kernel_size,
            activation=activation,
        )
        self.visual_module = VisualModule(
            stage_num=video_stage_n,
            conv_dim=n_video_channels,
            kernel_size=video_kernel_size,
            activation=activation,
        )
        self.fusion_module = FusionModule(
            audio_n_channels=n_audio_channels,
            video_n_channels=n_video_channels,
            thalamic_channels=thalamic_channels,
            activation=activation,
        )

        self.fb = ConvBlock(  # or conv1d?
            in_channels=1,
            out_channels=n_audio_channels,
            kernel_size=21,
            padding=10,
            activation=activation,
        )
        self.inv_fb = nn.ConvTranspose1d(
            in_channels=n_audio_channels,
            out_channels=1,
            kernel_size=21,
            padding=10,
        )

        self.linear_head = nn.Sequential(
            nn.Linear(in_features=n_audio_channels, out_features=n_audio_channels),
            activation(),
        )

    def forward(self, audio: Tensor, video: Tensor, **batch) -> dict:
        """
        Args:
            audio (torch.Tensor): input audio. Shape: (batch_size, 1, audio_len)
            video (torch.Tensor): input video. Shape: (batch_size, 1, video_len, h, w)
        Returns:
            output_audio (torch.Tensor): output audio. Shape: (batch_size, audio_len)
        """
        audio = self.fb(audio)
        video = self.video_feature_extractor(video, lengths=None).transpose(1, 2)
        video = self.video_downsample(video)

        for _ in range(self.fusion_steps):
            audio = self.audio_module(audio)
            video = self.visual_module(video)
            audio, video = self.fusion_module(audio, video)

        for _ in range(self.audio_only_steps):
            audio = self.audio_module(audio)

        mask = self.linear_head(audio.transpose(-1, -2)).transpose(-1, -2)

        audio = self.inv_fb(audio * mask).squeeze(1)
        return {"output_audio": audio}

    def __str__(self):
        """
        Model prints with the number of parameters.
        """
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
