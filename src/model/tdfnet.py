from typing import Optional, TypeVar

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from src.model.layers import ConvBlock, FusionModule
from src.model.tdf_layers import TDFBlock

TModule = TypeVar("TModule", bound=nn.Module)


class TDFNet(nn.Module):
    """
    AVSS model TDFNet.

    https://arxiv.org/pdf/2401.14185v1
    """

    def __init__(
        self,
        video_feature_extractor: nn.Module,
        in_video_features: int = 512,
        path_to_pretrained_video_extractor: Optional[str] = None,
        n_audio_channels: int = 512,
        n_video_channels: int = 64,
        audio_stage_n: int = 5,
        video_stage_n: int = 4,
        audio_kernel_size: int = 5,
        video_kernel_size: int = 3,
        fusion_steps: int = 3,
        audio_only_steps: int = 5,
        activation: TModule = nn.PReLU,
        use_grad_checkpointing: bool = False,
    ):
        """
        Initializes the TDFNet.

        Args:
            video_feature_extractor (nn.Module): feature extractor for video features
            in_video_features (int, optional): number of extracted video features by video_feature_extractor.
            path_to_pretrained_video_extractor (str, optional): path to pretrained video feature extractor.
            n_audio_channels (int, optional): number of feature channels for audio.
            n_video_channels (int, optional): number of feature channels for video.
            audio_stage_n (int, optional): number of stages for FRCNN in audio module.
            video_stage_n (int, optional): number of stages for FRCNN in video module.
            audio_kernel_size (int, optional): kernel size for audio module convolutions.
            video_kernel_size (int, optional): kernel size for video module convolutions.
            fusion_steps (int, optional): number of steps where audio and video are fused.
            audio_only_steps (int, optional): number of audio-only steps after fusion.
            activation (TModule, optional): activation function to use in modules.
            use_grad_checkpointing (bool, optional): whether to use gradient checkpointing.
        """
        super().__init__()
        self.fusion_steps = fusion_steps
        self.audio_only_steps = audio_only_steps
        self.in_video_features = in_video_features
        self.use_grad_checkpointing = use_grad_checkpointing

        self.video_feature_extractor = video_feature_extractor

        if path_to_pretrained_video_extractor is not None:
            print("loading pretrained video feature extractor")
            self.video_feature_extractor.load_state_dict(
                torch.load(path_to_pretrained_video_extractor)["model_state_dict"]
            )
        for param in self.video_feature_extractor.parameters():
            param.requires_grad = False

        self.video_downsample = ConvBlock(
            in_channels=in_video_features,
            out_channels=n_video_channels,
            kernel_size=1,
            stride=1,
            activation=nn.Identity,
            norm=nn.BatchNorm1d,
        )

        self.audio_module = TDFBlock(
            in_dim=n_audio_channels,
            stage_num=audio_stage_n,
            conv_dim=n_audio_channels,
            kernel_size=audio_kernel_size,
            activation=activation,
        )
        self.visual_module = TDFBlock(
            in_dim=n_video_channels,
            stage_num=video_stage_n,
            conv_dim=n_video_channels,
            kernel_size=video_kernel_size,
            activation=activation,
        )
        self.fusion_module = FusionModule(
            audio_n_channels=n_audio_channels,
            video_n_channels=n_video_channels,
        )

        self.fb = ConvBlock(
            in_channels=1,
            out_channels=n_audio_channels,
            kernel_size=21,
            stride=10,
            activation=nn.Identity,
        )
        self.inv_fb = nn.ConvTranspose1d(
            in_channels=n_audio_channels,
            out_channels=1,
            stride=10,
            kernel_size=21,
            output_padding=9,
        )

    def _checkpoint_audio(self, *inputs) -> Tensor:
        if self.use_grad_checkpointing:
            return checkpoint(self.audio_module, *inputs, use_reentrant=False)
        return self.audio_module(*inputs)

    def _checkpoint_video(self, *inputs) -> Tensor:
        if self.use_grad_checkpointing:
            return checkpoint(self.visual_module, *inputs, use_reentrant=False)
        return self.visual_module(*inputs)

    def _checkpoint_fusion(self, *inputs) -> Tensor:
        if self.use_grad_checkpointing:
            return checkpoint(self.fusion_module, *inputs, use_reentrant=False)
        return self.fusion_module(*inputs)

    def forward(self, mix_audio: Tensor, video: Tensor, **batch) -> dict:
        """
        Args:
            audio (torch.Tensor): input audio. Shape: (batch_size, 1, audio_len)
            video (torch.Tensor): input video. Shape: (batch_size, 1, video_len, h, w)
        Returns:
            output_audio (torch.Tensor): output audio. Shape: (batch_size, 1, audio_len)
        """
        audio = self.fb(mix_audio)
        with torch.no_grad():  # video_feature_extractor is fixed
            video = self.video_feature_extractor(video, lengths=None).transpose(1, 2)
        video = self.video_downsample(video)

        residual_audio, residual_video = audio, video

        for _ in range(self.fusion_steps):
            audio = self._checkpoint_audio(residual_audio + audio)
            video = self._checkpoint_video(residual_video + video)

            audio, video = self._checkpoint_fusion(residual_audio, residual_video)

        for _ in range(self.audio_only_steps):
            audio = self._checkpoint_audio(residual_audio + audio)

        audio = self.inv_fb(audio)
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
