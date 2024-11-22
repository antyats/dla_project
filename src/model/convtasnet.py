from torch import nn
from src.model.convtasnet_layers.encoder import Encoder, TCN, Decoder


class ConvTasNet(nn.Module):
    def __init__(
        self,
        n_audio_channels: int = 512,
        n_output_channels: int = 512,
        encoder_kernel_size: int = 8,
        encoder_padding: int = 8,
        encoder_stride: int = 4,
        decoder_kernel_size: int = 8,
        decoder_padding: int = 4,
        decoder_stride: int = 4,
        separator_kernel_size: int = 8,
        separator_size: int = 2,
        separator_count_blocks: int = 2,
    ):
        
        self.n_audio_channels = n_audio_channels
        self.n_output_channels = n_output_channels

        self.separator_size = separator_size
        self.separator_count_blocks = separator_count_blocks
        self.separator_kernel_size = separator_kernel_size

        self.encoder_kernel_size = encoder_kernel_size
        self.encoder_stride = encoder_stride
        self.encoder_padding = encoder_padding

        self.decoder_kernel_size = decoder_kernel_size
        self.decoder_stride = decoder_stride
        self.decoder_padding = decoder_padding

        self.encoder = Encoder(
            in_channels=1,
            out_channels=self.n_output_channels,
            kernel_size=self.encoder_kernel_size,
            padding=self.encoder_padding,
            stride=self.encoder_stride,
        )

        self.separator = TCN(
            in_channels=self.n_audio_channels,
            out_channels=self.separator_out_channels,
            kernel_size=self.separator_kernel_size,
            padding=self.separator_padding,
            separator_size=self.separator_size,
            separator_count_blocks=self.separator_count_blocks,
            dilation=self.dilation,
        )

        self.decoder = Decoder(
            in_channels=self.decoder_in_channels,
            out_channels=1,
            kernel_size=self.decoder_kernel_size,
            padding=self.decoder_padding,
            stride=self.decoder_stride,
        )
        
    def forward(self, mix_audio, **batch):    
        out = self.encoder(mix_audio)

        out = self.separator(out)
        
        out = self.decoder(out)

        return {"output_audio": out}

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
