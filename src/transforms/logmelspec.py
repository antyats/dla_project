import torch
from torch import Tensor, nn
from torchaudio.transforms import MelSpectrogram


class LogMelSpectrogram(nn.Module):
    """
    Batch-version of Normalize for 1D Input.
    Used as an example of a batch transform.
    """

    def __init__(self, sample_rate: int, n_fft: int, *args, **kwargs):
        """
        Args:
            sample_rate (int): mean used in the normalization.
            n_fft (int): std used in the normalization.
        """
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.transform = MelSpectrogram(
            sample_rate=sample_rate, n_fft=n_fft, *args, **kwargs
        )

    def forward(self, x: Tensor):
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): log-melspectrogram.
        """
        return torch.log(self.transform(x))
