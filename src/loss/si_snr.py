import torch
from torch import Tensor, nn
from torchmetrics.audio import ScaleInvariantSignalNoiseRatio


class SI_SNR(nn.Module):
    """
    Example of a loss function to use.
    """

    def __init__(self, device: str = "auto"):
        super().__init__()
        self.loss = ScaleInvariantSignalNoiseRatio()
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.loss = self.loss.to(device)

    def forward(self, output_audio: Tensor, target_audio: Tensor, **batch):
        """
        Loss function calculation logic.

        Args:
            output_audio (Tensor): model output predictions.
            target (Tensor): ground-truth audio.
        Returns:
            losses (dict): dict containing calculated loss functions.
        """
        return {"loss": self.loss(preds=output_audio, target=target_audio)}
