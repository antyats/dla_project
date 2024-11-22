import torch
from torch import Tensor, nn

nn.LayerNorm


class GlobalLayerNorm(nn.Module):
    def __init__(self, n_channels: int, eps: float = 1e-8):
        """
        Args:
            n_channels (int): number of channels in the input tensor.
            eps (float): small value added to the denominator for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones((n_channels, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros((n_channels, 1)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x (Tensor): input tensor.
        Returns:
            x (Tensor): normalized tensor.
        """
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = torch.std(x, dim=(1, 2), keepdim=True, unbiased=True)
        return (x - mean) / (std + self.eps) * self.gamma + self.beta
