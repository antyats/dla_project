import torch
from torch import Tensor, nn


class GlobalLayerNorm(nn.Module):
    def __init__(self, n_channels: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones((n_channels, 1)), requires_grad=True)
        self.beta = nn.Parameter(torch.ones((n_channels, 1)), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        mean = x.mean(dim=(1, 2), keepdim=True)
        std = torch.std(x, dim=(1, 2), keepdim=True, unbiased=True)
        return (x - mean) / (std + self.eps) * self.gamma + self.beta
