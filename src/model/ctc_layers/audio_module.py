from typing import TypeVar

from torch import Tensor, nn

from src.model.ctc_layers.frcnn import FRCNNBlock
from src.model.layers import GlobalLayerNorm

TModule = TypeVar("TModule", bound=nn.Module)


class AuditoryModule(nn.Module):
    def __init__(
        self,
        stage_num: int = 5,
        conv_dim: int = 512,
        kernel_size: int = 5,
        activation: TModule = nn.ReLU,
    ):
        super().__init__()
        self.frcnn = FRCNNBlock(
            stage_num=stage_num,
            conv_dim=conv_dim,
            kernel_size=kernel_size,
            activation=activation,
            norm=GlobalLayerNorm,
        )

    def forward(self, audio: Tensor) -> Tensor:
        return self.frcnn(audio) + audio
