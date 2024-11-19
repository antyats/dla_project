import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-8


class GlobalLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.activation = nn.GroupNorm(1, num_channels, eps=1e-8)

    def forward(self, x):
        return self.activation(x)


class BottomUp(nn.Module):
    def __init__(self, in_channels=128, out_channels=512, d=4):
        super().__init__()
        self.align = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.GlobalLayerNorm(out_channels),
        )
        self.d = d
        self.convs = nn.ModuleList()

        for i in range(d):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        out_channels, out_channels, kernel_size=5, stride=2, padding=2
                    ),
                    nn.GlobalLayerNorm(out_channels),
                )
            )

    def forward(self, emb):
        outputs = [self.align(emb)]
        for i in range(self.d):
            outputs.append(self.convs[i](outputs[-1]))

        sum = outputs[-1]
        pooling_size = outputs[-1].shape[-1]
        for i in range(self.d - 1):
            sum += F.adaptive_avg_pool1d(outputs[i], pooling_size)
        return outputs, sum
