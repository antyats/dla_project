import torch.nn as nn
import torch.nn.functional as F


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
            GlobalLayerNorm(out_channels),
        )
        self.d = d
        self.convs = nn.ModuleList()

        for i in range(d):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(
                        out_channels, out_channels, kernel_size=5, stride=2, padding=2
                    ),
                    GlobalLayerNorm(out_channels),
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


class InterA_T(nn.Module):
    def __init__(self, audio_channels=128, video_channels=128):
        super().__init__()
        self.conv_audio = nn.Sequential(
            nn.Conv1d(audio_channels, video_channels, kernel_size=1),
            GlobalLayerNorm(video_channels),
        )
        self.conv_video = nn.Sequential(
            nn.Conv1d(video_channels, audio_channels, kernel_size=1),
            GlobalLayerNorm(audio_channels),
        )

        self.ffn_a = nn.Sequential(
            nn.Conv1d(audio_channels, 2 * audio_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(2 * audio_channels, 2 * audio_channels, kernel_size=5, bias=True),
            nn.ReLU(),
            nn.Conv1d(2 * audio_channels, audio_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            GlobalLayerNorm(audio_channels),
        )
        self.ffn_v = nn.Sequential(
            nn.Conv1d(video_channels, 2 * video_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv1d(2 * video_channels, 2 * video_channels, kernel_size=5, bias=True),
            nn.ReLU(),
            nn.Conv1d(2 * video_channels, video_channels, kernel_size=1, bias=False),
            nn.ReLU(),
            GlobalLayerNorm(video_channels),
        )

    def forward(self, audio, video):
        audio_interpolate = F.interpolate(audio, size=video.shape[-1], mode="nearest")
        video_interpolate = F.interpolate(video, size=audio.shape[-1], mode="nearest")
        audio *= F.sigmoid(self.conv_video(video_interpolate))
        video *= F.sigmoid(self.conv_audio(audio_interpolate))
        return self.ffn_a(audio), self.ffn_v(video)


class IntraA(nn.Module):
    def __init__(self, x_channels, y_channels):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(y_channels, x_channels, kernel_size=1),
            GlobalLayerNorm(x_channels),
        )
        self.conv2 = nn.Sequential(
            nn.Conv1d(y_channels, x_channels, kernel_size=1),
            GlobalLayerNorm(x_channels),
        )

    def forward(self, x, y):
        y_interpolate = F.interpolate(y, size=x.shape[-1], mode="nearest")
        x *= F.sigmoid(self.conv1(y_interpolate))
        x += self.conv2(y_interpolate)
        return x


class InterA_M(nn.Module):
    def __init__(self, audio_channels=128, video_channels=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(video_channels, audio_channels, kernel_size=1),
            GlobalLayerNorm(audio_channels),
        )

    def forward(self, audio, video):
        video_interpolate = F.interpolate(video, audio.shape[-1], mode="nearest")
        audio *= F.sigmoid(self.conv(video_interpolate))
        return audio


class InterA_B(nn.Module):
    def __init__(self, audio_channels=128, video_channels=128):
        super().__init__()
        self.conv1_audio = nn.Sequential(
            nn.Conv1d(audio_channels, video_channels, kernel_size=1),
            GlobalLayerNorm(video_channels),
        )
        self.conv2_audio = nn.Sequential(
            nn.Conv1d(video_channels, audio_channels, kernel_size=1),
            GlobalLayerNorm(audio_channels),
        )
        self.conv1_video = nn.Sequential(
            nn.Conv1d(video_channels, audio_channels, kernel_size=1),
            GlobalLayerNorm(audio_channels),
        )
        self.conv2_video = nn.Sequential(
            nn.Conv1d(audio_channels, video_channels, kernel_size=1),
            GlobalLayerNorm(video_channels),
        )

    def forward(self, audio, video):
        audio_interpolate = F.interpolate(audio, size=video.shape[-1], mode="nearest")
        video_interpolate = F.interpolate(video, size=audio.shape[-1], mode="nearest")

        audio_fused = F.sigmoid(self.conv1_audio(audio)) * video_interpolate
        audio_fused = self.conv2_audio(audio_fused)
        audio_fused += audio

        video_fused = F.sigmoid(self.conv1_video(video)) * audio_interpolate
        video_fused = self.conv2_video(video_fused)
        video_fused += video

        return audio_fused, video_fused
