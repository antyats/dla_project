import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class GlobalLayerNorm(nn.Module):
    def __init__(self, num_channels):
        super().__init__()
        self.activation = nn.GroupNorm(1, num_channels, eps=1e-8)

    def forward(self, x):
        return self.activation(x)


class BottomUp(nn.Module):
    def __init__(self, channels=512, depth=4):
        super().__init__()

        self.depth = depth
        self.convs = nn.ModuleList()
        for i in range(depth):
            self.convs.append(
                nn.Sequential(
                    nn.Conv1d(channels, channels, kernel_size=5, stride=2, padding=2),
                    GlobalLayerNorm(channels),
                )
            )

    def forward(self, emb):
        outputs = [emb]
        for i in range(self.depth):
            outputs.append(self.convs[i](outputs[-1]))

        pooled_sum = outputs[-1]
        pooling_size = outputs[-1].shape[-1]
        for i in range(self.depth - 1):
            pooled_sum = pooled_sum + F.adaptive_avg_pool1d(outputs[i], pooling_size)
        return outputs, pooled_sum


class FFNblock(nn.Module):
    def __init__(self, channels=512):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv1d(channels, 2 * channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(2 * channels, 2 * channels, kernel_size=5, bias=True),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv1d(2 * channels, channels, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Dropout(0.1),
            GlobalLayerNorm(channels),
        )

    def forward(self, x):
        return self.ffn(x)


class InterA_T(nn.Module):
    def __init__(self, audio_channels=512, video_channels=512):
        super().__init__()
        self.conv_audio = nn.Sequential(
            nn.Conv1d(audio_channels, video_channels, kernel_size=1),
            GlobalLayerNorm(video_channels),
        )
        self.conv_video = nn.Sequential(
            nn.Conv1d(video_channels, audio_channels, kernel_size=1),
            GlobalLayerNorm(audio_channels),
        )

        self.ffn_a = FFNblock(audio_channels)
        self.ffn_v = FFNblock(video_channels)

    def forward(self, audio, video):
        audio_interpolate = F.interpolate(audio, size=video.shape[-1], mode="nearest")
        video_interpolate = F.interpolate(video, size=audio.shape[-1], mode="nearest")
        audio = audio * F.sigmoid(self.conv_video(video_interpolate))
        video = video * F.sigmoid(self.conv_audio(audio_interpolate))
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
        x = x * F.sigmoid(self.conv1(y_interpolate))
        x = x + self.conv2(y_interpolate)
        return x


class InterA_M(nn.Module):
    def __init__(self, audio_channels=512, video_channels=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(video_channels, audio_channels, kernel_size=1),
            GlobalLayerNorm(audio_channels),
        )

    def forward(self, audio, video):
        video_interpolate = F.interpolate(video, audio.shape[-1], mode="nearest")
        audio = audio * F.sigmoid(self.conv(video_interpolate))
        return audio


class InterA_B(nn.Module):
    def __init__(self, audio_channels=512, video_channels=512):
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
        audio_fused = audio_fused + audio

        video_fused = F.sigmoid(self.conv1_video(video)) * audio_interpolate
        video_fused = self.conv2_video(video_fused)
        video_fused = video_fused + video

        return audio_fused, video_fused


class AudioBlock(nn.Module):
    def __init__(self, audio_channels=512, depth=4):
        super().__init__()

        self.depth = depth
        self.bottom_up_audio = BottomUp(audio_channels, depth)

        self.ffn_s = FFNblock(audio_channels)

        self.global_intra_a_audio = nn.ModuleList()
        self.local_intra_a_audio = nn.ModuleList()

        for i in range(depth):
            self.global_intra_a_audio.append(IntraA(audio_channels, audio_channels))

        for i in range(depth - 1):
            self.local_intra_a_audio.append(IntraA(audio_channels, audio_channels))

    def forward(self, es):
        s, fs = self.bottom_up_audio(es)
        sg = self.ffn_s(fs)

        for i in range(self.depth):
            s[i] = self.global_intra_a_audio[i](s[i], sg)

        s0 = s[-1]
        for i in range(self.depth - 2, -1, -1):
            s0 = self.local_intra_a_audio[i](s[i], s0)
        return s0


class FusionBlock(nn.Module):
    def __init__(
        self,
        audio_block: AudioBlock,
        audio_channels=512,
        video_channels=512,
        depth=4,
    ):
        super().__init__()

        self.audio_block = audio_block

        self.depth = depth
        self.bottom_up_video = BottomUp(video_channels, depth)

        self.inter_a_t = InterA_T(audio_channels, video_channels)
        self.inter_a_b = InterA_B(audio_channels, video_channels)

        self.global_intra_a_video = nn.ModuleList()

        self.local_intra_a_video = nn.ModuleList()

        self.inter_a_m = nn.ModuleList()

        for i in range(depth):
            self.global_intra_a_video.append(IntraA(video_channels, video_channels))

        for i in range(depth - 1):
            self.local_intra_a_video.append(IntraA(video_channels, video_channels))

        for i in range(depth):
            self.inter_a_m.append(InterA_M(audio_channels, video_channels))

    def forward(self, es, ev):
        s, fs = self.audio_block.bottom_up_audio(es)
        v, fv = self.bottom_up_video(ev)
        sg, vg = self.inter_a_t(fs, fv)

        for i in range(self.depth):
            s[i] = self.audio_block.global_intra_a_audio[i](s[i], sg)
            v[i] = self.global_intra_a_video[i](v[i], vg)
            s[i] = self.inter_a_m[i](s[i], v[i])

        s0 = s[-1]
        v0 = v[-1]
        for i in range(self.depth - 2, -1, -1):
            s0 = self.audio_block.local_intra_a_audio[i](s[i], s0)
            v0 = self.local_intra_a_video[i](v[i], v0)

        new_es, new_ev = self.inter_a_b(s0, v0)
        return new_es, new_ev


class SeparationNetwork(nn.Module):
    def __init__(
        self,
        NF=4,
        NS=12,
        audio_channels=512,
        video_channels=512,
        depth=4,
        use_grad_checkpointing=True,
    ):
        super().__init__()
        self.NF = NF
        self.NS = NS
        self.use_grad_checkpointing = use_grad_checkpointing

        self.audio_block = AudioBlock(audio_channels, depth)
        self.fusion_block = FusionBlock(
            self.audio_block,
            audio_channels,
            video_channels,
            depth,
        )

    def _checkpoint(self, module, *inputs):
        if self.use_grad_checkpointing:
            return checkpoint(module, *inputs, use_reentrant=False)
        return module(*inputs)

    def forward(self, es, ev):
        for i in range(self.NF):
            es, ev = self._checkpoint(self.fusion_block, es, ev)
        for i in range(self.NS):
            es = self._checkpoint(self.audio_block, es)
        return es


class AudioEncoder(nn.Module):
    def __init__(self, out_channels=512, kernel_size=16, stride=8):
        super().__init__()

        self.encoder = nn.Conv1d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - stride) // 2,
            bias=False,
        )

    def forward(self, wav):
        return self.encoder(wav)


class AudioDecoder(nn.Module):
    def __init__(self, in_channels=512, kernel_size=16, stride=8):
        super().__init__()

        self.decoder = nn.ConvTranspose1d(
            in_channels=in_channels,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=(kernel_size - stride) // 2,
            bias=False,
        )

    def forward(self, emb):
        return self.decoder(emb)


class AlignChannels(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.align = nn.Identity()
        if in_channels != out_channels:
            self.align = nn.Conv1d(in_channels, out_channels, 1, bias=False)

    def forward(self, x):
        return self.align(x)


class IIANet(nn.Module):
    def __init__(
        self,
        video_encoder: nn.Module,
        path_to_pretrained_video_encoder: str,
        audio_emb_channels: int = 512,
        video_emb_channels: int = 50,
        hidden_audio_emb_channels: int = 512,
        hidden_video_emb_channels: int = 512,
        NF: int = 4,
        NS: int = 12,
        bottomup_depth: int = 4,
        use_grad_checkpointing: bool = True,
    ):
        super().__init__()

        self.align_audio_emb_before_sep = AlignChannels(
            audio_emb_channels, hidden_audio_emb_channels
        )
        self.align_video_emb_before_sep = AlignChannels(
            video_emb_channels, hidden_video_emb_channels
        )

        self.align_audio_emb_after_sep = AlignChannels(
            hidden_audio_emb_channels, audio_emb_channels
        )

        self.video_encoder = video_encoder
        self._load_video_encoder(path_to_pretrained_video_encoder)

        self.audio_encoder = AudioEncoder(audio_emb_channels)
        self.audio_decoder = AudioDecoder(audio_emb_channels)
        self.separation = SeparationNetwork(
            NF=NF,
            NS=NS,
            audio_channels=hidden_audio_emb_channels,
            video_channels=hidden_video_emb_channels,
            depth=bottomup_depth,
            use_grad_checkpointing=use_grad_checkpointing,
        )

    def forward(self, mix_audio, video, **batch):
        es = self.audio_encoder(mix_audio)
        with torch.no_grad():
            ev = self.video_encoder(video, lengths=None)
        es_aligned = self.align_audio_emb_before_sep(es)
        ev_aligned = self.align_video_emb_before_sep(ev)

        M = self.separation(es_aligned, ev_aligned)
        M = self.align_audio_emb_after_sep(M)
        M = F.relu(M)
        audio = self.audio_decoder(es * M)
        return {"output_audio": audio}

    def _load_video_encoder(self, path):
        self.video_encoder.load_state_dict(torch.load(path)["model_state_dict"])
        for param in self.video_encoder.parameters():
            param.requires_grad = False

    def __str__(self):
        all_parameters = sum([p.numel() for p in self.parameters()])
        trainable_parameters = sum(
            [p.numel() for p in self.parameters() if p.requires_grad]
        )

        result_info = super().__str__()
        result_info = result_info + f"\nAll parameters: {all_parameters}"
        result_info = result_info + f"\nTrainable parameters: {trainable_parameters}"

        return result_info
