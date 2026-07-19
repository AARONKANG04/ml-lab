import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([args.sin(), args.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.time_mlp = nn.Linear(time_dim, out_ch)
        self.shortcut = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t):
        h = F.silu(self.norm1(x))
        h = self.conv1(h)
        h = h + self.time_mlp(F.silu(t))[:, :, None, None]
        h = F.silu(self.norm2(h))
        h = self.conv2(h)
        return h + self.shortcut(x)


class UNet(nn.Module):
    def __init__(self, in_channels=1, base_channels=64, channel_mults=(1, 2, 4), time_dim=256):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        channels = [base_channels * m for m in channel_mults]

        # Encoder
        self.init_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        in_ch = channels[0]
        for out_ch in channels[1:]:
            self.down_blocks.append(ResBlock(in_ch, out_ch, time_dim))
            self.downsamples.append(nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1))
            in_ch = out_ch

        # Bottleneck
        self.mid_block = ResBlock(channels[-1], channels[-1], time_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for out_ch in reversed(channels[:-1]):
            self.upsamples.append(nn.ConvTranspose2d(in_ch, in_ch, 2, stride=2))
            self.up_blocks.append(ResBlock(in_ch + in_ch, out_ch, time_dim))
            in_ch = out_ch

        self.out_norm = nn.GroupNorm(8, channels[0])
        self.out_conv = nn.Conv2d(channels[0], in_channels, 3, padding=1)

    def forward(self, x, t):
        t = self.time_mlp(t)
        x = self.init_conv(x)

        skips = [x]
        for down_block, downsample in zip(self.down_blocks, self.downsamples):
            x = down_block(x, t)
            skips.append(x)
            x = downsample(x)

        x = self.mid_block(x, t)

        for up_block, upsample in zip(self.up_blocks, self.upsamples):
            x = upsample(x)
            x = torch.cat([x, skips.pop()], dim=1)
            x = up_block(x, t)

        x = F.silu(self.out_norm(x))
        return self.out_conv(x)