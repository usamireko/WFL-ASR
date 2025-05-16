import torch
import torch.nn as nn
import torch.nn.functional as F

class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.act = nn.GELU()
        self.res_conv = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        res = self.res_conv(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act(x)
        return x + res

class DownsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.conv = ResConvBlock(in_ch, out_ch, dilation=dilation)
        self.pool = nn.AvgPool1d(kernel_size=2)

    def forward(self, x):
        x = self.conv(x)
        return self.pool(x), x

class UpsampleBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        self.conv = ResConvBlock(in_ch + out_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        x = F.pad(x, (0, skip.size(-1) - x.size(-1)))  # match lengths
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)

class BoundaryPredictor(nn.Module):
    def __init__(self, mel_channels=80, base_channels=128):
        super().__init__()
        self.init = ResConvBlock(mel_channels, base_channels)

        # DILATION SLAY: increasing receptive field
        self.down1 = DownsampleBlock(base_channels, base_channels, dilation=2)
        self.down2 = DownsampleBlock(base_channels, base_channels * 2, dilation=4)

        self.bottleneck = ResConvBlock(base_channels * 2, base_channels * 2, dilation=4)

        self.up2 = UpsampleBlock(base_channels * 2, base_channels * 2)
        self.up1 = UpsampleBlock(base_channels * 2, base_channels)

        # Final and intermediate heads
        self.final = nn.Conv1d(base_channels, 1, kernel_size=1)
        self.intermediate1 = nn.Conv1d(base_channels * 2, 1, kernel_size=1) # from up2
        self.intermediate2 = nn.Conv1d(base_channels * 2, 1, kernel_size=1) # from bottleneck

    def forward(self, mel):
        x = self.init(mel)

        d1, skip1 = self.down1(x)
        d2, skip2 = self.down2(d1)
        b = self.bottleneck(d2)

        u2 = self.up2(b, skip2)
        u1 = self.up1(u2, skip1)

        out_main = self.final(u1).squeeze(1) # [B, T]
        out_mid1 = self.intermediate1(u2).squeeze(1) # [B, T]
        out_mid2 = self.intermediate2(b).squeeze(1) # [B, T]

        return out_main, out_mid1, out_mid2
