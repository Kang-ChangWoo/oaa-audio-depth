"""EchoScan baseline (Yeon et al.), waveform-1D-encoder variant for ERP depth.

EchoScan estimates room geometry (a 1024x1024 floorplan) + room HEIGHT from raw multi-channel
RIR waveforms via a 1D-conv ResNet encoder + GEM global descriptor + a 2D up-conv decoder. Here we:
  * keep the raw-WAVEFORM 1D encoder VERBATIM (the part worth reusing -- it consumes echoes directly
    instead of a spectrogram; we have binaural waveforms in cache/rx_wave),
  * DROP the height head entirely (depth task, no height),
  * retarget the floorplan decoder from a square 1024x1024 seed to a 4x8 seed so the SAME up-conv
    stack outputs a native 256x512 ERP depth (2:1) -- no post-hoc upsampling.

Encoder/ConvBlock/GlobalDescriptor/L2Norm/SimpleDecoder/ResnetBlock/Normalize* are copied verbatim
from nets/echoscan.py; only the decoder seed grid (16x16 -> 4x8) and the final output framing change.

forward(wave (B,in_ch,L)) -> depth (B,1,256,512) in [0,1] (x max_depth later).
Note: this model consumes the raw binaural WAVEFORM (cache/rx_wave), not the STFT `spec`.
"""
from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------- verbatim helpers

def nonlinearity(x):
    return F.relu6(x)


def Normalize(in_channels):
    return nn.BatchNorm2d(in_channels)


def Normalize1d(in_channels):
    return nn.BatchNorm1d(in_channels)


class L2Norm(nn.Module):
    def forward(self, x):
        assert x.dim() == 2, "the input tensor of L2Norm must be the shape of [B, C]"
        return F.normalize(x, p=2, dim=-1, eps=1e-8)


class GlobalDescriptor(nn.Module):
    def __init__(self, p=1):
        super().__init__()
        self.p = p
        self.eps = 1e-8

    def forward(self, x):
        assert x.dim() == 3, "the input tensor of GlobalDescriptor must be the shape of [B, C, D]"
        if self.p == 1:
            return x.mean(dim=[-1])
        elif self.p == float("inf"):
            return torch.flatten(F.adaptive_max_pool2d(x, output_size=(1024, 1)), start_dim=2)
        sum_value = x.pow(self.p).mean(dim=[-1])
        return torch.sign(sum_value) * ((torch.abs(sum_value + self.eps) + self.eps).pow(1.0 / self.p))


class ConvBlock(nn.Module):
    def __init__(self, in_ch, ch_expand):
        super().__init__()
        self.ch_expand = ch_expand
        kernel, pad = 5, 2
        if ch_expand:
            self.conv1 = nn.Conv1d(in_ch, in_ch * 2, kernel, stride=2, padding=pad)
            self.norm1 = Normalize1d(in_ch * 2)
            self.conv2 = nn.Conv1d(in_ch * 2, in_ch * 2, kernel, stride=1, padding=pad)
            self.norm2 = Normalize1d(in_ch * 2)
            self.proj_block = nn.Conv1d(in_ch, in_ch * 2, 1, stride=2)
        else:
            self.conv1 = nn.Conv1d(in_ch, in_ch, kernel, stride=1, padding=pad)
            self.norm1 = Normalize1d(in_ch)
            self.conv2 = nn.Conv1d(in_ch, in_ch, kernel, stride=1, padding=pad)
            self.norm2 = Normalize1d(in_ch)

    def forward(self, x):
        identity = x
        x = nonlinearity(self.norm1(self.conv1(x)))
        x = nonlinearity(self.norm2(self.conv2(x)))
        if self.ch_expand:
            identity = self.proj_block(identity)
        return x + identity


class Encoder(nn.Module):
    """Verbatim EchoScan 1D-conv ResNet encoder: waveform (B,in_ch,L) -> (B,512) descriptor."""

    def __init__(self, conf):
        super().__init__()
        self.conf = conf
        input_ch = conf.model.input_ch
        init_ch = conf.model.init_ch
        init_kernel = int(conf.model.fs * 0.001)          # 1 ms kernel
        if init_kernel % 2 == 0:
            init_kernel += 1
        self.preconv = nn.Conv1d(input_ch, init_ch, kernel_size=init_kernel, stride=2, padding=init_kernel // 2)
        self.preconv_norm = Normalize1d(init_ch)
        self.cb1 = nn.Sequential(ConvBlock(init_ch, False), ConvBlock(init_ch, False), ConvBlock(init_ch, False))
        self.cb2 = nn.Sequential(ConvBlock(init_ch, True), ConvBlock(init_ch * 2, False))
        self.cb3 = nn.Sequential(ConvBlock(init_ch * 2, True), ConvBlock(init_ch * 4, False))
        self.cb4 = nn.Sequential(ConvBlock(init_ch * 4, True), ConvBlock(init_ch * 8, False), ConvBlock(init_ch * 8, False))
        self.cb5 = nn.Sequential(ConvBlock(init_ch * 8, True), ConvBlock(init_ch * 16, False), ConvBlock(init_ch * 16, False))
        self.cb6 = nn.Sequential(ConvBlock(init_ch * 16, True), ConvBlock(init_ch * 32, False), ConvBlock(init_ch * 32, False))
        if conf.model.use_trainable_gdescriptor:
            self.p0 = nn.Parameter(torch.zeros([]))
            self.p1 = nn.Parameter(torch.ones([]))
        else:
            self.p0, self.p1 = 1, 3
        self.linear_g0 = nn.Sequential(nn.Linear(init_ch * 32, 256, bias=False), L2Norm())
        self.linear_g1 = nn.Sequential(nn.Linear(init_ch * 32, 256, bias=False), L2Norm())

    def forward(self, x):
        x = nonlinearity(self.preconv_norm(self.preconv(x)))
        x = self.cb1(x); x = self.cb2(x); x = self.cb3(x)
        x = self.cb4(x); x = self.cb5(x); x = self.cb6(x)
        if self.conf.model.use_trainable_gdescriptor:
            gd0 = GlobalDescriptor(p=1 + F.relu(self.p0))(x)
            gd1 = GlobalDescriptor(p=1 + F.relu(self.p1))(x)
        else:
            gd0 = GlobalDescriptor(p=self.p0)(x)
            gd1 = GlobalDescriptor(p=self.p1)(x)
        gd0 = self.linear_g0(gd0); gd1 = self.linear_g1(gd1)
        gd = torch.cat([gd0, gd1], dim=1)
        gd = F.normalize(gd, dim=-1, eps=1e-8)
        return gd, [self.p0, self.p1]


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False, dropout=0.0):
        super().__init__()
        out_channels = in_channels if out_channels is None else out_channels
        self.in_channels, self.out_channels = in_channels, out_channels
        self.use_conv_shortcut = conv_shortcut
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm1 = Normalize(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.norm2 = Normalize(out_channels)
        if in_channels != out_channels:
            if conv_shortcut:
                self.conv_shortcut = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
            else:
                self.nin_shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        h = nonlinearity(self.norm1(self.conv1(x)))
        h = nonlinearity(self.norm2(self.conv2(h)))
        if self.in_channels != self.out_channels:
            x = self.conv_shortcut(x) if self.use_conv_shortcut else self.nin_shortcut(x)
        return x + h


class SimpleDecoder(nn.Module):
    def __init__(self, in_channels, out_channels, upsample_level=2):
        super().__init__()
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=upsample_level, mode="nearest"),
            ResnetBlock(in_channels=in_channels, out_channels=in_channels, dropout=0.0),
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(),
        )

    def forward(self, x):
        return self.dec(x)


# ----------------------------------------------------------------- depth decoder (retargeted)

class DepthDecoder(nn.Module):
    """EchoScan floorplan decoder retargeted from a 16x16->1024x1024 square path to a 4x8->256x512
    ERP path (same up-conv stack + skip branches, only the seed grid and output framing changed)."""

    def __init__(self, seed_h=4, seed_w=8):
        super().__init__()
        self.seed_h, self.seed_w = seed_h, seed_w
        self.reshape_lin = nn.Sequential(nn.Linear(256, 64 * seed_h * seed_w), nn.ReLU6())
        self.dcb1 = SimpleDecoder(128, 64, upsample_level=2)
        self.dcb2 = SimpleDecoder(64, 64, upsample_level=2)
        self.res_dcb2 = SimpleDecoder(128, 1, upsample_level=4)
        self.dcb3 = SimpleDecoder(64, 64, upsample_level=2)
        self.dcb4 = SimpleDecoder(64, 64, upsample_level=2)
        self.res_dcb4 = SimpleDecoder(128, 1, upsample_level=16)
        self.dcb5 = SimpleDecoder(64, 32, upsample_level=2)
        self.dcb6 = SimpleDecoder(32, 32, upsample_level=2)
        self.dcb_out = nn.Sequential(nn.Conv2d(32, 1, 1), nn.Sigmoid())

    def forward(self, x):                                   # x: (B, 512)
        x = torch.stack(torch.chunk(x, 2, dim=1), dim=1)    # (B, 2, 256)
        x = self.reshape_lin(x)
        x = x.reshape(-1, 128, self.seed_h, self.seed_w).contiguous()   # (B, 128, 4, 8)
        identity = x
        x = self.dcb2(self.dcb1(x))
        x = x + self.res_dcb2(identity)
        x = self.dcb4(self.dcb3(x))
        x = x + self.res_dcb4(identity)
        x = self.dcb6(self.dcb5(x))
        return self.dcb_out(x)                              # (B, 1, 256, 512)


class EchoScanDepth(nn.Module):
    """EchoScan waveform encoder + retargeted depth decoder (height head removed).

    forward(wave (B,in_ch,L)) -> depth (B,1,256,512) in [0,1]. Consumes the raw binaural WAVEFORM
    (e.g. cache/rx_wave, (B,2,3200) @ 48 kHz), NOT the STFT spectrogram.
    """

    def __init__(self, in_ch=2, init_ch=32, fs=48000, use_trainable_gdescriptor=False):
        super().__init__()
        conf = SimpleNamespace(model=SimpleNamespace(
            input_ch=in_ch, init_ch=init_ch, fs=fs, use_trainable_gdescriptor=use_trainable_gdescriptor))
        self.encoder = Encoder(conf)
        self.decoder = DepthDecoder()

    def forward(self, wave):
        gd, _ = self.encoder(wave)
        return self.decoder(gd)
