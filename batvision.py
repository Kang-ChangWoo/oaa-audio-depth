"""BatVision U-Net baseline (channel-stacked conv; the comparison model to OAA).

  "The Audio-Visual BatVision Dataset for Research on Sight and Sound"
  Brunetto, Hornauer, Yu, Moutarde — IROS 2023 (UNetSoundOnly reference impl).

RotDepth = BatVisionUNet(in_ch=2/4/6/8) -> feat_c feature -> 1x1-ish head -> sigmoid depth.
Cleaned reference copy (aux/self-attn branches removed); state-dict compatible with
aux="none" checkpoints from the research repo (e.g. bat_r8_s0).

Test MAE: 2ch 0.920 / 4ch(cB) 0.852 / 6ch 0.808 / 8ch 0.804 — saturates at 6ch
(channel-stacking cannot exploit pose-alignable information; see OAA for the contrast).
"""
import functools
import torch
import torch.nn as nn
from torch.nn import init


def _init_weights(net, init_type="normal", init_gain=0.02):
    def init_func(m):
        name = m.__class__.__name__
        if hasattr(m, "weight") and (name.find("Conv") != -1 or name.find("Linear") != -1):
            if init_type == "normal":
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            if hasattr(m, "bias") and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif name.find("BatchNorm2d") != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)
    net.apply(init_func)


class UNetBlock(nn.Module):
    """Recursive pix2pix-style UNet submodule with skip connection."""

    def __init__(self, outer_nc, inner_nc, input_nc=None, submodule=None,
                 outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, depth_norm=True):
        super().__init__()
        self.outermost = outermost
        use_bias = (norm_layer == nn.InstanceNorm2d or
                    (isinstance(norm_layer, functools.partial) and norm_layer.func == nn.InstanceNorm2d))
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)
        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            up = [uprelu, upconv, nn.Sigmoid() if depth_norm else nn.ReLU()]
            model = [downconv] + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = [downrelu, downconv] + [uprelu, upconv, upnorm]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = [downrelu, downconv, downnorm] + [submodule] + [uprelu, upconv, upnorm]
            if use_dropout:
                model = model + [nn.Dropout(0.5)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        return torch.cat([x, self.model(x)], 1)


class _Cfg:
    class dataset:
        depth_norm = True


class BatVisionUNet(nn.Module):
    """BatVision 8-block UNet for audio-only depth prediction."""

    def __init__(self, cfg=_Cfg, input_nc=2, output_nc=1, num_downs=8, ngf=64, use_dropout=False):
        super().__init__()
        depth_norm = getattr(cfg.dataset, "depth_norm", True)
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
        block = UNetBlock(ngf * 8, ngf * 8, submodule=None, norm_layer=norm_layer,
                          innermost=True, depth_norm=depth_norm)
        for _ in range(num_downs - 5):
            block = UNetBlock(ngf * 8, ngf * 8, submodule=block, norm_layer=norm_layer,
                              use_dropout=use_dropout, depth_norm=depth_norm)
        block = UNetBlock(ngf * 4, ngf * 8, submodule=block, norm_layer=norm_layer, depth_norm=depth_norm)
        block = UNetBlock(ngf * 2, ngf * 4, submodule=block, norm_layer=norm_layer, depth_norm=depth_norm)
        block = UNetBlock(ngf, ngf * 2, submodule=block, norm_layer=norm_layer, depth_norm=depth_norm)
        self.model = UNetBlock(output_nc, ngf, input_nc=input_nc, submodule=block,
                               outermost=True, norm_layer=norm_layer, depth_norm=depth_norm)
        _init_weights(self)

    def forward(self, x):
        return self.model(x)


class RotDepth(nn.Module):
    """Channel-stacked BatVision backbone -> depth head. in_ch = number of orientation-ear channels."""

    def __init__(self, in_ch, feat_c=32, ngf=64):
        super().__init__()
        self.enc = BatVisionUNet(_Cfg, input_nc=in_ch, output_nc=feat_c, ngf=ngf)
        self.register_buffer("reference", torch.zeros(1))     # ckpt-compat (unused in baseline)
        self.head = nn.Conv2d(feat_c, 1, 3, padding=1)

    def forward(self, spec):
        """spec (B, in_ch, 256, 512) -> depth (B,1,256,512) in [0,1] (×max_depth = m)."""
        return torch.sigmoid(self.head(self.enc(spec)))
