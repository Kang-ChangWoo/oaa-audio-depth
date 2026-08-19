"""Pretrained-backbone baselines: ImageNet ResNet-50 / ViT-B-16 adapted to audio depth.

The "simple" comparison the OAA README alludes to: take an off-the-shelf pretrained vision
backbone and change only the input stem so it eats an `in_ch`-channel spectrogram instead of a
3-channel RGB image. A learnable 1x1 conv projects `in_ch -> 3` (pseudo-RGB), everything after
is the standard ImageNet-pretrained encoder + a light depth decoder.

  PretrainedResNet(in_ch)  ResNet-50 encoder + FPN-style skip decoder
  PretrainedViT(in_ch)     ViT-B/16 encoder (pos-embeds interpolated to 16x32) + convT decoder

forward(spec (B,in_ch,256,512)) -> depth (B,1,256,512) in [0,1] (x max_depth later).

Ported from baseline/models/pretrain/{pretrained_resnet,pretrained_vit}.py, with the SimpleNamespace
cfg dependency removed and the input adapter generalised from 2ch to arbitrary in_ch so the same
model runs across the r2/cB/r6/r8 modes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights


# --------------------------------------------------------------------------- ResNet-50

class _ResNetDecoder(nn.Module):
    """Progressive upsampling decoder with skip connections from the ResNet encoder."""

    def __init__(self):
        super().__init__()
        self.reduce4 = nn.Sequential(nn.Conv2d(2048, 512, 1), nn.BatchNorm2d(512), nn.ReLU(True))
        self.up3 = self._up_block(512 + 1024, 256)
        self.up2 = self._up_block(256 + 512, 128)
        self.up1 = self._up_block(128 + 256, 64)
        self.up0 = self._up_block(64 + 64, 32)
        self.head = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(True), nn.Conv2d(16, 1, 1))

    @staticmethod
    def _up_block(in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True))

    def forward(self, feats, target_h, target_w):
        x = self.reduce4(feats["layer4"])
        for skip, up in ((feats["layer3"], self.up3), (feats["layer2"], self.up2),
                         (feats["layer1"], self.up1), (feats["stem"], self.up0)):
            x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=False)
            x = up(torch.cat([x, skip], dim=1))
        x = torch.sigmoid(self.head(x))
        if x.shape[2] != target_h or x.shape[3] != target_w:
            x = F.interpolate(x, (target_h, target_w), mode="bilinear", align_corners=False)
        return x


class PretrainedResNet(nn.Module):
    """ImageNet ResNet-50 encoder + multi-scale decoder for audio->depth."""

    def __init__(self, in_ch=2, pretrained=True, freeze_encoder=False):
        super().__init__()
        self.input_adapter = nn.Conv2d(in_ch, 3, kernel_size=1)          # spec -> pseudo-RGB
        resnet = tv_models.resnet50(weights=ResNet50_Weights.DEFAULT if pretrained else None)
        self.stem = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1, self.layer2 = resnet.layer1, resnet.layer2
        self.layer3, self.layer4 = resnet.layer3, resnet.layer4
        if freeze_encoder:
            for mod in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
                for p in mod.parameters():
                    p.requires_grad = False
        self.decoder = _ResNetDecoder()

    def forward(self, spec):
        h, w = spec.shape[2], spec.shape[3]
        x = self.input_adapter(spec)
        s = self.stem(x)
        l1 = self.layer1(s); l2 = self.layer2(l1); l3 = self.layer3(l2); l4 = self.layer4(l3)
        return self.decoder({"stem": s, "layer1": l1, "layer2": l2, "layer3": l3, "layer4": l4}, h, w)


# --------------------------------------------------------------------------- ViT-B/16

class _ViTDecoder(nn.Module):
    """Progressive convT upsampling from the patch-token grid to a dense depth map."""

    def __init__(self, embed_dim, grid_h, grid_w):
        super().__init__()
        self.grid_h, self.grid_w = grid_h, grid_w
        ch = embed_dim
        layers = []
        for out_ch in (256, 128, 64, 32):
            layers.append(nn.Sequential(
                nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1), nn.BatchNorm2d(out_ch), nn.ReLU(True)))
            ch = out_ch
        self.up = nn.ModuleList(layers)
        self.head = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.ReLU(True), nn.Conv2d(16, 1, 1))

    def forward(self, tokens, target_h, target_w):
        B, N, C = tokens.shape
        x = tokens.transpose(1, 2).reshape(B, C, self.grid_h, self.grid_w)
        for layer in self.up:
            x = layer(x)
        x = torch.sigmoid(self.head(x))
        if x.shape[2] != target_h or x.shape[3] != target_w:
            x = F.interpolate(x, (target_h, target_w), mode="bilinear", align_corners=False)
        return x


def _interpolate_pos_embed(pos_embed, old_grid, new_grid):
    """Interpolate ViT positional embeddings (CLS + patch tokens) to a new grid size."""
    cls_token, patch_pos = pos_embed[:, :1, :], pos_embed[:, 1:, :]
    D = patch_pos.shape[-1]
    oH, oW = old_grid; nH, nW = new_grid
    patch_pos = patch_pos.reshape(1, oH, oW, D).permute(0, 3, 1, 2)
    patch_pos = F.interpolate(patch_pos, size=(nH, nW), mode="bicubic", align_corners=False)
    patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, nH * nW, D)
    return torch.cat([cls_token, patch_pos], dim=1)


class PretrainedViT(nn.Module):
    """ImageNet ViT-B/16 for audio->depth (pos-embeds interpolated to the 256x512 grid at init)."""

    def __init__(self, in_ch=2, target_h=256, target_w=512, pretrained=True, freeze_encoder=False):
        super().__init__()
        self.target_h, self.target_w = target_h, target_w
        patch_size, embed_dim = 16, 768
        self.input_adapter = nn.Conv2d(in_ch, 3, kernel_size=1)
        vit = tv_models.vit_b_16(weights=ViT_B_16_Weights.DEFAULT if pretrained else None)
        self.patch_embed = vit.conv_proj          # Conv2d(3, 768, 16, 16)
        self.cls_token = vit.class_token
        self.encoder = vit.encoder
        self.grid_h, self.grid_w = target_h // patch_size, target_w // patch_size   # 16, 32
        new_pos = _interpolate_pos_embed(vit.encoder.pos_embedding.data, (14, 14), (self.grid_h, self.grid_w))
        self.encoder.pos_embedding = nn.Parameter(new_pos)
        if freeze_encoder:
            for p in self.patch_embed.parameters():
                p.requires_grad = False
            for p in self.encoder.parameters():
                p.requires_grad = False
            self.cls_token.requires_grad = False
        self.decoder = _ViTDecoder(embed_dim, self.grid_h, self.grid_w)

    def forward(self, spec):
        h, w = spec.shape[2], spec.shape[3]
        x = self.input_adapter(spec)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)          # (B, n_patches, 768)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.encoder(x)
        return self.decoder(x[:, 1:, :], h, w)
