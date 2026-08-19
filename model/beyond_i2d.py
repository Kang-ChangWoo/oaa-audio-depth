"""Beyond-Image-to-Depth baseline (Parida et al., CVPR 2021), audio-only variant.

Faithful port of their full 4-branch model (audio + RGB-depth + material + attention fusion),
with the ONLY change being that the vision input is a fixed constant (all-zeros by default)
instead of a real RGB image -- turning the published audio-visual model into an audio-only
baseline while keeping every branch and the exact fusion intact.

  depth = alpha * audio_depth + (1 - alpha) * img_depth        (alpha from attentionNet)

Resolution handling (see project note): prior work ran at 128x128, ours is 256x512. To keep the
comparison FAIR we predict natively at 256x512 (no post-hoc upsampling that would blur/handicap
the baseline). Two faithful adaptations:
  * audio decoder: their 1x1->128x128 square decoder is reseeded to a 512x2x4 grid so the SAME
    7 up-conv blocks reach 256x512 (2:1 ERP aspect) natively.
  * audio ENCODER input: the spec width (512) is interpolated fake-resolution (real STFT is ~18
    frames), so we slim it to 256x128 before the conv stack -- this only shrinks the flatten->FC
    bottleneck (avoids a ~7M-param waste), it does not change the output resolution.
The RGB/material/attention branches run at the full 256x512 on the constant image.

Modules (unet_conv/unet_upconv/create_conv/weights_init/SimpleAudioDepthNet/RGBDepthNet/
MaterialPropertyNet/attentionNet) are copied verbatim from the repo's models/networks.py except
for the reseed noted above. forward(spec) -> depth (B,1,256,512) in [0,1] (x max_depth later).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# ------------------------------------------------------- verbatim helpers (networks.py)

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    return nn.Sequential(downconv, norm_layer(output_nc), nn.LeakyReLU(0.2, True))


def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    if not outermost:
        return nn.Sequential(upconv, norm_layer(output_nc), nn.ReLU(True))
    return nn.Sequential(upconv, nn.Sigmoid())


def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride=stride, padding=paddings)]
    if batch_norm:
        model.append(nn.BatchNorm2d(output_channels))
    if Relu:
        model.append(nn.ReLU())
    return nn.Sequential(*model)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find("Linear") != -1:
        m.weight.data.normal_(0.0, 0.02)


# ------------------------------------------------------- audio branch (reseeded to 256x512)

class SimpleAudioDepthNet(nn.Module):
    """3-conv CNN -> 1x1 audio feature -> up-conv decoder. Adapted from VisualEchoes (ECCV'20).

    Faithful to the original except the decoder is seeded from a (512,2,4) grid (via a linear
    projection of the 1x1 audio feature) so the 7 unchanged up-conv blocks output 256x512 instead
    of 128x128. `audio_feat` (B,512,1,1) is preserved unchanged for the attention branch.
    """

    def __init__(self, conv1x1_dim, audio_shape, audio_feature_length, output_nc=1,
                 seed_h=2, seed_w=4):
        super().__init__()
        self._n_input_audio = audio_shape[0]
        self._cnn_layers_kernel_size = [(8, 8), (4, 4), (3, 3)]
        self._cnn_layers_stride = [(4, 4), (2, 2), (1, 1)]
        cnn_dims = np.array(audio_shape[1:], dtype=np.float32)
        for k, s in zip(self._cnn_layers_kernel_size, self._cnn_layers_stride):
            cnn_dims = self._conv_output_dim(cnn_dims, np.array([0, 0], np.float32),
                                             np.array([1, 1], np.float32),
                                             np.array(k, np.float32), np.array(s, np.float32))
        self.conv1 = create_conv(self._n_input_audio, 32, self._cnn_layers_kernel_size[0], 0, stride=self._cnn_layers_stride[0])
        self.conv2 = create_conv(32, 64, self._cnn_layers_kernel_size[1], 0, stride=self._cnn_layers_stride[1])
        self.conv3 = create_conv(64, conv1x1_dim, self._cnn_layers_kernel_size[2], 0, stride=self._cnn_layers_stride[2])
        self.feature_extraction = nn.Sequential(self.conv1, self.conv2, self.conv3)
        self.conv1x1 = create_conv(int(conv1x1_dim * cnn_dims[0] * cnn_dims[1]), audio_feature_length, 1, 0)

        self.seed_h, self.seed_w = seed_h, seed_w
        self.seed = nn.Linear(audio_feature_length, 512 * seed_h * seed_w)   # 1x1 feat -> 512x2x4 grid
        self.rgbdepth_upconvlayer1 = unet_upconv(512, 512)
        self.rgbdepth_upconvlayer2 = unet_upconv(512, 256)
        self.rgbdepth_upconvlayer3 = unet_upconv(256, 128)
        self.rgbdepth_upconvlayer4 = unet_upconv(128, 64)
        self.rgbdepth_upconvlayer5 = unet_upconv(64, 32)
        self.rgbdepth_upconvlayer6 = unet_upconv(32, 16)
        self.rgbdepth_upconvlayer7 = unet_upconv(16, output_nc, True)

    def _conv_output_dim(self, dimension, padding, dilation, kernel_size, stride):
        out = []
        for i in range(len(dimension)):
            out.append(int(np.floor((dimension[i] + 2 * padding[i] - dilation[i] * (kernel_size[i] - 1) - 1) / stride[i] + 1)))
        return tuple(out)

    def forward(self, x):
        x = self.feature_extraction(x)
        x = x.view(x.shape[0], -1, 1, 1)
        audio_feat = self.conv1x1(x)                                    # (B, 512, 1, 1)
        seed = self.seed(audio_feat.flatten(1)).view(-1, 512, self.seed_h, self.seed_w)
        h = self.rgbdepth_upconvlayer1(seed)
        h = self.rgbdepth_upconvlayer2(h)
        h = self.rgbdepth_upconvlayer3(h)
        h = self.rgbdepth_upconvlayer4(h)
        h = self.rgbdepth_upconvlayer5(h)
        h = self.rgbdepth_upconvlayer6(h)
        depth_prediction = self.rgbdepth_upconvlayer7(h)                # (B, 1, 256, 512)
        return depth_prediction, audio_feat


# ------------------------------------------------------- vision / material / fusion (verbatim)

class attentionNet(nn.Module):
    def __init__(self, att_out_nc, input_nc):
        super().__init__()
        self.attention_img = nn.Bilinear(512, 512, att_out_nc)
        self.attention_material = nn.Bilinear(512, 512, att_out_nc)
        self.upconvlayer1 = unet_upconv(input_nc, 512)
        self.upconvlayer2 = unet_upconv(512, 256)
        self.upconvlayer3 = unet_upconv(256, 128)
        self.upconvlayer4 = unet_upconv(128, 64)
        self.upconvlayer5 = unet_upconv(64, 1, True)

    def forward(self, rgb_feat, echo_feat, mat_feat):
        rgb_feat = rgb_feat.permute(0, 2, 3, 1).contiguous()
        echo_feat = echo_feat.permute(0, 2, 3, 1).contiguous()
        mat_feat = mat_feat.permute(0, 2, 3, 1).contiguous()
        attentionImg = self.attention_img(rgb_feat, echo_feat).permute(0, 3, 1, 2).contiguous()
        attentionMat = self.attention_material(mat_feat, echo_feat).permute(0, 3, 1, 2).contiguous()
        audioVisual_feature = torch.cat((attentionImg, attentionMat), dim=1)
        h = self.upconvlayer1(audioVisual_feature)
        h = self.upconvlayer2(h)
        h = self.upconvlayer3(h)
        h = self.upconvlayer4(h)
        attention = self.upconvlayer5(h)
        return attention, audioVisual_feature


class RGBDepthNet(nn.Module):
    def __init__(self, ngf=64, input_nc=3, output_nc=1):
        super().__init__()
        self.rgbdepth_convlayer1 = unet_conv(input_nc, ngf)
        self.rgbdepth_convlayer2 = unet_conv(ngf, ngf * 2)
        self.rgbdepth_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.rgbdepth_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.rgbdepth_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.rgbdepth_upconvlayer1 = unet_upconv(512, ngf * 8)
        self.rgbdepth_upconvlayer2 = unet_upconv(ngf * 16, ngf * 4)
        self.rgbdepth_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.rgbdepth_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.rgbdepth_upconvlayer5 = unet_upconv(ngf * 2, output_nc, True)

    def forward(self, x):
        c1 = self.rgbdepth_convlayer1(x)
        c2 = self.rgbdepth_convlayer2(c1)
        c3 = self.rgbdepth_convlayer3(c2)
        c4 = self.rgbdepth_convlayer4(c3)
        c5 = self.rgbdepth_convlayer5(c4)
        u1 = self.rgbdepth_upconvlayer1(c5)
        u2 = self.rgbdepth_upconvlayer2(torch.cat((u1, c4), dim=1))
        u3 = self.rgbdepth_upconvlayer3(torch.cat((u2, c3), dim=1))
        u4 = self.rgbdepth_upconvlayer4(torch.cat((u3, c2), dim=1))
        depth_prediction = self.rgbdepth_upconvlayer5(torch.cat((u4, c1), dim=1))
        return depth_prediction, c5


class MaterialPropertyNet(nn.Module):
    def __init__(self, nclass, backbone):
        super().__init__()
        self.pretrained = backbone
        self.pool = nn.AvgPool2d(4)
        self.fc = nn.Linear(512, nclass)

    def forward(self, x):
        x = self.pretrained.conv1(x)
        x = self.pretrained.bn1(x)
        x = self.pretrained.relu(x)
        x = self.pretrained.maxpool(x)
        x = self.pretrained.layer1(x)
        x = self.pretrained.layer2(x)
        x = self.pretrained.layer3(x)
        feat = self.pretrained.layer4(x)
        x = self.pool(feat)
        x = x.view(-1, 512)
        x = self.fc(x)               # material logits (unused by the audio-only wrapper)
        return x, feat


# ------------------------------------------------------- audio-only wrapper

class BeyondI2DDepth(nn.Module):
    """Full Beyond-I2D model with the vision input pinned to a constant (audio-only baseline).

    forward(spec (B,in_ch,256,512)) -> depth (B,1,256,512) in [0,1].
    The RGB/material branches receive a constant `const_val` image; only the audio path (and the
    attention weighting it drives) responds to the input.
    """

    def __init__(self, in_ch=2, const_val=0.0, audio_enc_hw=(256, 128), max_depth=10.0,
                 material_nclass=23, pretrained_material=True):
        super().__init__()
        self.const_val = float(const_val)
        self.audio_enc_hw = tuple(audio_enc_hw)
        self.net_audio = SimpleAudioDepthNet(8, audio_shape=[in_ch, *self.audio_enc_hw], audio_feature_length=512)
        self.net_rgbdepth = RGBDepthNet(ngf=64, input_nc=3, output_nc=1)
        backbone = torchvision.models.resnet18(weights=(
            torchvision.models.ResNet18_Weights.DEFAULT if pretrained_material else None))
        self.net_material = MaterialPropertyNet(material_nclass, backbone)
        self.net_attention = attentionNet(att_out_nc=512, input_nc=2 * 512)
        # faithful init for the from-scratch audio/rgb/attention branches (material stays pretrained)
        for net in (self.net_audio, self.net_rgbdepth, self.net_attention):
            net.apply(weights_init)

    def forward(self, spec):
        B, _, H, W = spec.shape
        a_in = F.interpolate(spec, size=self.audio_enc_hw, mode="bilinear", align_corners=False)
        audio_depth, audio_feat = self.net_audio(a_in)                  # (B,1,256,512), (B,512,1,1)
        rgb = spec.new_full((B, 3, H, W), self.const_val)               # constant "image"
        img_depth, img_feat = self.net_rgbdepth(rgb)                    # (B,1,256,512), (B,512,8,16)
        _, material_feat = self.net_material(rgb)                       # (B,512,8,16)
        echo_feat = audio_feat.repeat(1, 1, img_feat.shape[-2], img_feat.shape[-1])
        alpha, _ = self.net_attention(img_feat, echo_feat, material_feat)   # (B,1,256,512)
        depth = alpha * audio_depth + (1.0 - alpha) * img_depth        # convex comb -> [0,1]
        return depth
