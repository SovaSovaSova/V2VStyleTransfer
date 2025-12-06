import torch
import torch.nn as nn
from torchvision import models

# --- AdaIN Layer (The Core of Style Transfer) ---


def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def adain(content_feat, style_feat):
    """
    Adaptive Instance Normalization
    Aligns the mean and std of content_feat to match style_feat.
    """
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    style_mean, style_std = calc_mean_std(style_feat)
    content_mean, content_std = calc_mean_std(content_feat)

    normalized_feat = (content_feat - content_mean) / content_std
    return normalized_feat * style_std + style_mean

# --- Encoder (Fixed VGG) ---


class VGGEncoder(nn.Module):
    def __init__(self):
        super(VGGEncoder, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        # CCPL uses up to relu4_1
        self.slice1 = vgg[:2]   # relu1_1
        self.slice2 = vgg[2:7]  # relu2_1
        self.slice3 = vgg[7:12]  # relu3_1
        self.slice4 = vgg[12:21]  # relu4_1

        # Register normalization buffers
        self.register_buffer("mean", torch.tensor(
            [0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(
            [0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        # Auto-normalize input [0, 1] -> ImageNet Standard
        x = (x - self.mean) / self.std

        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        h4 = self.slice4(h3)
        return h1, h2, h3, h4

# --- Decoder (Symmetric to VGG) ---


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # Inverse of VGG-19 relu4_1
        self.rc1 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(512, 256, 3, 1, 0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.rc2 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 256, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(256, 128, 3, 1, 0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.rc3 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 128, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(128, 64, 3, 1, 0),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest')
        )
        self.rc4 = nn.Sequential(
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 64, 3, 1, 0),
            nn.ReLU(),
            nn.ReflectionPad2d((1, 1, 1, 1)),
            nn.Conv2d(64, 3, 3, 1, 0)
        )

    def forward(self, x):
        x = self.rc1(x)
        x = self.rc2(x)
        x = self.rc3(x)
        x = self.rc4(x)
        return x

# --- Unified Network ---


class AdaINStyleTransferNetwork(nn.Module):
    def __init__(self, encoder):
        super(AdaINStyleTransferNetwork, self).__init__()
        self.encoder = encoder
        self.decoder = Decoder()

    def forward(self, content, style, alpha=1.0):
        # 1. Encode both
        c_feats = self.encoder(content)
        s_feats = self.encoder(style)

        # Use the deepest feature (relu4_1) for AdaIN
        content_feat = c_feats[-1]
        style_feat = s_feats[-1]

        # 2. AdaIN Transformation
        # t is the target feature map
        t = adain(content_feat, style_feat)

        # Alpha control (interpolate between content and stylized features)
        t = alpha * t + (1 - alpha) * content_feat

        # 3. Decode
        g_t = self.decoder(t)

        return g_t, t, content_feat, style_feat
