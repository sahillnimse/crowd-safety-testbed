"""
DM-Count density-map head counter — VGG19 backbone, density head.

Vendored architecture from DM-Count (Wang et al., AAAI 2021), matching
``DM-Count-Kit/dmcount/models.py`` in the source workspace byte-for-byte in
every layer shape and forward behaviour. The checkpoint this loads is a bare
state dict whose keys are tied to exactly these module names, so nothing here
may be renamed or reordered:

    input   (B, 3, H, W), ImageNet-normalised, H and W multiples of 16
    output  mu        (B, 1, H/8, W/8) density map — sums to the head count
            mu_normed (B, 1, H/8, W/8) row-stochastic copy of mu

Stride arithmetic: VGG19's four MaxPool2d(2) stages give H/16, then the
forward upsamples x2 bilinearly before the regression head, so the density
grid is at stride 8. Callers must zero-pad inputs to a multiple of 16 and
crop the output back to valid_h//8 x valid_w//8 — the ReLU density head
emits small positive values along a pad seam that would otherwise inflate
the count.

LOCAL PATCH (carried over from the source workspace): ``pretrained`` is
switchable and defaults to False. Upstream always pulled the ~550 MB ImageNet
VGG19 from torch's model zoo even though the DM-Count checkpoint loaded a
line later overwrites every tensor — pure download for no effect. We never
want a surprise network fetch inside a video pipeline.
"""

from __future__ import annotations

import torch.nn as nn
from torch.nn import functional as F

__all__ = ["vgg19", "INPUT_STRIDE"]

#: Density-map stride relative to the padded input (4 pools = /16, x2 upsample).
INPUT_STRIDE = 8


class VGG(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features
        self.reg_layer = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.density_layer = nn.Sequential(nn.Conv2d(128, 1, 1), nn.ReLU())

    def forward(self, x):
        x = self.features(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x = self.reg_layer(x)
        mu = self.density_layer(x)
        B, C, H, W = mu.size()
        mu_sum = mu.view([B, -1]).sum(1).unsqueeze(1).unsqueeze(2).unsqueeze(3)
        mu_normed = mu / (mu_sum + 1e-6)
        return mu, mu_normed


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M',
          512, 512, 512, 512, 'M', 512, 512, 512, 512]
}


def vgg19(pretrained=False):
    """VGG 19-layer model (configuration "E") with the DM-Count density head."""
    model = VGG(make_layers(cfg['E']))
    # `pretrained` intentionally does nothing beyond the flag check: keeping
    # the parameter preserves call-site compatibility with upstream code, but
    # downloading ImageNet weights here is never useful because every caller
    # immediately loads a full DM-Count checkpoint over the top.
    return model
