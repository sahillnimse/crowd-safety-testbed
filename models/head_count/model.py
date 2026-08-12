"""
Density-map head counter.

Architecture
------------
ResNet-18 frontend truncated after layer3, with layer3's stride replaced by
dilation so the output is stride 8 rather than 16, then a small dilated
backend and a 1x1 head producing a single-channel density map.

    input   (B, 3, H, W)
    output  (B, 1, H/8, W/8)   non-negative; sum over any region = head count

Why a density map rather than boxes or points
---------------------------------------------
This exists to answer "how many people are in this grid cell", which is the
rho in Helbing crowd pressure.  A density map answers that by integration
over the cell — no detection, no NMS, no identity, and no threshold to tune.
That matters because the regime where the count is most needed is exactly
the one where box detectors collapse: at the back of a dense crowd bodies
are 90% occluded and boxes merge, while the summed density is still a
count.  Point localisation is recoverable from local maxima when it is
wanted (see infer.py), but the count never depends on it.

Why stride 8
------------
At stride 16 a 1280x720 frame yields an 80x45 map — coarser than the 16 px
metrics grid the density has to be aggregated onto, so cells would have to
share map pixels and the per-cell counts would be interpolation rather than
measurement.  Stride 8 gives 160x90, two map pixels per 16 px cell in each
direction, so every metrics cell integrates over its own values.

Why ResNet-18 and not something larger
--------------------------------------
It has to fine-tune on a 4 GB card.  ResNet-18 at stride 8 with 512x512
crops fits in roughly 2 GB at batch 8; ResNet-50 does not fit at any useful
batch size, and gradient accumulation on a batch of 1 makes batch-norm
statistics useless.  The frontend is ImageNet-pretrained, which is what
makes training on a few hundred hand-labelled patches viable at all.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Every part of the pipeline agrees on this: the dataset builds targets at
# this stride, the model outputs at it, and inference scales by it.
OUTPUT_STRIDE = 8


class HeadCountNet(nn.Module):
    """ResNet-18 frontend + dilated backend -> single-channel density map."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        import torchvision.models as tvm

        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        net = tvm.resnet18(weights=weights)

        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1            # stride 4,  64 ch
        self.layer2 = net.layer2            # stride 8, 128 ch
        self.layer3 = net.layer3            # stride 16 -> dilated to 8, 256 ch

        # Replace layer3's downsampling with dilation.  Striding here would
        # halve the output grid again; dilation keeps the receptive field
        # that the pretrained weights expect while holding the resolution.
        for module in self.layer3.modules():
            if isinstance(module, nn.Conv2d):
                if module.stride == (2, 2):
                    module.stride = (1, 1)
                if module.kernel_size == (3, 3):
                    module.dilation = (2, 2)
                    module.padding = (2, 2)

        self.backend = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1), nn.ReLU(inplace=True),
        )
        self.head = nn.Conv2d(32, 1, 1)

        for m in list(self.backend.modules()) + [self.head]:
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.backend(x)
        x = self.head(x)
        # Density is a count per unit area and cannot be negative.  softplus
        # rather than relu: relu's zero gradient on the whole negative side
        # lets a head that initialises slightly negative stop learning
        # entirely on the empty patches, which are most of a crowd image.
        return F.softplus(x)

    @torch.no_grad()
    def count(self, x: torch.Tensor) -> torch.Tensor:
        """Predicted head count per image in the batch."""
        return self.forward(x).sum(dim=(1, 2, 3))


class CountLoss(nn.Module):
    """
    Pixelwise density loss plus a direct count term.

    The pixel term alone optimises a number that is ~0 almost everywhere, so
    a model that predicts all zeros starts at a very low loss and the count
    can be badly wrong while the MSE looks converged.  The count term is what
    the metric actually is, and weighting it explicitly stops the optimiser
    from being satisfied by an empty prediction.
    """

    def __init__(self, count_weight: float = 0.1, pixel_scale: float = 100.0) -> None:
        super().__init__()
        self.count_weight = count_weight
        # Targets sum to the head count spread over thousands of pixels, so
        # raw values are ~1e-3 and the MSE gradient is negligible against
        # float32 noise.  Scaling both sides puts it in a trainable range.
        self.pixel_scale = pixel_scale

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> tuple:
        pixel = F.mse_loss(pred * self.pixel_scale, target * self.pixel_scale)
        pred_n = pred.sum(dim=(1, 2, 3))
        true_n = target.sum(dim=(1, 2, 3))
        count = F.l1_loss(pred_n, true_n)
        return pixel + self.count_weight * count, pixel.detach(), count.detach()
