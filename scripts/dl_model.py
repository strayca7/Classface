"""
U-Net with ResNet-18 encoder for face foreground segmentation.

Architecture:
  Encoder: ResNet-18 (ImageNet pretrained via torchvision)
           → 5 feature maps: stem(56²), layer1(28²), layer2(14²), layer3(7²), layer4(4²)
  Decoder: Bilinear upsample + skip concat + ConvBNReLU×2 at each level
  Output : 1×112×112 sigmoid mask (binary foreground)

Device priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU
"""

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Return the best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Device: CUDA — %s", torch.cuda.get_device_name(0))
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Device: Apple Silicon MPS")
    else:
        device = torch.device("cpu")
        logger.info("Device: CPU (no GPU detected)")
    return device


class _ConvBlock(nn.Module):
    """Conv3×3 → BN → ReLU, repeated twice."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResUNet(nn.Module):
    """
    ResNet-18 encoder + U-Net decoder for binary face segmentation.

    Input : (B, 3, 112, 112) float32, normalised to ImageNet mean/std
    Output: (B, 1, 112, 112) float32, sigmoid probability map
    """

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()

        # ── Encoder: ResNet-18 ──────────────────────────────────────────
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        backbone = models.resnet18(weights=weights)

        # Stem: conv7×7 + bn + relu  →  (B, 64, 56, 56)
        self.enc_stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu
        )
        self.enc_pool = backbone.maxpool  # → (B, 64, 28, 28)
        self.enc1 = backbone.layer1  # → (B,  64, 28, 28)
        self.enc2 = backbone.layer2  # → (B, 128, 14, 14)
        self.enc3 = backbone.layer3  # → (B, 256,  7,  7)
        self.enc4 = backbone.layer4  # → (B, 512,  4,  4)

        # ── Decoder ─────────────────────────────────────────────────────
        # Each level: upsample bottleneck → concat skip → ConvBlock
        self.dec4 = _ConvBlock(512 + 256, 256)   # enc4 up + enc3
        self.dec3 = _ConvBlock(256 + 128, 128)   # dec4 up + enc2
        self.dec2 = _ConvBlock(128 + 64, 64)     # dec3 up + enc1
        self.dec1 = _ConvBlock(64 + 64, 32)      # dec2 up + stem (before pool)

        # Final upsample stem→112 and classify
        self.head = nn.Sequential(
            nn.Conv2d(32, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        s0 = self.enc_stem(x)          # 64 × 56 × 56
        e1 = self.enc1(self.enc_pool(s0))  # 64 × 28 × 28
        e2 = self.enc2(e1)             # 128 × 14 × 14
        e3 = self.enc3(e2)             # 256 ×  7 ×  7
        e4 = self.enc4(e3)             # 512 ×  4 ×  4

        # Decode with skip connections (bilinear resize to match skip size)
        d4 = self.dec4(torch.cat([_up(e4, e3), e3], dim=1))   # 256 × 7 × 7
        d3 = self.dec3(torch.cat([_up(d4, e2), e2], dim=1))   # 128 × 14 × 14
        d2 = self.dec2(torch.cat([_up(d3, e1), e1], dim=1))   # 64  × 28 × 28
        d1 = self.dec1(torch.cat([_up(d2, s0), s0], dim=1))   # 32  × 56 × 56

        # Upsample to original input size (112×112) and produce mask
        out = _up(d1, x)               # 32 × 112 × 112
        return torch.sigmoid(self.head(out))  # 1 × 112 × 112


def _up(src: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Bilinear upsample `src` to the spatial size of `ref`."""
    return F.interpolate(src, size=ref.shape[2:], mode="bilinear", align_corners=False)
