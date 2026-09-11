"""A small residual U-Net that learns the low-emissivity surface correction.

Design rationale (why residual, why small)
------------------------------------------
* On the ~62 % of the board that is trusted, the correct output *is* the input
  field: the bare board reads the truth there.  A plain U-Net would spend most
  of its capacity learning to copy pixels and could drift on them.  This network
  therefore predicts only a correction, and the prediction is

      predicted theta = input theta + net(input theta, masks).

  Capacity then concentrates on the footprint pixels (38 % of the board), which
  is the actual problem.
* The board is 157 x 103 pixels and the training archive is 240 frames from one
  bare/painted pair: a few hundred thousand parameters are the right size, and
  the fit stays cheap on a CPU (the same local-training decision the project
  made for the Stage 3 plate model).
* Temperature is passed in a fixed unit (theta / 60 K, so the network input is
  ~0..0.9 for this campaign) - never per-map normalisation, because the size of
  the artifact correction genuinely depends on absolute temperature.

The correction is a temperature in kelvin, nothing else.  No emissivity,
conductivity, or power is produced anywhere in this module.
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as Fn


class _ConvBlock(nn.Module):
    """Two same-padding convolutions with group norm and SiLU."""

    def __init__(self, in_channels: int, channels: int, groups: int = 8) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels),
            nn.SiLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualUNet(nn.Module):
    """Predicts a temperature correction; add it to the input theta field."""

    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 24,
        depth: int = 3,
        groups: int = 8,
        divisor: int = 8,
    ) -> None:
        super().__init__()
        self.base = int(base_channels)
        self.depth = int(depth)
        self.divisor = int(divisor)
        channels = [self.base * (2 ** level) for level in range(self.depth)]

        self.head = nn.Sequential(
            nn.Conv2d(in_channels, channels[0], 3, padding=1, bias=False),
            nn.GroupNorm(groups, channels[0]),
            nn.SiLU(),
        )
        self.encoders = nn.ModuleList([_ConvBlock(c, c, groups) for c in channels])
        self.downsamples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels[level], channels[level + 1], 3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(groups, channels[level + 1]),
                    nn.SiLU(),
                )
                for level in range(self.depth - 1)
            ]
        )
        self.bottleneck = _ConvBlock(channels[-1], channels[-1], groups)
        self.upsamples = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(channels[level + 1], channels[level], 2, stride=2, bias=False),
                    nn.GroupNorm(groups, channels[level]),
                    nn.SiLU(),
                )
                for level in reversed(range(self.depth - 1))
            ]
        )
        # Each decoder consumes the previous coarse feature concatenated with the
        # same-scale encoder skip: 2 * channels[level] input channels.
        self.decoders = nn.ModuleList(
            [_ConvBlock(2 * channels[level], channels[level], groups) for level in reversed(range(self.depth - 1))]
        )
        self.tail = nn.Conv2d(channels[0], 1, 1)

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        h, w = x.shape[-2:]
        ph = (self.divisor - h % self.divisor) % self.divisor
        pw = (self.divisor - w % self.divisor) % self.divisor
        if ph or pw:
            x = Fn.pad(x, (0, pw, 0, ph), mode="reflect")
        return x, (ph, pw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the correction field in the same units as ``x``."""
        x, (ph, pw) = self._pad(x)
        # feats layout after the encode loop (depth 3 example):
        #   [head0, enc0, down0, enc1, down1, ...]; the skip for decoder
        #   level L is feats[2 * L + 1] (the encoder output at scale L).
        feats = [self.head(x)]
        for level in range(self.depth - 1):
            feats.append(self.encoders[level](feats[-1]))
            feats.append(self.downsamples[level](feats[-1]))
        latent = self.bottleneck(feats[-1])
        for step, level in enumerate(reversed(range(self.depth - 1))):
            latent = self.upsamples[step](latent)
            skip = feats[2 * level + 1]
            latent = self.decoders[step](torch.cat([latent, skip], dim=1))
        correction = self.tail(latent)
        if ph or pw:
            correction = correction[:, :, : x.shape[-2] - ph, : x.shape[-1] - pw]
        return correction

    def n_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())


def build_residual_unet(in_channels: int = 3, base_channels: int = 24, depth: int = 3) -> ResidualUNet:
    """Construct the default small correction network."""
    return ResidualUNet(in_channels=in_channels, base_channels=base_channels, depth=depth)
