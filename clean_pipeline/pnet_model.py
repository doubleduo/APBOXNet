"""Clean RGB-only ZoomNeXt PNet.

The network always receives one RGB tensor. Multi-scale images and the optional
Sobel cue are created inside the model, so evaluation never needs Box/GT/edge
files. ``edge_mode`` controls only an internal auxiliary branch:

* ``none``  - mask prediction only;
* ``aux``   - predict an auxiliary boundary during training;
* ``sobel`` - derive a deterministic RGB Sobel cue and predict a boundary.
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Sequence, Tuple

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelNormalizer(nn.Module):
    def __init__(
        self,
        mean: Sequence[float] = (0.485, 0.456, 0.406),
        std: Sequence[float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(std).view(1, 3, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class ConvBNAct(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        activate: bool = True,
    ) -> None:
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        ]
        if activate:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class SimpleASPP(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.local = nn.ModuleList(
            [
                ConvBNAct(in_channels, out_channels, 1),
                ConvBNAct(in_channels, out_channels, 3, padding=3, dilation=3),
                ConvBNAct(in_channels, out_channels, 3, padding=6, dilation=6),
                ConvBNAct(in_channels, out_channels, 3, padding=9, dilation=9),
            ]
        )
        # No BatchNorm after 1x1 global pooling: batch-size 1 remains valid.
        self.global_proj = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.fuse = ConvBNAct(out_channels * 5, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = self.global_proj(F.adaptive_avg_pool2d(x, 1))
        pooled = F.interpolate(
            pooled, x.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.fuse(torch.cat([pooled] + [branch(x) for branch in self.local], 1))


class MultiScaleIntegration(nn.Module):
    """Per-position, per-channel-group competition over three image scales."""

    def __init__(self, channels: int, groups: int = 4) -> None:
        super().__init__()
        if channels % groups:
            raise ValueError(f"channels={channels} must be divisible by groups={groups}")
        self.groups = int(groups)
        self.channels_per_group = channels // groups
        self.pre = nn.ModuleList(
            [ConvBNAct(channels, channels, 3, padding=1) for _ in range(3)]
        )
        self.attention = nn.Sequential(
            ConvBNAct(channels * 3, channels, 1),
            ConvBNAct(channels, channels, 3, padding=1),
            nn.Conv2d(channels, groups * 3, 1),
        )
        self.out = ConvBNAct(channels, channels, 3, padding=1)

    def forward(
        self,
        large: torch.Tensor,
        medium: torch.Tensor,
        small: torch.Tensor,
    ) -> torch.Tensor:
        target_size = medium.shape[-2:]
        large = F.adaptive_avg_pool2d(large, target_size) + F.adaptive_max_pool2d(
            large, target_size
        )
        small = F.interpolate(small, target_size, mode="bilinear", align_corners=False)
        features = [layer(x) for layer, x in zip(self.pre, (large, medium, small))]

        batch, channels, height, width = medium.shape
        weights = self.attention(torch.cat(features, dim=1))
        weights = weights.view(batch, self.groups, 3, height, width).softmax(dim=2)
        stacked = torch.stack(
            [
                feat.view(
                    batch, self.groups, self.channels_per_group, height, width
                )
                for feat in features
            ],
            dim=2,
        )
        fused = (weights.unsqueeze(3) * stacked).sum(dim=2)
        return self.out(fused.reshape(batch, channels, height, width))


class RGPU(nn.Module):
    """Compact recursive gated processing unit used by the top-down decoder."""

    def __init__(self, channels: int, groups: int = 4) -> None:
        super().__init__()
        if channels % groups:
            raise ValueError(f"channels={channels} must be divisible by groups={groups}")
        width = channels // groups
        self.groups = groups
        self.blocks = nn.ModuleList(
            [
                ConvBNAct(width if i == 0 else width * 2, width, 3, padding=1)
                for i in range(groups)
            ]
        )
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, max(channels // 2, 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(channels // 2, 8), channels, 1),
            nn.Sigmoid(),
        )
        self.fuse = ConvBNAct(channels, channels, 3, padding=1, activate=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        chunks = x.chunk(self.groups, dim=1)
        outputs = []
        previous = None
        for chunk, block in zip(chunks, self.blocks):
            current = chunk if previous is None else torch.cat((chunk, previous), dim=1)
            previous = block(current)
            outputs.append(previous)
        merged = torch.cat(outputs, dim=1)
        return F.relu(x + self.fuse(merged * (2.0 * self.gate(merged))), inplace=True)


class ZoomNeXtPNet(nn.Module):
    """Shared-PVT three-scale ZoomNeXt; final inference input is RGB only."""

    EDGE_MODES = {"none", "aux", "sobel"}

    def __init__(
        self,
        backbone_name: str = "pvt_v2_b2",
        pretrained: bool = True,
        channels: int = 64,
        scale_factors: Sequence[float] = (1.5, 1.0, 0.5),
        siu_groups: int = 4,
        rgpu_groups: int = 4,
        gradient_checkpointing: bool = False,
        edge_mode: str = "aux",
    ) -> None:
        super().__init__()
        if len(scale_factors) != 3 or not (
            scale_factors[0] > scale_factors[1] > scale_factors[2] > 0
        ):
            raise ValueError("scale_factors must be ordered large > medium > small > 0")
        edge_mode = str(edge_mode).lower()
        if edge_mode not in self.EDGE_MODES:
            raise ValueError(f"edge_mode must be one of {sorted(self.EDGE_MODES)}")

        self.scale_factors = tuple(float(x) for x in scale_factors)
        self.edge_mode = edge_mode
        self.normalizer = PixelNormalizer()
        self.encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        if gradient_checkpointing and hasattr(self.encoder, "set_grad_checkpointing"):
            self.encoder.set_grad_checkpointing(enable=True)

        encoder_channels = list(self.encoder.feature_info.channels())
        if len(encoder_channels) != 4:
            raise RuntimeError(f"Expected four PVT stages, got {encoder_channels}")

        self.transitions = nn.ModuleList(
            [
                ConvBNAct(encoder_channels[0], channels, 3, padding=1),
                ConvBNAct(encoder_channels[1], channels, 3, padding=1),
                ConvBNAct(encoder_channels[2], channels, 3, padding=1),
                SimpleASPP(encoder_channels[3], channels),
            ]
        )
        self.scale_fusion = nn.ModuleList(
            [MultiScaleIntegration(channels, siu_groups) for _ in range(4)]
        )
        self.decoder = nn.ModuleList([RGPU(channels, rgpu_groups) for _ in range(4)])
        self.mask_heads = nn.ModuleList([nn.Conv2d(channels, 1, 1) for _ in range(4)])

        if edge_mode == "sobel":
            self.sobel_fuse = ConvBNAct(channels + 1, channels, 3, padding=1)
        else:
            self.sobel_fuse = None

        self.final_refine = nn.Sequential(
            ConvBNAct(channels, channels, 3, padding=1),
            ConvBNAct(channels, channels // 2, 3, padding=1),
        )
        self.final_mask = nn.Conv2d(channels // 2, 1, 1)
        self.edge_head = (
            nn.Conv2d(channels // 2, 1, 1) if edge_mode != "none" else None
        )

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x, persistent=False)
        self.register_buffer("sobel_y", sobel_x.transpose(-1, -2), persistent=False)

    @staticmethod
    def _resize_image(image: torch.Tensor, scale: float) -> torch.Tensor:
        if abs(scale - 1.0) < 1e-8:
            return image
        height = max(32, int(round(image.shape[-2] * scale)))
        width = max(32, int(round(image.shape[-1] * scale)))
        return F.interpolate(
            image, (height, width), mode="bilinear", align_corners=False
        )

    def _encode(self, image: torch.Tensor) -> Sequence[torch.Tensor]:
        return list(self.encoder(self.normalizer(image)))

    def _rgb_sobel(self, image: torch.Tensor) -> torch.Tensor:
        gray = (
            image[:, 0:1] * 0.2989
            + image[:, 1:2] * 0.5870
            + image[:, 2:3] * 0.1140
        )
        gx = F.conv2d(gray, self.sobel_x, padding=1)
        gy = F.conv2d(gray, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx.square() + gy.square() + 1e-6)
        return magnitude / magnitude.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)

    def forward(self, image: torch.Tensor) -> Dict[str, object]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"image must have shape [B,3,H,W], got {tuple(image.shape)}")
        output_size = image.shape[-2:]
        multi_scale_features = [
            self._encode(self._resize_image(image, scale))
            for scale in self.scale_factors
        ]

        fused = []
        for stage in range(4):
            scale_feats = [
                self.transitions[stage](features[stage])
                for features in multi_scale_features
            ]
            fused.append(self.scale_fusion[stage](*scale_feats))

        decoded = [None] * 4
        current = None
        for stage in range(3, -1, -1):
            feature = fused[stage]
            if current is not None:
                current = F.interpolate(
                    current, feature.shape[-2:], mode="bilinear", align_corners=False
                )
                feature = feature + current
            current = self.decoder[stage](feature)
            decoded[stage] = current

        if self.sobel_fuse is not None:
            sobel = F.interpolate(
                self._rgb_sobel(image),
                decoded[0].shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            decoded[0] = self.sobel_fuse(torch.cat((decoded[0], sobel), dim=1))

        auxiliary = [
            F.interpolate(
                self.mask_heads[stage](decoded[stage]),
                output_size,
                mode="bilinear",
                align_corners=False,
            )
            for stage in range(3, -1, -1)
        ]
        refined = self.final_refine(
            F.interpolate(decoded[0], scale_factor=2.0, mode="bilinear", align_corners=False)
        )
        final_mask = F.interpolate(
            self.final_mask(refined), output_size, mode="bilinear", align_corners=False
        )
        edge_logits = None
        if self.edge_head is not None:
            edge_logits = F.interpolate(
                self.edge_head(refined), output_size, mode="bilinear", align_corners=False
            )

        return {
            "mask_logits": tuple(auxiliary) + (final_mask,),
            "edge_logits": edge_logits,
        }

    @torch.no_grad()
    def predict(self, image: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(image)["mask_logits"][-1])

    def parameter_groups(self) -> Tuple[Iterable[nn.Parameter], Iterable[nn.Parameter]]:
        return self.encoder.parameters(), (
            param
            for name, param in self.named_parameters()
            if not name.startswith("encoder.")
        )


def structure_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weight = 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
    )
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (weight * bce).sum((2, 3)) / weight.sum((2, 3)).clamp_min(1e-6)
    probability = torch.sigmoid(logits)
    intersection = (probability * target * weight).sum((2, 3))
    union = ((probability + target) * weight).sum((2, 3))
    wiou = 1.0 - (intersection + 1.0) / (union - intersection + 1.0)
    return (bce + wiou).mean()


def soft_pseudo_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
) -> torch.Tensor:
    confidence = confidence.clamp(0.0, 1.0)
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce * confidence).sum((1, 2, 3)) / confidence.sum((1, 2, 3)).clamp_min(1.0)
    probability = torch.sigmoid(logits)
    intersection = (probability * target * confidence).sum((1, 2, 3))
    union = ((probability + target) * confidence).sum((1, 2, 3))
    wiou = 1.0 - (intersection + 1.0) / (union - intersection + 1.0)
    return (bce + wiou).mean()


def edge_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor | None = None,
) -> torch.Tensor:
    probability = torch.sigmoid(logits)
    if confidence is None:
        confidence = torch.ones_like(target)
    probability = (probability * confidence).flatten(1)
    target = (target * confidence).flatten(1)
    numerator = 2.0 * (probability * target).sum(1) + 1.0
    denominator = probability.square().sum(1) + target.square().sum(1) + 1.0
    return (1.0 - numerator / denominator).mean()


def compute_pnet_loss(
    outputs: Dict[str, object],
    mask: torch.Tensor,
    edge: torch.Tensor,
    is_pseudo: torch.Tensor,
    confidence: torch.Tensor,
    mask_level_weights: Sequence[float] = (0.0625, 0.125, 0.25, 0.5, 1.0),
    labeled_weight: float = 1.0,
    pseudo_weight: float = 1.0,
    edge_weight: float = 1.0,
    pseudo_edge_weight: float = 0.0,
    ual_weight: float = 0.0,
    progress: float = 1.0,
) -> Dict[str, torch.Tensor]:
    predictions = outputs["mask_logits"]
    if len(predictions) != len(mask_level_weights):
        raise ValueError("mask_level_weights must match the five mask predictions")

    is_pseudo = is_pseudo.bool().view(-1)
    labeled_idx = ~is_pseudo
    pseudo_idx = is_pseudo
    zero = predictions[-1].sum() * 0.0

    labeled_loss = zero
    if bool(labeled_idx.any()):
        labeled_loss = sum(
            float(weight) * structure_loss(pred[labeled_idx], mask[labeled_idx])
            for weight, pred in zip(mask_level_weights, predictions)
        )

    pseudo_loss = zero
    if bool(pseudo_idx.any()):
        pseudo_loss = sum(
            float(weight)
            * soft_pseudo_loss(
                pred[pseudo_idx], mask[pseudo_idx], confidence[pseudo_idx]
            )
            for weight, pred in zip(mask_level_weights, predictions)
        )

    present_weight = 0.0
    mask_loss = zero
    if bool(labeled_idx.any()) and labeled_weight > 0:
        mask_loss = mask_loss + float(labeled_weight) * labeled_loss
        present_weight += float(labeled_weight)
    if bool(pseudo_idx.any()) and pseudo_weight > 0:
        mask_loss = mask_loss + float(pseudo_weight) * pseudo_loss
        present_weight += float(pseudo_weight)
    if present_weight <= 0:
        raise ValueError("At least one active source loss weight is required")
    mask_loss = mask_loss / present_weight

    edge_loss = zero
    edge_logits = outputs.get("edge_logits")
    if edge_logits is not None and edge_weight > 0:
        edge_numerator = zero
        edge_denominator = 0.0
        if bool(labeled_idx.any()):
            edge_numerator = edge_numerator + edge_dice_loss(
                edge_logits[labeled_idx], edge[labeled_idx]
            )
            edge_denominator += 1.0
        if bool(pseudo_idx.any()) and pseudo_edge_weight > 0:
            edge_numerator = edge_numerator + float(pseudo_edge_weight) * edge_dice_loss(
                edge_logits[pseudo_idx],
                edge[pseudo_idx],
                confidence[pseudo_idx],
            )
            edge_denominator += float(pseudo_edge_weight)
        if edge_denominator > 0:
            edge_loss = edge_numerator / edge_denominator

    ual_loss = zero
    if ual_weight > 0:
        probability = torch.sigmoid(predictions[-1])
        uncertainty = 1.0 - (2.0 * probability - 1.0).square()
        reliability = torch.ones_like(confidence)
        if bool(pseudo_idx.any()):
            reliability[pseudo_idx] = confidence[pseudo_idx]
        ual_loss = (uncertainty * reliability).sum() / reliability.sum().clamp_min(1.0)
        progress = min(max(float(progress), 0.0), 1.0)
        ual_loss = ual_loss * (0.5 - 0.5 * math.cos(math.pi * progress))

    total = mask_loss + float(edge_weight) * edge_loss + float(ual_weight) * ual_loss
    return {
        "total": total,
        "mask": mask_loss.detach(),
        "labeled": labeled_loss.detach(),
        "pseudo": pseudo_loss.detach(),
        "edge": edge_loss.detach(),
        "ual": ual_loss.detach(),
    }
