"""Box-aware ANet used only for pseudo-label generation.

The public interface is deliberately small:

    model = NoisyCODANet(...)
    outputs = model(image, box_mask)

``image`` is RGB in [0, 1]. ``box_mask`` is a filled binary mask.  The model
constructs ``image * box_mask`` internally, so callers cannot accidentally
feed a box at PNet inference time.
"""

from __future__ import annotations

import math
from typing import Dict, Sequence, Tuple

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
        kernel_size,
        stride: int = 1,
        padding=0,
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


class ETM(nn.Module):
    """Efficient multi-receptive-field transition module from Noisy-COD."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.branch0 = ConvBNAct(in_channels, out_channels, 3, padding=1)
        self.branch1 = nn.Sequential(
            ConvBNAct(out_channels, out_channels, 1),
            ConvBNAct(out_channels, out_channels, (1, 3), padding=(0, 1)),
            ConvBNAct(out_channels, out_channels, (3, 1), padding=(1, 0)),
            ConvBNAct(out_channels, out_channels, (1, 5), padding=(0, 2)),
            ConvBNAct(out_channels, out_channels, (5, 1), padding=(2, 0)),
        )
        self.branch2 = nn.Sequential(
            ConvBNAct(out_channels, out_channels, 1),
            ConvBNAct(out_channels, out_channels, (1, 5), padding=(0, 2)),
            ConvBNAct(out_channels, out_channels, (5, 1), padding=(2, 0)),
            ConvBNAct(out_channels, out_channels, (1, 7), padding=(0, 3)),
            ConvBNAct(out_channels, out_channels, (7, 1), padding=(3, 0)),
        )
        self.branch3 = nn.Sequential(
            ConvBNAct(out_channels, out_channels, 1),
            ConvBNAct(out_channels, out_channels, (1, 7), padding=(0, 3)),
            ConvBNAct(out_channels, out_channels, (7, 1), padding=(3, 0)),
            ConvBNAct(out_channels, out_channels, (1, 9), padding=(0, 4)),
            ConvBNAct(out_channels, out_channels, (9, 1), padding=(4, 0)),
        )
        self.fuse = ConvBNAct(out_channels * 4, out_channels, 1)
        self.shortcut = ConvBNAct(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x0 = self.branch0(x)
        x1 = self.branch1(x0)
        x2 = self.branch2(x1)
        x3 = self.branch3(x2)
        return F.relu(
            self.fuse(torch.cat((x0, x1, x2, x3), dim=1))
            + self.shortcut(x),
            inplace=True,
        )


class HaarDWT(nn.Module):
    """Parameter-free 2-D Haar transform returning LL/LH/HL/HH bands."""

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pad_h = x.shape[-2] % 2
        pad_w = x.shape[-1] % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode="replicate")

        x1 = x[:, :, 0::2, 0::2] * 0.5
        x2 = x[:, :, 1::2, 0::2] * 0.5
        x3 = x[:, :, 0::2, 1::2] * 0.5
        x4 = x[:, :, 1::2, 1::2] * 0.5
        ll = x1 + x2 + x3 + x4
        lh = -x1 + x2 - x3 + x4
        hl = -x1 - x2 + x3 + x4
        hh = x1 - x2 - x3 + x4
        return ll, lh, hl, hh


class GlobalContextDWT(nn.Module):
    def __init__(self, in_channels: Sequence[int], channels: int) -> None:
        super().__init__()
        self.transitions = nn.ModuleList(
            [ETM(in_c, channels) for in_c in in_channels]
        )
        self.fuse = nn.Conv2d(channels * 4, channels, 3, padding=1)
        self.dwt = HaarDWT()

    def forward(self, features: Sequence[torch.Tensor]):
        projected = [layer(feat) for layer, feat in zip(self.transitions, features)]
        target_size = projected[0].shape[-2:]
        aligned = [
            feat
            if feat.shape[-2:] == target_size
            else F.interpolate(feat, target_size, mode="bilinear", align_corners=False)
            for feat in projected
        ]
        bands = self.dwt(self.fuse(torch.cat(aligned, dim=1)))
        return projected, bands


class FrequencyBranch(nn.Module):
    """One ConvNeXt branch with the released HH-shallow/LL-deep prior."""

    def __init__(
        self,
        encoder: nn.Module,
        encoder_channels: Sequence[int],
        channels: int,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.context = GlobalContextDWT(encoder_channels, channels)
        self.inject = nn.ModuleList([ETM(channels * 2, channels) for _ in range(4)])

    def forward(self, x: torch.Tensor):
        raw_features = list(self.encoder(x))
        features, (ll, _lh, _hl, hh) = self.context(raw_features)
        routed = []
        for stage, feature in enumerate(features):
            band = hh if stage < 2 else ll
            band = F.interpolate(
                band,
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            routed.append(self.inject[stage](torch.cat((feature, band), dim=1)))
        return routed, raw_features[-1]


class GlobalPrior(nn.Module):
    def __init__(self, in_channels: int, depth: int = 32) -> None:
        super().__init__()
        self.global_proj = nn.Conv2d(in_channels, depth, 1, bias=False)
        self.branches = nn.ModuleList(
            [
                ConvBNAct(in_channels, depth, 1),
                ConvBNAct(in_channels, depth, 3, padding=6, dilation=6),
                ConvBNAct(in_channels, depth, 3, padding=12, dilation=12),
                ConvBNAct(in_channels, depth, 3, padding=18, dilation=18),
            ]
        )
        self.fuse = ConvBNAct(depth * 5, depth, 1)
        self.out = nn.Sequential(
            nn.Conv2d(depth, depth, 3, padding=1, bias=False),
            nn.BatchNorm2d(depth),
            nn.PReLU(),
            nn.Conv2d(depth, 1, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        size = x.shape[-2:]
        pooled = F.adaptive_avg_pool2d(x, 1)
        pooled = F.interpolate(
            self.global_proj(pooled), size=size, mode="bilinear", align_corners=False
        )
        return self.out(self.fuse(torch.cat([pooled] + [b(x) for b in self.branches], dim=1)))


class RefineBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.prior_fuse = nn.Conv2d(channels * 2, channels, 1)
        self.object_detail = nn.Sequential(
            ConvBNAct(channels, channels, 3, padding=1),
            ConvBNAct(channels, channels, 3, padding=1),
        )
        self.mask_head = nn.Sequential(
            ConvBNAct(channels * 2, channels, 3, padding=1),
            ConvBNAct(channels, channels // 2, 3, padding=1),
            nn.Conv2d(channels // 2, 1, 3, padding=1),
        )
        self.edge_head = nn.Sequential(
            ConvBNAct(channels, channels // 2, 3, padding=1),
            nn.Conv2d(channels // 2, 1, 3, padding=1),
        )

    def forward(self, feature: torch.Tensor, prior: torch.Tensor):
        prior = F.interpolate(
            prior, feature.shape[-2:], mode="bilinear", align_corners=False
        )
        expanded = prior.expand(-1, feature.shape[1], -1, -1)
        detail = self.object_detail(self.prior_fuse(torch.cat((feature, expanded), dim=1)))
        reverse = 1.0 - torch.sigmoid(prior)
        residual = self.mask_head(torch.cat((reverse * feature, detail), dim=1))
        return prior + residual, self.edge_head(detail)


class RefineDecoder(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([RefineBlock(channels) for _ in range(4)])

    def forward(
        self,
        features: Sequence[torch.Tensor],
        prior: torch.Tensor,
        output_size: Tuple[int, int],
    ):
        mask_logits = []
        edge_logits = []
        current = prior
        for stage in range(3, -1, -1):
            current, edge = self.blocks[stage](features[stage], current)
            mask_logits.append(
                F.interpolate(current, output_size, mode="bilinear", align_corners=False)
            )
            edge_logits.append(
                F.interpolate(edge, output_size, mode="bilinear", align_corners=False)
            )
        return tuple(mask_logits), tuple(edge_logits)


class NoisyCODANet(nn.Module):
    """Two-branch Box-aware teacher. It is never used as the final RGB model."""

    def __init__(
        self,
        backbone_name: str = "convnext_base.fb_in22k_ft_in1k_384",
        pretrained: bool = True,
        channels: int = 64,
        gradient_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        rgb_encoder = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        box_encoder = timm.create_model(
            backbone_name,
            pretrained=False,
            features_only=True,
            out_indices=(0, 1, 2, 3),
        )
        # Both branches start from exactly the same ImageNet representation.
        box_encoder.load_state_dict(rgb_encoder.state_dict(), strict=True)

        if gradient_checkpointing:
            for encoder in (rgb_encoder, box_encoder):
                if hasattr(encoder, "set_grad_checkpointing"):
                    encoder.set_grad_checkpointing(enable=True)

        encoder_channels = list(rgb_encoder.feature_info.channels())
        if len(encoder_channels) != 4:
            raise RuntimeError(f"Expected four ConvNeXt stages, got {encoder_channels}")

        self.normalizer = PixelNormalizer()
        self.rgb_branch = FrequencyBranch(rgb_encoder, encoder_channels, channels)
        self.box_branch = FrequencyBranch(box_encoder, encoder_channels, channels)
        self.prior = GlobalPrior(encoder_channels[-1] * 2)
        self.decoder = RefineDecoder(channels * 2)

    def forward(self, image: torch.Tensor, box_mask: torch.Tensor) -> Dict[str, tuple]:
        if image.ndim != 4 or image.shape[1] != 3:
            raise ValueError(f"image must have shape [B,3,H,W], got {tuple(image.shape)}")
        if box_mask.ndim == 3:
            box_mask = box_mask.unsqueeze(1)
        if box_mask.shape[:2] != (image.shape[0], 1):
            raise ValueError(
                f"box_mask must have shape [B,1,H,W], got {tuple(box_mask.shape)}"
            )
        if box_mask.shape[-2:] != image.shape[-2:]:
            box_mask = F.interpolate(box_mask.float(), image.shape[-2:], mode="nearest")

        box_mask = box_mask.to(dtype=image.dtype).clamp(0.0, 1.0)
        rgb = self.normalizer(image)
        box_rgb = self.normalizer(image * box_mask)
        rgb_features, rgb_deep = self.rgb_branch(rgb)
        box_features, box_deep = self.box_branch(box_rgb)
        fused = [torch.cat((a, b), dim=1) for a, b in zip(rgb_features, box_features)]

        prior = self.prior(torch.cat((rgb_deep, box_deep), dim=1))
        coarse = F.interpolate(
            prior, image.shape[-2:], mode="bilinear", align_corners=False
        )
        refined, edges = self.decoder(fused, prior, image.shape[-2:])
        return {
            "mask_logits": (coarse,) + refined,
            "edge_logits": edges,
        }

    @torch.no_grad()
    def predict(self, image: torch.Tensor, box_mask: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self(image, box_mask)["mask_logits"][-1])


def structure_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    weight = 1.0 + 5.0 * torch.abs(
        F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target
    )
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (weight * bce).sum((2, 3)) / weight.sum((2, 3)).clamp_min(1e-6)
    prob = torch.sigmoid(logits)
    inter = (prob * target * weight).sum((2, 3))
    union = ((prob + target) * weight).sum((2, 3))
    wiou = 1.0 - (inter + 1.0) / (union - inter + 1.0)
    return (bce + wiou).mean()


def edge_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits).flatten(1)
    target = target.flatten(1)
    numerator = 2.0 * (prob * target).sum(1) + 1.0
    denominator = prob.square().sum(1) + target.square().sum(1) + 1.0
    return (1.0 - numerator / denominator).mean()


def uncertainty_loss(logits: torch.Tensor) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    return (1.0 - (2.0 * prob - 1.0).square()).mean()


def cosine_ramp(progress: float) -> float:
    progress = min(max(float(progress), 0.0), 1.0)
    return 0.5 * (1.0 - math.cos(math.pi * progress))


def compute_anet_loss(
    outputs: Dict[str, tuple],
    mask: torch.Tensor,
    edge: torch.Tensor,
    progress: float,
    edge_weight: float = 4.0,
    ual_weight: float = 2.0,
) -> Dict[str, torch.Tensor]:
    masks = outputs["mask_logits"]
    edges = outputs["edge_logits"]
    mask_weights = (0.0625, 0.125, 0.25, 0.5, 1.0)
    mask_loss = sum(w * structure_loss(pred, mask) for w, pred in zip(mask_weights, masks))
    # Match the original convention: supervise the last three refined edges.
    edge_loss = sum(
        w * edge_dice_loss(pred, edge)
        for w, pred in zip((0.125, 0.25, 0.5), edges[1:])
    )
    ual_coef = cosine_ramp(progress)
    ual_loss = uncertainty_loss(masks[-1]) * ual_coef
    total = mask_loss + float(edge_weight) * edge_loss + float(ual_weight) * ual_loss
    return {
        "total": total,
        "mask": mask_loss.detach(),
        "edge": edge_loss.detach(),
        "ual": ual_loss.detach(),
        "ual_coef": torch.as_tensor(ual_coef, device=mask.device),
    }
