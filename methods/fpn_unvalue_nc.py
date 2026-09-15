# -*- coding: utf-8 -*-
"""Plain PVTv2-B4 FPN with mask-anchored unvalue curriculum loss.

Architecture and inference are exactly the repository's
``PvtV2B4_FPN_Baseline``.  Boxes and EMA predictions are training-only
supervision.  In particular, an unvalue pseudo mask is never trusted over the
whole image: box-outside background is supervised directly, while the mask
inside a box is used only where an EMA teacher is confident and HFlip-stable.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fpn_baseline import PvtV2B4_FPN_Baseline


def _cosine_value(progress, start, end, low, high):
    progress = float(min(max(progress, 0.0), 1.0))
    if progress <= start:
        return float(low)
    if progress >= end:
        return float(high)
    local = (progress - start) / max(end - start, 1e-8)
    coefficient = 0.5 * (1.0 - math.cos(math.pi * local))
    return float(low + (high - low) * coefficient)


def _resize(value, size, mode="nearest"):
    if value.shape[-2:] == size:
        return value
    if mode == "nearest":
        return F.interpolate(value.float(), size=size, mode=mode)
    return F.interpolate(
        value.float(), size=size, mode=mode, align_corners=False
    )


def _dilate(mask, kernel_size):
    if kernel_size <= 1:
        return mask
    if kernel_size % 2 == 0:
        kernel_size += 1
    return F.max_pool2d(
        mask,
        kernel_size=kernel_size,
        stride=1,
        padding=kernel_size // 2,
    )


def _sample_mean(value, pixel_weight, eps=1e-6):
    numerator = (value * pixel_weight).flatten(1).sum(dim=1)
    denominator = pixel_weight.flatten(1).sum(dim=1).clamp_min(eps)
    return numerator / denominator


def _selected_mean(value, selector):
    selector = selector.to(device=value.device, dtype=value.dtype)
    return (value * selector).sum() / selector.sum().clamp_min(1.0)


class MaskAnchoredUnvalueLoss(nn.Module):
    """NC on mask pools; box/teacher loss on unvalue samples."""

    def __init__(
        self,
        mask_loss_mode="bce",
        q_switch_ratio=0.40,
        boundary_kernel=31,
        boundary_gain=5.0,
        box_dilate_kernel=9,
        teacher_confidence=0.90,
        teacher_disagreement=0.05,
        min_teacher_foreground=16,
        outside_weight=0.20,
        dynamic_weight=0.25,
        consistency_weight=0.05,
        dynamic_mix_max=0.50,
        eps=1e-6,
    ):
        super().__init__()
        self.mask_loss_mode = str(mask_loss_mode).lower()
        if self.mask_loss_mode not in {"bce", "nc"}:
            raise ValueError("mask_loss_mode must be 'bce' or 'nc'.")
        self.q_switch_ratio = float(q_switch_ratio)
        self.boundary_kernel = int(boundary_kernel)
        self.boundary_gain = float(boundary_gain)
        self.box_dilate_kernel = int(box_dilate_kernel)
        self.teacher_confidence = float(teacher_confidence)
        self.teacher_disagreement = float(teacher_disagreement)
        self.min_teacher_foreground = int(min_teacher_foreground)
        self.outside_weight = float(outside_weight)
        self.dynamic_weight = float(dynamic_weight)
        self.consistency_weight = float(consistency_weight)
        self.dynamic_mix_max = float(dynamic_mix_max)
        self.eps = float(eps)

    def _mask_pool_loss(self, logits, target, mask_selector, progress):
        padding = self.boundary_kernel // 2
        local_mean = F.avg_pool2d(
            target,
            kernel_size=self.boundary_kernel,
            stride=1,
            padding=padding,
        )
        boundary_weight = (
            1.0
            + self.boundary_gain * torch.abs(local_mean - target)
        )
        bce_map = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        wbce_each = _sample_mean(
            bce_map, boundary_weight, eps=self.eps
        )

        if self.mask_loss_mode == "bce":
            bce_each = bce_map.flatten(1).mean(1)
            zero_each = torch.zeros_like(bce_each)
            return (
                _selected_mean(bce_each, mask_selector),
                _selected_mean(bce_each, mask_selector),
                _selected_mean(zero_each, mask_selector),
                0.0,
            )

        q = 2.0 if float(progress) <= self.q_switch_ratio else 1.0
        probability = logits.sigmoid()
        numerator = torch.abs(probability - target).pow(q).flatten(1).sum(1)
        union = (
            probability + target - probability * target
        ).flatten(1).sum(1).clamp_min(self.eps)
        nc_each = numerator / union

        if q == 2.0:
            total_each = wbce_each + nc_each
        else:
            total_each = 2.0 * nc_each

        return (
            _selected_mean(total_each, mask_selector),
            _selected_mean(wbce_each, mask_selector),
            _selected_mean(nc_each, mask_selector),
            q,
        )

    def _unvalue_loss(
        self,
        logits,
        original_target,
        box_mask,
        teacher_prob,
        teacher_disagreement,
        teacher_valid,
        unvalue_selector,
        progress,
    ):
        zero = logits.sum() * 0.0
        if box_mask is None:
            return zero, zero, zero, zero, zero

        box = (box_mask >= 0.5).to(logits.dtype)
        outside = 1.0 - _dilate(box, self.box_dilate_kernel)
        outside_bce = F.softplus(logits)  # BCEWithLogits(logits, 0)
        outside_each = _sample_mean(outside_bce, outside, self.eps)
        outside_loss = _selected_mean(outside_each, unvalue_selector)

        if teacher_prob is None:
            return outside_loss, zero, zero, zero, zero

        teacher = teacher_prob.detach().clamp(0.0, 1.0)
        if teacher_disagreement is None:
            stable = torch.ones_like(teacher)
        else:
            stable = (
                teacher_disagreement.detach()
                <= self.teacher_disagreement
            ).to(logits.dtype)

        confident = (
            (teacher >= self.teacher_confidence)
            | (teacher <= 1.0 - self.teacher_confidence)
        ).to(logits.dtype)
        reliable = box * stable * confident

        foreground = (
            box * stable * (teacher >= self.teacher_confidence)
        )
        foreground_count = foreground.flatten(1).sum(1)
        has_foreground = (
            foreground_count >= self.min_teacher_foreground
        ).to(logits.dtype)
        if teacher_valid is not None:
            has_foreground = has_foreground * teacher_valid.to(logits.dtype)
        valid_unvalue = unvalue_selector.to(logits.dtype) * has_foreground

        mix = _cosine_value(
            progress,
            start=40.0 / 150.0,
            end=90.0 / 150.0,
            low=0.20,
            high=self.dynamic_mix_max,
        )
        dynamic_target = (
            (1.0 - mix) * original_target + mix * teacher
        ).detach()

        probability = logits.sigmoid()
        numerator = (
            reliable * torch.abs(probability - dynamic_target)
        ).flatten(1).sum(1)
        union = (
            reliable
            * (
                probability
                + dynamic_target
                - probability * dynamic_target
            )
        ).flatten(1).sum(1).clamp_min(self.eps)
        dynamic_each = numerator / union
        dynamic_loss = _selected_mean(dynamic_each, valid_unvalue)

        consistency_map = (probability - teacher).square()
        consistency_each = _sample_mean(
            consistency_map, reliable, self.eps
        )
        consistency_loss = _selected_mean(
            consistency_each, valid_unvalue
        )

        reliable_ratio_each = reliable.flatten(1).mean(1)
        reliable_ratio = _selected_mean(
            reliable_ratio_each, unvalue_selector
        )
        accepted_ratio = _selected_mean(
            has_foreground, unvalue_selector
        )
        return (
            outside_loss,
            dynamic_loss,
            consistency_loss,
            reliable_ratio,
            accepted_ratio,
        )

    def forward(self, logits, target, data, iter_percentage):
        progress = float(iter_percentage)
        batch_size = logits.shape[0]
        device, dtype = logits.device, logits.dtype

        pool_id = data.get("pool_id")
        if pool_id is None:
            pool_id = torch.zeros(batch_size, device=device, dtype=torch.long)
        pool_id = pool_id.to(device=device).reshape(batch_size, -1)[:, 0]
        mask_selector = pool_id < 2
        unvalue_selector = pool_id == 2

        mask_loss, wbce, nc, q = self._mask_pool_loss(
            logits, target, mask_selector, progress
        )

        box_mask = data.get("box_mask")
        if box_mask is not None:
            box_mask = _resize(
                box_mask.to(device=device, dtype=dtype),
                logits.shape[-2:],
            ).clamp(0.0, 1.0)

        teacher_prob = data.get("teacher_prob")
        if teacher_prob is not None:
            teacher_prob = _resize(
                teacher_prob.to(device=device, dtype=dtype),
                logits.shape[-2:],
                mode="bilinear",
            )
        disagreement = data.get("teacher_disagreement")
        if disagreement is not None:
            disagreement = _resize(
                disagreement.to(device=device, dtype=dtype),
                logits.shape[-2:],
                mode="bilinear",
            )
        teacher_valid = data.get("teacher_valid")
        if teacher_valid is not None:
            teacher_valid = teacher_valid.to(device=device, dtype=dtype).reshape(-1)

        (
            outside,
            dynamic,
            consistency,
            reliable_ratio,
            accepted_ratio,
        ) = self._unvalue_loss(
            logits=logits,
            original_target=target,
            box_mask=box_mask,
            teacher_prob=teacher_prob,
            teacher_disagreement=disagreement,
            teacher_valid=teacher_valid,
            unvalue_selector=unvalue_selector,
            progress=progress,
        )

        box_gate = _cosine_value(
            progress, 40.0 / 150.0, 60.0 / 150.0, 0.0, self.outside_weight
        )
        dynamic_gate = _cosine_value(
            progress, 60.0 / 150.0, 90.0 / 150.0, 0.0, self.dynamic_weight
        )
        consistency_gate = _cosine_value(
            progress, 60.0 / 150.0, 90.0 / 150.0, 0.0, self.consistency_weight
        )

        total = (
            mask_loss
            + box_gate * outside
            + dynamic_gate * dynamic
            + consistency_gate * consistency
        )
        return total, {
            "bce": wbce.detach(),
            "wbce": wbce.detach(),
            "nc": nc.detach(),
            "q": logits.new_tensor(q),
            "unvalue_out": outside.detach(),
            "unvalue_out_weight": logits.new_tensor(box_gate),
            "unvalue_dynamic": dynamic.detach(),
            "unvalue_dynamic_weight": logits.new_tensor(dynamic_gate),
            "unvalue_consistency": consistency.detach(),
            "unvalue_reliable_pixel": reliable_ratio.detach(),
            "unvalue_accepted": accepted_ratio.detach(),
            "unvalue_batch_count": unvalue_selector.sum().detach().to(dtype),
        }


class PvtV2B4_FPN_Unvalue(PvtV2B4_FPN_Baseline):
    """Exact FPN baseline plus box/teacher use of the unvalue pool."""

    def __init__(self, pretrained=True, input_norm=True, fpn_dim=64,
                 use_checkpoint=False, **kwargs):
        loss_keys = {
            "mask_loss_mode",
            "q_switch_ratio", "boundary_kernel", "boundary_gain",
            "box_dilate_kernel", "teacher_confidence",
            "teacher_disagreement", "min_teacher_foreground",
            "outside_weight", "dynamic_weight", "consistency_weight",
            "dynamic_mix_max",
        }
        loss_kwargs = {k: kwargs.pop(k) for k in list(kwargs) if k in loss_keys}
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )
        self.unvalue_loss = MaskAnchoredUnvalueLoss(**loss_kwargs)

    def forward(self, data, iter_percentage=1.0, **kwargs):
        del kwargs
        logits = self.body(data=data)
        if not self.training:
            return logits

        target = data["mask"].to(device=logits.device, dtype=logits.dtype)
        target = _resize(target, logits.shape[-2:]).clamp(0.0, 1.0)
        total, items = self.unvalue_loss(
            logits=logits,
            target=target,
            data=data,
            iter_percentage=iter_percentage,
        )
        items["total"] = total.detach()
        return {
            "logits": logits,
            "vis": {"sal": logits.sigmoid()},
            "loss": total,
            "loss_items": items,
            "loss_str": (
                f"L:{total.detach().item():.4f} "
                f"NC:{items['nc'].item():.4f} Q:{items['q'].item():.1f} "
                f"OUT:{items['unvalue_out'].item():.4f} "
                f"DYN:{items['unvalue_dynamic'].item():.4f} "
                f"ACC:{items['unvalue_accepted'].item():.2f}"
            ),
        }
