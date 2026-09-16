# -*- coding: utf-8 -*-
"""
PVTv2-B4 + FPN frequency-selection / frequency-guided-sampling ablations.

Drop this file into:
    APBOXNet/methods/fpn_freqguided.py

It is designed for the existing:
    methods/fpn_baseline.py::PvtV2B4_FPN_Baseline

Ablations:
    A0: original FPN baseline
    A1: Frequency Selection only
    A2: Frequency-guided Sampling only
    A3: Frequency Selection + Frequency-guided Sampling

Design principles:
1) Keep encoder / FPN width / smooth convs / predictor / loss unchanged.
2) Frequency Selection is neutral-initialized:
       gate = 2 * sigmoid(logit), last layer zero-init => gate == 1 at start.
3) Frequency-guided Sampling keeps bilinear upsampling as a residual anchor:
       out = (1-mix) * bilinear + mix * guided_sample
4) Frequency controls sampling radius; local cross-scale similarity controls direction.

Notes:
- FFT is executed in fp32 for AMP stability.
- The four octave-like bands are:
    [0,1/16), [1/16,1/8), [1/8,1/4), [1/4,1/2].
- C5/P5 is left untouched. A1/A2/A3 modify only the three top-down fusions:
    P5->P4, P4->P3, P3->P2.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .fpn_baseline import PvtV2B4_FPN_Baseline


class FourierBandDecomposer(nn.Module):
    """Differentiable 2-D Fourier band decomposition."""

    def __init__(self, thresholds=(0.0, 1/16, 1/8, 1/4, 1/2)):
        super().__init__()
        if len(thresholds) < 3:
            raise ValueError("Need at least two frequency bands.")
        self.thresholds = tuple(float(v) for v in thresholds)

    def _masks(self, h, w, device):
        fy = torch.fft.fftfreq(h, device=device, dtype=torch.float32).abs()
        fx = torch.fft.fftfreq(w, device=device, dtype=torch.float32).abs()
        radius = torch.maximum(fy[:, None], fx[None, :])

        masks = []
        for i in range(len(self.thresholds) - 1):
            lo = self.thresholds[i]
            hi = self.thresholds[i + 1]
            if i == len(self.thresholds) - 2:
                mask = (radius >= lo) & (radius <= hi + 1e-7)
            else:
                mask = (radius >= lo) & (radius < hi)
            masks.append(mask.to(torch.float32)[None, None])
        return masks

    def forward(self, x):
        input_dtype = x.dtype
        x32 = x.float()
        spectrum = torch.fft.fft2(x32, dim=(-2, -1), norm="ortho")
        masks = self._masks(x.shape[-2], x.shape[-1], x.device)

        bands = []
        for mask in masks:
            band_spec = spectrum * mask
            band = torch.fft.ifft2(
                band_spec, dim=(-2, -1), norm="ortho"
            ).real
            bands.append(band.to(input_dtype))
        return bands


def _band_energy(x):
    return x.float().abs().mean(dim=1, keepdim=True)


def _high_frequency_ratio(bands, eps=1e-6):
    energies = [_band_energy(b) for b in bands]
    total = torch.stack(energies, dim=0).sum(dim=0)
    high = torch.stack(energies[-2:], dim=0).sum(dim=0)
    return (high / (total + eps)).clamp_(0.0, 1.0)


class SpatialFrequencySelector(nn.Module):
    """FreqSelect / FBM-style spatially variant feature-band selection."""

    def __init__(
        self,
        channels=64,
        hidden=16,
        thresholds=(0.0, 1/16, 1/8, 1/4, 1/2),
    ):
        super().__init__()
        self.channels = int(channels)
        self.decomposer = FourierBandDecomposer(thresholds)
        self.num_bands = len(thresholds) - 1
        if self.num_bands != 4:
            raise ValueError("This implementation expects four bands.")

        self.gate_net = nn.Sequential(
            nn.Conv2d(self.num_bands, hidden, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, self.num_bands - 1, 1, 1, 0, bias=True),
        )
        nn.init.zeros_(self.gate_net[-1].weight)
        nn.init.zeros_(self.gate_net[-1].bias)

    def forward(self, x):
        bands = self.decomposer(x)
        desc = torch.cat(
            [_band_energy(b) for b in bands], dim=1
        ).to(dtype=x.dtype)

        gate_logits = self.gate_net(desc)
        gates = 2.0 * torch.sigmoid(gate_logits)  # init = 1

        selected = bands[0]
        selected_bands = [bands[0]]
        for idx in range(1, self.num_bands):
            g = gates[:, idx - 1:idx]
            b = bands[idx] * g
            selected = selected + b
            selected_bands.append(b)

        freq_score = _high_frequency_ratio(selected_bands)
        return selected.to(dtype=x.dtype), freq_score.to(dtype=x.dtype)


class FrequencyScore(nn.Module):
    """A2 helper: measure frequency without altering the lateral feature."""

    def __init__(self, thresholds=(0.0, 1/16, 1/8, 1/4, 1/2)):
        super().__init__()
        self.decomposer = FourierBandDecomposer(thresholds)

    def forward(self, x):
        bands = self.decomposer(x)
        return _high_frequency_ratio(bands).to(dtype=x.dtype)


class FrequencyGuidedSampler(nn.Module):
    """
    Frequency decides sampling radius; cross-scale similarity decides direction.

    High-frequency location -> small sampling radius.
    Low-frequency location  -> large sampling radius.
    """

    def __init__(
        self,
        channels=64,
        key_dim=16,
        r_min=0.0,
        r_max=2.0,
        temperature=0.10,
        residual_init=0.10,
        detach_frequency=True,
    ):
        super().__init__()
        if not (0.0 < residual_init < 1.0):
            raise ValueError("residual_init must be in (0,1).")
        if temperature <= 0:
            raise ValueError("temperature must be > 0.")
        if r_max < r_min:
            raise ValueError("r_max must be >= r_min.")

        self.channels = int(channels)
        self.key_dim = int(key_dim)
        self.r_min = float(r_min)
        self.r_max = float(r_max)
        self.temperature = float(temperature)
        self.detach_frequency = bool(detach_frequency)

        self.q_proj = nn.Conv2d(channels, key_dim, 1, bias=False)
        self.k_proj = nn.Conv2d(channels, key_dim, 1, bias=False)

        dirs = torch.tensor(
            [
                [0.0, 0.0],
                [-1.0, 0.0], [1.0, 0.0],
                [0.0, -1.0], [0.0, 1.0],
                [-1.0, -1.0], [1.0, -1.0],
                [-1.0, 1.0], [1.0, 1.0],
            ],
            dtype=torch.float32,
        )
        self.register_buffer("directions", dirs, persistent=False)

        logit = math.log(residual_init / (1.0 - residual_init))
        self.mix_logit = nn.Parameter(torch.tensor(logit, dtype=torch.float32))

    @staticmethod
    def _base_grid(b, h, w, device, dtype):
        ys = (
            (torch.arange(h, device=device, dtype=dtype) + 0.5)
            * (2.0 / h) - 1.0
        )
        xs = (
            (torch.arange(w, device=device, dtype=dtype) + 0.5)
            * (2.0 / w) - 1.0
        )
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        grid = torch.stack((xx, yy), dim=-1)
        return grid.unsqueeze(0).expand(b, -1, -1, -1)

    def forward(self, top_down, lateral, freq_score):
        b, c, h, w = lateral.shape
        if c != self.channels:
            raise RuntimeError(f"Expected {self.channels} channels, got {c}.")

        up = F.interpolate(
            top_down,
            size=(h, w),
            mode="bilinear",
            align_corners=False,
        )

        score = freq_score.float().clamp(0.0, 1.0)
        if self.detach_frequency:
            score = score.detach()

        radius = self.r_min + (self.r_max - self.r_min) * (1.0 - score)

        q = F.normalize(self.q_proj(lateral).float(), dim=1, eps=1e-6)
        k_map = self.k_proj(up).float()
        sample_src = torch.cat([up.float(), k_map], dim=1)

        base_grid = self._base_grid(
            b, h, w, lateral.device, torch.float32
        )

        candidates = []
        similarities = []
        for dxy in self.directions:
            dx, dy = dxy[0], dxy[1]
            offset_x = dx * radius[:, 0] * (2.0 / w)
            offset_y = dy * radius[:, 0] * (2.0 / h)

            grid = base_grid.clone()
            grid[..., 0] = grid[..., 0] + offset_x
            grid[..., 1] = grid[..., 1] + offset_y

            sampled = F.grid_sample(
                sample_src,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=False,
            )

            value = sampled[:, :c]
            key = F.normalize(sampled[:, c:], dim=1, eps=1e-6)
            sim = (q * key).sum(dim=1, keepdim=True)

            candidates.append(value)
            similarities.append(sim)

        sim_stack = torch.cat(similarities, dim=1)
        alpha = torch.softmax(sim_stack / self.temperature, dim=1)

        guided = torch.zeros_like(up.float())
        for idx, value in enumerate(candidates):
            guided = guided + value * alpha[:, idx:idx + 1]

        mix = torch.sigmoid(self.mix_logit)
        out = (1.0 - mix) * up.float() + mix * guided
        return out.to(dtype=up.dtype)


class PvtV2B4_FPN_FreqAblation(PvtV2B4_FPN_Baseline):
    """Shared implementation for A1/A2/A3."""

    def __init__(
        self,
        pretrained=True,
        input_norm=True,
        fpn_dim=64,
        use_checkpoint=False,
        use_freq_select=False,
        use_freq_sampling=False,
        freq_hidden=16,
        sampling_key_dim=16,
        sampling_r_min=0.0,
        sampling_r_max=2.0,
        sampling_temperature=0.10,
        sampling_residual_init=0.10,
        sampling_detach_frequency=True,
        **kwargs,
    ):
        super().__init__(
            pretrained=pretrained,
            input_norm=input_norm,
            fpn_dim=fpn_dim,
            use_checkpoint=use_checkpoint,
            **kwargs,
        )
        self.use_freq_select = bool(use_freq_select)
        self.use_freq_sampling = bool(use_freq_sampling)

        if self.use_freq_select:
            self.freq_select_2 = SpatialFrequencySelector(fpn_dim, freq_hidden)
            self.freq_select_3 = SpatialFrequencySelector(fpn_dim, freq_hidden)
            self.freq_select_4 = SpatialFrequencySelector(fpn_dim, freq_hidden)

        if self.use_freq_sampling:
            if not self.use_freq_select:
                self.freq_score_2 = FrequencyScore()
                self.freq_score_3 = FrequencyScore()
                self.freq_score_4 = FrequencyScore()

            sampler_kwargs = dict(
                channels=fpn_dim,
                key_dim=sampling_key_dim,
                r_min=sampling_r_min,
                r_max=sampling_r_max,
                temperature=sampling_temperature,
                residual_init=sampling_residual_init,
                detach_frequency=sampling_detach_frequency,
            )
            self.freq_sample_2 = FrequencyGuidedSampler(**sampler_kwargs)
            self.freq_sample_3 = FrequencyGuidedSampler(**sampler_kwargs)
            self.freq_sample_4 = FrequencyGuidedSampler(**sampler_kwargs)

    def _prepare_lateral(self, level, x):
        if self.use_freq_select:
            return getattr(self, f"freq_select_{level}")(x)

        if self.use_freq_sampling:
            score = getattr(self, f"freq_score_{level}")(x)
            return x, score

        return x, None

    def _topdown(self, level, top_down, lateral, freq_score):
        if not self.use_freq_sampling:
            return F.interpolate(
                top_down,
                size=lateral.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        return getattr(self, f"freq_sample_{level}")(
            top_down=top_down,
            lateral=lateral,
            freq_score=freq_score,
        )

    def body(self, data):
        image = data["image_m"]
        c2, c3, c4, c5 = self.normalize_encoder(image)

        # Keep P5 unchanged.
        l5 = self.lateral_5(c5)
        p5 = self.smooth_5(l5)

        l4_raw = self.lateral_4(c4)
        l4, f4 = self._prepare_lateral(4, l4_raw)
        u4 = self._topdown(4, p5, l4, f4)
        p4 = self.smooth_4(l4 + u4)

        l3_raw = self.lateral_3(c3)
        l3, f3 = self._prepare_lateral(3, l3_raw)
        u3 = self._topdown(3, p4, l3, f3)
        p3 = self.smooth_3(l3 + u3)

        l2_raw = self.lateral_2(c2)
        l2, f2 = self._prepare_lateral(2, l2_raw)
        u2 = self._topdown(2, p3, l2, f2)
        p2 = self.smooth_2(l2 + u2)

        logits = self.predictor(p2)
        return F.interpolate(
            logits,
            size=image.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )


class PvtV2B4_FPN_A0(PvtV2B4_FPN_Baseline):
    """A0: untouched baseline."""
    pass


class PvtV2B4_FPN_A1_FreqSelect(PvtV2B4_FPN_FreqAblation):
    """A1: Frequency Selection only."""
    def __init__(self, **kwargs):
        super().__init__(
            use_freq_select=True,
            use_freq_sampling=False,
            **kwargs,
        )


class PvtV2B4_FPN_A2_FreqSampling(PvtV2B4_FPN_FreqAblation):
    """A2: Frequency-guided Sampling only."""
    def __init__(self, **kwargs):
        super().__init__(
            use_freq_select=False,
            use_freq_sampling=True,
            **kwargs,
        )


class PvtV2B4_FPN_A3_FreqSelectSampling(PvtV2B4_FPN_FreqAblation):
    """A3: Frequency Selection + Frequency-guided Sampling."""
    def __init__(self, **kwargs):
        super().__init__(
            use_freq_select=True,
            use_freq_sampling=True,
            **kwargs,
        )
