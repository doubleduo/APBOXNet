# -*- coding: utf-8 -*-
"""
Train Noisy-COD ANet with the released 20% protocol and generate pseudo labels
for the remaining 80%.

Default:
    python anet20_main.py --config configs/anet20_noisycod.py --mode all

Train only:
    python anet20_main.py --config configs/anet20_noisycod.py --mode train

Generate only:
    python anet20_main.py --config configs/anet20_noisycod.py \
        --mode generate \
        --checkpoint ANet_outputs/NoisyCOD_ANet_F20/checkpoints/Net_epoch_best.pth

Notes
-----
1. The released split_data.py calls the setting "20%" but actually chooses
   exactly 800 samples from the 4040-image pool using NumPy seed 2024.
2. The remaining 3240 samples are never used in the ANet training loss.
3. Their GT can be used for validation/model selection, matching the released
   TrainDDP.py protocol, but generation still feeds only RGB + Box to ANet.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

try:
    import albumentations as A
except Exception:
    A = None

from methods.anet_noisycod_A3_model import (
    NoisyCODANet,
    cal_ual,
    dice_loss,
    get_ual_coef,
    structure_loss,
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def load_cfg(path: str) -> dict:
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("anet20_cfg", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cfg


def seed_everything(seed=2024):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def setup_logger(log_file: Path):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ANet20")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(message)s")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def list_stems(folder: Path, suffix: str) -> List[str]:
    return sorted(
        p.name[: -len(suffix)]
        for p in folder.iterdir()
        if p.is_file() and p.name.endswith(suffix)
    )


def official_f20_split(
    image_dir: Path,
    image_suffix: str,
    expected_total: int,
    labeled_count: int,
    seed: int,
    strict_total: bool,
) -> Tuple[List[str], List[str]]:
    names = list_stems(image_dir, image_suffix)

    if strict_total and len(names) != expected_total:
        raise RuntimeError(
            f"Faithful F20 split expects exactly {expected_total} images, "
            f"but found {len(names)} in {image_dir}."
        )
    if labeled_count > len(names):
        raise ValueError(
            f"labeled_count={labeled_count} > dataset size={len(names)}"
        )

    # Reproduce released split_data.py:
    # np.random.seed(2024)
    # np.random.choice(4040, 800, replace=False)
    np.random.seed(seed)
    sampled_idx = np.random.choice(
        len(names), labeled_count, replace=False
    )

    names_np = np.asarray(sorted(names))
    labeled = names_np[sampled_idx].tolist()
    labeled_set = set(labeled)
    unlabeled = [n for n in sorted(names) if n not in labeled_set]

    return labeled, unlabeled


def write_split(path: Path, names: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def read_labelme_box(path: Path, h: int, w: int) -> np.ndarray:
    obj = json.loads(path.read_text(encoding="utf-8"))
    mask = np.zeros((h, w), dtype=np.uint8)

    for shape in obj.get("shapes", []):
        points = shape.get("points", [])
        if not points:
            continue
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
        x1 = max(0, min(w, int(math.floor(min(xs)))))
        y1 = max(0, min(h, int(math.floor(min(ys)))))
        x2 = max(0, min(w, int(math.ceil(max(xs)))))
        y2 = max(0, min(h, int(math.ceil(max(ys)))))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = 255
    return mask


def load_box_mask(path: Path, h: int, w: int, fmt: str) -> np.ndarray:
    if fmt == "labelme_json" or path.suffix.lower() == ".json":
        return read_labelme_box(path, h, w)

    box = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if box is None:
        raise FileNotFoundError(path)
    if box.shape != (h, w):
        box = cv2.resize(
            box, (w, h), interpolation=cv2.INTER_NEAREST
        )
    return (box > 0).astype(np.uint8) * 255


def official_mask_to_edge(mask_u8: np.ndarray) -> np.ndarray:
    """
    Released utils/edegs.py behavior:
      boundary = max_pool(1-mask, 3) - (1-mask)
      boundary = max_pool(boundary, 3) * mask
    """
    mask = torch.from_numpy(
        (mask_u8.astype(np.float32) / 255.0)
    ).unsqueeze(0).unsqueeze(0)

    boundary = F.max_pool2d(
        1 - mask, kernel_size=3, stride=1, padding=1
    )
    boundary = boundary - (1 - mask)
    boundary = F.max_pool2d(
        boundary, kernel_size=3, stride=1, padding=1
    ) * mask

    return np.clip(
        boundary.squeeze().numpy() * 255.0, 0, 255
    ).astype(np.uint8)


# ---------------------------------------------------------------------------
# Released strong augmentation
# ---------------------------------------------------------------------------

def build_official_augmentation(strict_v1=False):
    if A is None:
        raise ImportError(
            "Albumentations is required for the released ANet augmentation. "
            "For closest behavior use: pip install albumentations==1.3.1"
        )

    version = getattr(A, "__version__", "unknown")
    if strict_v1 and not str(version).startswith("1."):
        raise RuntimeError(
            f"Expected albumentations 1.x for faithful augmentation, got {version}"
        )

    ops = [
        A.ColorJitter(0.5, 0.5, 0.5, 0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]

    # A.Flip existed in the released environment but was removed in newer
    # Albumentations. Use it when available; otherwise use a close fallback.
    if hasattr(A, "Flip"):
        ops.append(A.Flip(p=0.5))
    else:
        ops.append(
            A.OneOf(
                [
                    A.HorizontalFlip(p=1.0),
                    A.VerticalFlip(p=1.0),
                ],
                p=0.5,
            )
        )

    ops += [
        A.GaussNoise(p=0.5),
        A.Blur(p=0.2),
        A.ShiftScaleRotate(rotate_limit=30),
        A.RGBShift(p=0.5),
        A.CLAHE(p=0.5),
        A.ChannelShuffle(p=0.5),
        A.ISONoise(p=0.5),
        A.Superpixels(p=0.1),
        A.ToGray(p=0.2),
        A.CoarseDropout(),
        A.RandomGridShuffle(p=0.2),
        A.Emboss(p=0.5),
        A.Posterize(p=0.5),
        A.ToSepia(p=0.2),
        A.Perspective(p=0.5),
    ]

    return A.Compose(
        ops,
        additional_targets={
            "image2": "image",
            "mask": "mask",
            "edge": "mask",
        },
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ANetDataset(Dataset):
    def __init__(
        self,
        names: Sequence[str],
        data_cfg: dict,
        image_size: int,
        training: bool,
        augment: bool = False,
        strict_albumentations_v1: bool = False,
        return_gt: bool = True,
    ):
        self.names = list(names)
        self.cfg = data_cfg
        self.image_size = int(image_size)
        self.training = bool(training)
        self.return_gt = bool(return_gt)

        self.image_dir = Path(data_cfg["image_dir"])
        self.mask_dir = Path(data_cfg["mask_dir"])
        self.box_dir = Path(data_cfg["box_dir"])
        self.edge_dir = (
            None
            if data_cfg.get("edge_dir") in (None, "")
            else Path(data_cfg["edge_dir"])
        )

        self.image_suffix = data_cfg["image_suffix"]
        self.mask_suffix = data_cfg["mask_suffix"]
        self.box_suffix = data_cfg["box_suffix"]
        self.edge_suffix = data_cfg.get("edge_suffix", ".png")
        self.box_format = data_cfg.get("box_format", "labelme_json")

        self.aug = (
            build_official_augmentation(strict_albumentations_v1)
            if training and augment else None
        )

        # Exactly mirrors released torchvision post-augmentation transforms.
        self.img_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )
        self.gt_transform = transforms.Compose(
            [
                transforms.Resize((self.image_size, self.image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.names)

    def _paths(self, name):
        return (
            self.image_dir / f"{name}{self.image_suffix}",
            self.mask_dir / f"{name}{self.mask_suffix}",
            self.box_dir / f"{name}{self.box_suffix}",
        )

    def __getitem__(self, index):
        name = self.names[index]
        image_path, mask_path, box_path = self._paths(name)

        with Image.open(image_path) as im:
            image_pil = im.convert("RGB")

        w0, h0 = image_pil.size
        image_np = np.asarray(image_pil, dtype=np.uint8)

        box_mask = load_box_mask(
            box_path, h=h0, w=w0, fmt=self.box_format
        )
        box_image_np = (
            image_np.astype(np.float32)
            * (box_mask[..., None].astype(np.float32) / 255.0)
        ).astype(np.uint8)

        gt_pil = None
        gt_np = None
        edge_np = None

        if self.training or self.return_gt:
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)
            with Image.open(mask_path) as gt_im:
                gt_pil = gt_im.convert("L")
            gt_np = np.asarray(gt_pil, dtype=np.uint8)

            if gt_np.shape != image_np.shape[:2]:
                image_np = cv2.resize(
                    image_np,
                    (gt_np.shape[1], gt_np.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                box_image_np = cv2.resize(
                    box_image_np,
                    (gt_np.shape[1], gt_np.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                box_mask = cv2.resize(
                    box_mask,
                    (gt_np.shape[1], gt_np.shape[0]),
                    interpolation=cv2.INTER_NEAREST,
                )
                h0, w0 = gt_np.shape

            if self.training:
                if self.edge_dir is not None:
                    edge_path = self.edge_dir / f"{name}{self.edge_suffix}"
                    edge_np = cv2.imread(
                        str(edge_path), cv2.IMREAD_GRAYSCALE
                    )
                    if edge_np is None:
                        raise FileNotFoundError(edge_path)
                else:
                    edge_np = official_mask_to_edge(gt_np)

        if self.training and self.aug is not None:
            augmented = self.aug(
                image=image_np,
                image2=box_image_np,
                mask=gt_np,
                edge=edge_np,
            )
            image_np = augmented["image"]
            box_image_np = augmented["image2"]
            gt_np = augmented["mask"]
            edge_np = augmented["edge"]

        image_t = self.img_transform(Image.fromarray(image_np))
        box_image_t = self.img_transform(Image.fromarray(box_image_np))

        out = {
            "image": image_t,
            "box_image": box_image_t,
            "name": name,
            "orig_h": int(h0),
            "orig_w": int(w0),
        }

        if gt_np is not None:
            out["gt"] = self.gt_transform(Image.fromarray(gt_np))
        if edge_np is not None:
            out["edge"] = self.gt_transform(Image.fromarray(edge_np))

        return out


# ---------------------------------------------------------------------------
# Loss and learning rate — released TrainDDP.py behavior
# ---------------------------------------------------------------------------

def adjust_lr(
    now_epoch: int,
    top_epoch: int,
    max_epoch: int,
    init_lr: float,
    top_lr: float,
    min_lr: float,
    optimizer,
):
    # `init_lr` is kept for API parity. The released function effectively
    # warms from min_lr to top_lr.
    del init_lr

    if now_epoch < top_epoch:
        lr = min_lr + abs(top_lr - min_lr) / top_epoch * now_epoch
    else:
        progress = (now_epoch - top_epoch) / (max_epoch - top_epoch)
        lr = min_lr + (top_lr - min_lr) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )

    for group in optimizer.param_groups:
        group["lr"] = lr
    return lr


def compute_anet_loss(preds, gts, edges, step_idx, total_step, cfg_train):
    # UAL restarts from 0 -> 1 inside EVERY epoch, matching released code.
    ual_coef = get_ual_coef(step_idx / total_step)
    ual_loss = cal_ual(preds[4], gts) * ual_coef

    loss_init = (
        structure_loss(preds[0], gts) * 0.0625
        + structure_loss(preds[1], gts) * 0.125
        + structure_loss(preds[2], gts) * 0.25
        + structure_loss(preds[3], gts) * 0.5
    )
    loss_final = structure_loss(preds[4], gts)

    # IMPORTANT: released REU boundary prediction is already edge_enhance()
    # clamped into [0,1], so official dice_loss is applied directly, no sigmoid.
    loss_edge = (
        dice_loss(preds[6], edges) * 0.125
        + dice_loss(preds[7], edges) * 0.25
        + dice_loss(preds[8], edges) * 0.5
    )

    total = (
        loss_init
        + loss_final
        + float(cfg_train["edge_loss_weight"]) * loss_edge
        + float(cfg_train["ual_loss_weight"]) * ual_loss
    )
    return total, loss_init, loss_final, loss_edge, ual_loss, ual_coef


# ---------------------------------------------------------------------------
# Validation and generation
# ---------------------------------------------------------------------------

@torch.no_grad()
def validate_mae(model, loader, device):
    model.eval()
    maes = []

    for batch in tqdm(loader, desc="VAL F20->U80", ncols=96):
        image = batch["image"].to(device, non_blocking=True)
        box_image = batch["box_image"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        pred = torch.sigmoid(model(image, box_image)[4])
        maes.extend(
            torch.mean(torch.abs(gt - pred), dim=(1, 2, 3))
            .detach()
            .cpu()
            .tolist()
        )

    return float(np.mean(maes))


@torch.no_grad()
def generate_pseudo_labels(
    model,
    loader,
    device,
    mask_dir: Path,
    edge_dir: Path,
):
    model.eval()
    mask_dir.mkdir(parents=True, exist_ok=True)
    edge_dir.mkdir(parents=True, exist_ok=True)

    mae_list = []

    for batch in tqdm(loader, desc="GENERATE U80", ncols=96):
        image = batch["image"].to(device, non_blocking=True)
        box_image = batch["box_image"].to(device, non_blocking=True)

        result = model(image, box_image)
        masks = torch.sigmoid(result[4]).detach().cpu().numpy()[:, 0]
        edges = result[8].detach().cpu().numpy()[:, 0]

        if "gt" in batch:
            gt = batch["gt"]
            pred_t = torch.from_numpy(masks).unsqueeze(1)
            mae_list.extend(
                torch.mean(torch.abs(gt - pred_t), dim=(1, 2, 3)).tolist()
            )

        for i, name in enumerate(batch["name"]):
            h = int(batch["orig_h"][i])
            w = int(batch["orig_w"][i])

            mask = cv2.resize(
                masks[i], (w, h), interpolation=cv2.INTER_LINEAR
            )
            edge = cv2.resize(
                edges[i], (w, h), interpolation=cv2.INTER_LINEAR
            )

            cv2.imwrite(
                str(mask_dir / f"{name}.png"),
                np.clip(mask * 255.0, 0, 255).astype(np.uint8),
            )
            cv2.imwrite(
                str(edge_dir / f"{name}.png"),
                np.clip(edge * 255.0, 0, 255).astype(np.uint8),
            )

    return float(np.mean(mae_list)) if mae_list else None


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def build_paths(cfg):
    out = Path(cfg["output"]["root"])
    stats_name = cfg.get("logging", {}).get("stats_dir", "frequency_stats")
    return dict(
        root=out,
        split_dir=out / cfg["output"]["split_dir"],
        ckpt_dir=out / cfg["output"]["checkpoint_dir"],
        pseudo_mask_dir=out / cfg["output"]["pseudo_mask_dir"],
        pseudo_edge_dir=out / cfg["output"]["pseudo_edge_dir"],
        log_file=out / cfg["output"]["log_file"],
        stats_dir=out / stats_name,
        train_csv=out / stats_name / "train_metrics.csv",
        freq_csv=out / stats_name / "frequency_stats.csv",
    )


def _normalize_list_name(line: str) -> str:
    x = line.strip().replace("\\", "/")
    if not x or x.startswith("#"):
        return ""
    x = x.split("/")[-1]
    for suf in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"):
        if x.lower().endswith(suf):
            x = x[:-len(suf)]
            break
    return x


def read_sample_list(path: str) -> List[str]:
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"sample_list not found: {p}")
    names = []
    seen = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        name = _normalize_list_name(line)
        if name and name not in seen:
            names.append(name)
            seen.add(name)
    if not names:
        raise RuntimeError(f"No valid sample names found in {p}")
    return names


def prepare_splits(cfg, paths, logger):
    exp = cfg["experiment"]
    data_cfg = cfg["data"]
    tcfg = cfg["train"]

    all_names = list_stems(Path(data_cfg["image_dir"]), data_cfg["image_suffix"])
    expected_total = int(exp.get("expected_total", len(all_names)))
    if bool(exp.get("strict_total", False)) and len(all_names) != expected_total:
        raise RuntimeError(
            f"Expected exactly {expected_total} images, found {len(all_names)} in {data_cfg['image_dir']}"
        )

    sample_list = tcfg.get("sample_list")
    if sample_list:
        labeled = read_sample_list(sample_list)
        all_set = set(all_names)
        missing = [n for n in labeled if n not in all_set]
        if missing:
            raise RuntimeError(
                f"{len(missing)} names from sample_list are not in image_dir; first={missing[:10]}"
            )
        labeled_set = set(labeled)
        unlabeled = [n for n in all_names if n not in labeled_set]
        write_split(paths["split_dir"] / "train_clean_list.txt", labeled)
        write_split(paths["split_dir"] / "remaining_list.txt", unlabeled)
        logger.info(
            "Experiment clean-pseudo TXT mode | list=%s | output=%s",
            sample_list,
            paths["root"],
        )
        logger.info(
            "Clean-pseudo mode | train_txt=%s | clean_train=%d | remaining=%d",
            sample_list,
            len(labeled),
            len(unlabeled),
        )
        logger.info(
            "Training supervision masks: %s",
            tcfg.get("target_mask_dir") or data_cfg["mask_dir"],
        )
        logger.info(
            "Training source=clean_pseudo | train_samples=%d | remaining=%d",
            len(labeled),
            len(unlabeled),
        )
        return labeled, unlabeled

    ratio = int(exp.get("ratio", 20))
    if bool(exp.get("official_ratio_only", True)) and ratio not in (1, 5, 10, 20):
        raise ValueError(f"Official Noisy-COD ratio must be one of 1/5/10/20, got {ratio}")
    labeled_count = int(ratio * 400 / 10)  # released split_data.py rule
    labeled, unlabeled = official_f20_split(
        image_dir=Path(data_cfg["image_dir"]),
        image_suffix=data_cfg["image_suffix"],
        expected_total=expected_total,
        labeled_count=labeled_count,
        seed=int(exp.get("seed", 2024)),
        strict_total=bool(exp.get("strict_total", False)),
    )
    write_split(paths["split_dir"] / f"F{ratio}_labeled_{len(labeled)}.txt", labeled)
    write_split(paths["split_dir"] / f"U{100-ratio}_generate_{len(unlabeled)}.txt", unlabeled)
    logger.info(
        "Official ratio split | F%d | total=%d | labeled=%d | generate=%d | seed=%d",
        ratio,
        len(labeled) + len(unlabeled),
        len(labeled),
        len(unlabeled),
        int(exp.get("seed", 2024)),
    )
    return labeled, unlabeled


def build_model(cfg, device, logger):
    mcfg = cfg["model"]
    model = NoisyCODANet(
        backbone_name=mcfg.get("backbone_name", "convnext_base.fb_in22k_ft_in1k_384"),
        channels=int(mcfg.get("channels", 64)),
        ablation=str(mcfg.get("ablation", "A0")),
        router_hidden=int(mcfg.get("router_hidden", 32)),
        router_temperature=float(mcfg.get("router_temperature", 1.0)),
        router_alpha_init=float(mcfg.get("router_alpha_init", 0.0)),
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    router_params = sum(
        p.numel() for n, p in model.named_parameters()
        if "freq_router" in n or "router_alpha" in n
    )
    logger.info("ANet parameters: %.3f M | trainable=%.3f M", params / 1e6, trainable / 1e6)
    logger.info("Backbone: %s", mcfg.get("backbone_name"))
    logger.info(
        "DWT ablation: %s | router_hidden=%s | T=%.3f | alpha_init=%.4f | router_params=%.4f M",
        str(mcfg.get("ablation", "A0")),
        mcfg.get("router_hidden", 32),
        float(mcfg.get("router_temperature", 1.0)),
        float(mcfg.get("router_alpha_init", 0.0)),
        router_params / 1e6,
    )
    return model


class ScalarMeter:
    def __init__(self):
        self.sum = 0.0
        self.n = 0

    def update(self, value, n=1):
        self.sum += float(value) * int(n)
        self.n += int(n)

    @property
    def avg(self):
        return self.sum / max(self.n, 1)


class FrequencyEpochMeter:
    """Average the model's detached router diagnostics over one training epoch."""
    def __init__(self):
        self.rows = {}

    def update(self, model):
        stats = model.get_frequency_stats()
        for branch_name in ("rgb_branch", "box_branch"):
            branch = stats.get(branch_name, {}) or {}
            alpha = branch.get("alpha", [float("nan")] * 4)
            for sid in range(1, 5):
                stage = branch.get(f"stage{sid}")
                if not stage:
                    continue
                key = (branch_name, sid)
                if key not in self.rows:
                    self.rows[key] = {
                        "count": 0,
                        "mean": np.zeros(4, dtype=np.float64),
                        "std": np.zeros(4, dtype=np.float64),
                        "entropy": 0.0,
                        "entropy_norm": 0.0,
                        "alpha": 0.0,
                    }
                r = self.rows[key]
                r["count"] += 1
                r["mean"] += np.asarray(stage["mean"], dtype=np.float64)
                r["std"] += np.asarray(stage["std"], dtype=np.float64)
                r["entropy"] += float(stage["entropy"])
                r["entropy_norm"] += float(stage["entropy_norm"])
                r["alpha"] += float(alpha[sid - 1]) if len(alpha) >= sid else float("nan")

    def summary(self):
        out = {}
        for key, r in self.rows.items():
            n = max(r["count"], 1)
            out[key] = {
                "mean": (r["mean"] / n).tolist(),
                "std": (r["std"] / n).tolist(),
                "entropy": r["entropy"] / n,
                "entropy_norm": r["entropy_norm"] / n,
                "alpha": r["alpha"] / n,
            }
        return out


def _append_csv(path: Path, fieldnames, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def _write_frequency_csv(path: Path, epoch: int, summary: dict):
    fields = [
        "epoch", "branch", "stage", "base_prior", "alpha",
        "LL_mean", "LH_mean", "HL_mean", "HH_mean",
        "LL_std", "LH_std", "HL_std", "HH_std",
        "entropy", "entropy_norm",
    ]
    for (branch, sid), st in sorted(summary.items()):
        mean, std = st["mean"], st["std"]
        row = {
            "epoch": epoch,
            "branch": branch,
            "stage": sid,
            "base_prior": "HH" if sid <= 2 else "LL",
            "alpha": f"{st['alpha']:.8f}",
            "LL_mean": f"{mean[0]:.8f}",
            "LH_mean": f"{mean[1]:.8f}",
            "HL_mean": f"{mean[2]:.8f}",
            "HH_mean": f"{mean[3]:.8f}",
            "LL_std": f"{std[0]:.8f}",
            "LH_std": f"{std[1]:.8f}",
            "HL_std": f"{std[2]:.8f}",
            "HH_std": f"{std[3]:.8f}",
            "entropy": f"{st['entropy']:.8f}",
            "entropy_norm": f"{st['entropy_norm']:.8f}",
        }
        _append_csv(path, fields, row)


def _log_frequency_summary(logger, summary: dict):
    if not summary:
        logger.info("Frequency router stats: unavailable for this ablation")
        return
    logger.info("Frequency router diagnostics (epoch mean; entropy max=ln4=1.386294):")
    for branch in ("rgb_branch", "box_branch"):
        pretty = "RGB" if branch == "rgb_branch" else "BOX"
        for sid in range(1, 5):
            st = summary.get((branch, sid))
            if st is None:
                continue
            m, sd = st["mean"], st["std"]
            logger.info(
                "  %s S%d base=%s | alpha=%+.5f | W[LL/LH/HL/HH]=%.4f/%.4f/%.4f/%.4f | "
                "STD=%.4f/%.4f/%.4f/%.4f | H=%.4f (%.1f%%)",
                pretty, sid, "HH" if sid <= 2 else "LL", st["alpha"],
                m[0], m[1], m[2], m[3], sd[0], sd[1], sd[2], sd[3],
                st["entropy"], st["entropy_norm"] * 100.0,
            )


def train(cfg, model, labeled, unlabeled, paths, device, logger):
    tcfg = cfg["train"]
    lcfg = cfg.get("logging", {})

    # Training supervision may be GT or a clean-pseudo folder (e.g. S1_GT).
    train_data_cfg = dict(cfg["data"])
    if tcfg.get("target_mask_dir"):
        train_data_cfg["mask_dir"] = tcfg["target_mask_dir"]
        train_data_cfg["mask_suffix"] = tcfg.get(
            "target_mask_suffix", train_data_cfg.get("mask_suffix", ".png")
        )

    train_set = ANetDataset(
        names=labeled,
        data_cfg=train_data_cfg,
        image_size=tcfg["image_size"],
        training=True,
        augment=bool(tcfg["augment"]),
        strict_albumentations_v1=bool(tcfg.get("strict_albumentations_v1", False)),
        return_gt=True,
    )

    # Validation always uses the original GT directory from cfg['data'].
    val_set = ANetDataset(
        names=unlabeled,
        data_cfg=cfg["data"],
        image_size=tcfg["image_size"],
        training=False,
        augment=False,
        return_gt=True,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=int(tcfg["batch_size"]),
        shuffle=True,
        num_workers=int(tcfg["num_workers"]),
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=min(160, max(1, int(tcfg["batch_size"]) * 4)),
        shuffle=False,
        num_workers=int(tcfg["num_workers"]),
        pin_memory=True,
    )

    if str(tcfg["optimizer"]).lower() != "adam":
        raise ValueError("ANet config requires optimizer='adam'.")

    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["init_lr"]))
    amp_enabled = bool(tcfg["amp"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    epochs = int(tcfg["epochs"])
    total_step = len(train_loader)
    best_mae = float("inf")
    best_epoch = -1

    paths["ckpt_dir"].mkdir(parents=True, exist_ok=True)
    paths["stats_dir"].mkdir(parents=True, exist_ok=True)
    best_path = paths["ckpt_dir"] / "Net_epoch_best.pth"

    logger.info(
        "Training setup | epochs=%d | batch=%d | steps/epoch=%d | AMP=%s | augment=%s",
        epochs, int(tcfg["batch_size"]), total_step, amp_enabled, bool(tcfg["augment"]),
    )
    logger.info(
        "LR schedule | init=%.2e | top_epoch=%d | top_lr=%.2e | min_lr=%.2e",
        float(tcfg["init_lr"]), int(tcfg["top_epoch"]), float(tcfg["top_lr"]), float(tcfg["min_lr"]),
    )
    logger.info(
        "Loss weights | edge=%.3f | ual=%.3f",
        float(tcfg["edge_loss_weight"]), float(tcfg["ual_loss_weight"]),
    )
    logger.info("Detailed metrics CSV: %s", paths["train_csv"])
    logger.info("Frequency stats CSV: %s", paths["freq_csv"])

    metric_fields = [
        "epoch", "lr", "loss_total", "loss_init", "loss_final", "loss_edge",
        "loss_ual", "ual_coef_mean", "val_mae", "best_mae", "best_epoch",
        "epoch_seconds", "gpu_alloc_gb", "gpu_reserved_gb", "amp_scale",
    ]

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        lr = adjust_lr(
            now_epoch=epoch,
            top_epoch=int(tcfg["top_epoch"]),
            max_epoch=epochs,
            init_lr=float(tcfg["init_lr"]),
            top_lr=float(tcfg["top_lr"]),
            min_lr=float(tcfg["min_lr"]),
            optimizer=optimizer,
        )

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        model.train()
        meters = {
            "total": ScalarMeter(),
            "init": ScalarMeter(),
            "final": ScalarMeter(),
            "edge": ScalarMeter(),
            "ual": ScalarMeter(),
            "ual_coef": ScalarMeter(),
        }
        freq_meter = FrequencyEpochMeter()

        pbar = tqdm(
            enumerate(train_loader, start=1),
            total=total_step,
            desc=f"TRAIN E{epoch:03d}/{epochs:03d}",
            ncols=160,
        )

        for i, batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            box_images = batch["box_image"].to(device, non_blocking=True)
            gts = batch["gt"].to(device, non_blocking=True)
            edges = batch["edge"].to(device, non_blocking=True)
            bs = int(images.shape[0])

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                preds = model(images, box_images)
                loss, loss_init, loss_final, loss_edge, loss_ual, ual_coef = compute_anet_loss(
                    preds, gts, edges, i, total_step, tcfg
                )

            # Capture router state from this TRAIN forward before validation can overwrite it.
            if str(cfg["model"].get("ablation", "A0")).upper() in ("A2", "A3"):
                freq_meter.update(model)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            meters["total"].update(loss.detach().item(), bs)
            meters["init"].update(loss_init.detach().item(), bs)
            meters["final"].update(loss_final.detach().item(), bs)
            meters["edge"].update(loss_edge.detach().item(), bs)
            meters["ual"].update(loss_ual.detach().item(), bs)
            meters["ual_coef"].update(ual_coef, bs)

            pbar.set_postfix(
                lr=f"{lr:.2e}",
                loss=f"{meters['total'].avg:.4f}",
                final=f"{meters['final'].avg:.4f}",
                edge=f"{meters['edge'].avg:.4f}",
                ual=f"{meters['ual'].avg:.4f}",
            )

            print_freq = int(lcfg.get("print_freq", 0))
            if print_freq > 0 and (i % print_freq == 0 or i == total_step):
                logger.info(
                    "STEP E%03d %04d/%04d | loss=%.5f init=%.5f final=%.5f edge=%.5f ual=%.5f uc=%.3f",
                    epoch, i, total_step,
                    meters["total"].avg, meters["init"].avg, meters["final"].avg,
                    meters["edge"].avg, meters["ual"].avg, meters["ual_coef"].avg,
                )

        freq_summary = freq_meter.summary()
        epoch_seconds = time.time() - epoch_t0
        gpu_alloc = 0.0
        gpu_reserved = 0.0
        if device.type == "cuda":
            gpu_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
            gpu_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

        logger.info(
            "Epoch %03d | lr=%.8f | total=%.6f | init=%.6f | final=%.6f | "
            "edge=%.6f | ual=%.6f | ual_coef=%.4f | time=%.1fs | GPU=%.2f/%.2f GB",
            epoch, lr, meters["total"].avg, meters["init"].avg,
            meters["final"].avg, meters["edge"].avg, meters["ual"].avg,
            meters["ual_coef"].avg, epoch_seconds, gpu_alloc, gpu_reserved,
        )

        freq_interval = int(lcfg.get("frequency_interval", 1))
        if freq_summary and (epoch % max(freq_interval, 1) == 0):
            _log_frequency_summary(logger, freq_summary)
            if bool(lcfg.get("save_frequency_csv", True)):
                _write_frequency_csv(paths["freq_csv"], epoch, freq_summary)

        if epoch > epochs - int(tcfg["save_last_epochs"]):
            ep_path = paths["ckpt_dir"] / f"Net_epoch_{epoch}.pth"
            torch.save(model.state_dict(), ep_path)
            logger.info("Saved tail checkpoint: %s", ep_path)

        val_mae = float("nan")
        if epoch % int(tcfg["validate_every"]) == 0:
            val_t0 = time.time()
            val_mae = validate_mae(model, val_loader, device)
            val_seconds = time.time() - val_t0
            logger.info(
                "VAL epoch=%d | MAE=%.6f | best=%s @ epoch=%d | val_time=%.1fs",
                epoch,
                val_mae,
                "inf" if not np.isfinite(best_mae) else f"{best_mae:.6f}",
                best_epoch,
                val_seconds,
            )

            if val_mae < best_mae:
                best_mae = val_mae
                best_epoch = epoch
                torch.save(model.state_dict(), best_path)
                logger.info("Saved best ANet: %s", best_path)

        amp_scale = float(scaler.get_scale()) if amp_enabled else 1.0
        _append_csv(
            paths["train_csv"],
            metric_fields,
            {
                "epoch": epoch,
                "lr": f"{lr:.10f}",
                "loss_total": f"{meters['total'].avg:.8f}",
                "loss_init": f"{meters['init'].avg:.8f}",
                "loss_final": f"{meters['final'].avg:.8f}",
                "loss_edge": f"{meters['edge'].avg:.8f}",
                "loss_ual": f"{meters['ual'].avg:.8f}",
                "ual_coef_mean": f"{meters['ual_coef'].avg:.8f}",
                "val_mae": "" if not np.isfinite(val_mae) else f"{val_mae:.8f}",
                "best_mae": "" if not np.isfinite(best_mae) else f"{best_mae:.8f}",
                "best_epoch": best_epoch,
                "epoch_seconds": f"{epoch_seconds:.3f}",
                "gpu_alloc_gb": f"{gpu_alloc:.4f}",
                "gpu_reserved_gb": f"{gpu_reserved:.4f}",
                "amp_scale": f"{amp_scale:.1f}",
            },
        )

    if not best_path.is_file():
        torch.save(model.state_dict(), best_path)
        best_epoch = epochs
        logger.info("No validation checkpoint existed; saved final model as best: %s", best_path)

    logger.info(
        "Training complete | best MAE=%s @ epoch=%d",
        "n/a" if not np.isfinite(best_mae) else f"{best_mae:.6f}",
        best_epoch,
    )
    logger.info("Train metrics CSV: %s", paths["train_csv"])
    logger.info("Frequency CSV: %s", paths["freq_csv"])
    return best_path


def load_checkpoint(model, path: Path, device):
    state = torch.load(path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    cleaned = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[len("module."):]
        cleaned[k] = v
    model.load_state_dict(cleaned, strict=True)


def run_generate(cfg, model, unlabeled, paths, device, checkpoint, logger):
    load_checkpoint(model, Path(checkpoint), device)
    logger.info("Loaded ANet checkpoint: %s", checkpoint)

    gcfg = cfg["generate"]
    gen_set = ANetDataset(
        names=unlabeled,
        data_cfg=cfg["data"],
        image_size=gcfg["image_size"],
        training=False,
        augment=False,
        return_gt=True,  # only for optional MAE reporting; never model input
    )
    gen_loader = DataLoader(
        gen_set,
        batch_size=int(gcfg["batch_size"]),
        shuffle=False,
        num_workers=int(gcfg["num_workers"]),
        pin_memory=True,
    )

    mae = generate_pseudo_labels(
        model=model,
        loader=gen_loader,
        device=device,
        mask_dir=paths["pseudo_mask_dir"],
        edge_dir=paths["pseudo_edge_dir"],
    )

    logger.info("Pseudo masks -> %s", paths["pseudo_mask_dir"])
    logger.info("Pseudo edges -> %s", paths["pseudo_edge_dir"])
    if mae is not None:
        logger.info(
            "Diagnostic U80 pseudo-label MAE (GT not used as input): %.6f",
            mae,
        )


def parse_args():
    parser = argparse.ArgumentParser("Noisy-COD ANet A3 with detailed logging")
    parser.add_argument("--config", default="configs/anet_noisycod_A3_detailed.py", type=str)
    parser.add_argument("--mode", choices=["train", "generate", "all"], default="all")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")

    # Runtime overrides matching your ratio / clean-pseudo workflow.
    parser.add_argument("--ratio", type=int, default=None)
    parser.add_argument("--train-txt", type=str, default=None)
    parser.add_argument("--target-mask-dir", type=str, default=None)
    parser.add_argument("--output-root", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_cfg(args.config)

    if args.ratio is not None:
        cfg["experiment"]["ratio"] = int(args.ratio)
    if args.train_txt is not None:
        cfg["train"]["sample_list"] = args.train_txt
    if args.target_mask_dir is not None:
        cfg["train"]["target_mask_dir"] = args.target_mask_dir
    if args.output_root is not None:
        cfg["output"]["root"] = args.output_root

    seed_everything(int(cfg["experiment"].get("seed", 2024)))

    paths = build_paths(cfg)
    paths["root"].mkdir(parents=True, exist_ok=True)
    logger = setup_logger(paths["log_file"])

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Config: %s", Path(args.config).resolve())
    logger.info("Output root: %s", paths["root"])
    logger.info("Seed: %d", int(cfg["experiment"].get("seed", 2024)))

    labeled, unlabeled = prepare_splits(cfg, paths, logger)
    model = build_model(cfg, device, logger)

    best_path = None
    if args.mode in ("train", "all"):
        best_path = train(cfg, model, labeled, unlabeled, paths, device, logger)

    if args.mode in ("generate", "all"):
        checkpoint = args.checkpoint or best_path
        if checkpoint is None:
            checkpoint = paths["ckpt_dir"] / "Net_epoch_best.pth"
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(f"Generation checkpoint not found: {checkpoint}")
        run_generate(cfg, model, unlabeled, paths, device, checkpoint, logger)


if __name__ == "__main__":
    main()
