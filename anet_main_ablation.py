# -*- coding: utf-8 -*-
"""
Noisy-COD ANet training + pseudo-label generation with configurable GT ratio.

Examples
--------
20% (config default):
    python anet_main.py \
        --config configs/anet_noisycod_ratio.py \
        --mode all \
        --device cuda:0

10% and custom output path:
    python anet_main.py \
        --config configs/anet_noisycod_ratio.py \
        --mode all \
        --ratio 10 \
        --output-root /your/path/ANet_F10 \
        --device cuda:0

5% train only:
    python anet_main.py \
        --config configs/anet_noisycod_ratio.py \
        --mode train \
        --ratio 5 \
        --output-root ./ANet_outputs/NoisyCOD_ANet_F5

Generate only:
    python anet_main.py \
        --config configs/anet_noisycod_ratio.py \
        --mode generate \
        --ratio 10 \
        --output-root ./ANet_outputs/NoisyCOD_ANet_F10 \
        --checkpoint ./ANet_outputs/NoisyCOD_ANet_F10/checkpoints/Net_epoch_best.pth

Important
---------
This follows the released Noisy-COD split rule rather than `4040 * ratio / 100`:

    labeled_count = int(ratio * 400 / 10)

Thus F20 = 800, not 808.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import math
import os
import random
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

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

from methods.anet20_noisycod_ablation import (
    NoisyCODANet,
    cal_ual,
    dice_loss,
    get_ual_coef,
    structure_loss,
)


# -----------------------------------------------------------------------------
# Config / runtime
# -----------------------------------------------------------------------------

OFFICIAL_RATIOS = (1, 5, 10, 20)


def load_cfg(path: str) -> dict:
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location("anet_ratio_cfg", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import config: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.cfg


def parse_args():
    parser = argparse.ArgumentParser("Noisy-COD ANet ratio reproduction")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/anet_dwt_ablation.py",
    )
    parser.add_argument(
        "--mode",
        choices=["train", "generate", "all"],
        default="all",
    )
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--checkpoint", type=str, default=None)

    # NEW: override GT ratio from bash.
    parser.add_argument(
        "--ratio",
        type=int,
        default=None,
        help="GT ratio. Faithful Noisy-COD ratios: 1, 5, 10, 20.",
    )

    # NEW: override output root from bash.
    parser.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override cfg['output']['root'].",
    )

    # DWT ablation switch.
    parser.add_argument(
        "--ablation",
        choices=["A0", "A1", "A2"],
        default=None,
        help="A0=original, A1=four-band mean, A2=adaptive frequency router.",
    )

    # Optional clean-pseudo training mode. When --train-list is provided,
    # ratio sampling is bypassed for TRAINING samples.
    parser.add_argument(
        "--train-list",
        type=str,
        default=None,
        help="TXT file containing clean training image names/stems.",
    )
    parser.add_argument(
        "--train-mask-dir",
        type=str,
        default=None,
        help="Mask directory used as training supervision, e.g. clean pseudo labels.",
    )
    parser.add_argument(
        "--train-mask-suffix",
        type=str,
        default=None,
        help="Training mask suffix, usually .png.",
    )
    return parser.parse_args()


def apply_cli_overrides(cfg: dict, args) -> dict:
    if args.ratio is not None:
        cfg["experiment"]["ratio"] = int(args.ratio)

    if args.output_root is not None:
        cfg["output"]["root"] = str(args.output_root)

    if args.ablation is not None:
        cfg["model"]["ablation"] = str(args.ablation)

    if args.train_list is not None:
        cfg["train"]["sample_list"] = str(args.train_list)

    if args.train_mask_dir is not None:
        cfg["train"]["target_mask_dir"] = str(args.train_mask_dir)

    if args.train_mask_suffix is not None:
        cfg["train"]["target_mask_suffix"] = str(args.train_mask_suffix)

    return cfg


def validate_ratio_cfg(cfg: dict) -> Tuple[int, int, int]:
    exp = cfg["experiment"]
    ratio = int(exp["ratio"])
    total = int(exp["expected_total"])

    if ratio <= 0 or ratio >= 100:
        raise ValueError(f"ratio must be in (0, 100), got {ratio}")

    if bool(exp.get("official_ratio_only", True)) and ratio not in OFFICIAL_RATIOS:
        raise ValueError(
            f"For faithful Noisy-COD experiments, ratio must be one of "
            f"{OFFICIAL_RATIOS}; got {ratio}."
        )

    # Released Noisy-COD split_data.py behavior.
    labeled_count = int(ratio * 400 / 10)
    unlabeled_count = total - labeled_count

    if labeled_count <= 0 or unlabeled_count <= 0:
        raise ValueError(
            f"Invalid split from ratio={ratio}: "
            f"labeled={labeled_count}, unlabeled={unlabeled_count}."
        )

    return ratio, labeled_count, unlabeled_count


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

    logger = logging.getLogger("ANetRatio")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("[%(asctime)s] %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def build_paths(cfg: dict):
    root = Path(cfg["output"]["root"])
    return dict(
        root=root,
        split_dir=root / cfg["output"]["split_dir"],
        ckpt_dir=root / cfg["output"]["checkpoint_dir"],
        pseudo_mask_dir=root / cfg["output"]["pseudo_mask_dir"],
        pseudo_edge_dir=root / cfg["output"]["pseudo_edge_dir"],
        log_file=root / cfg["output"]["log_file"],
    )


# -----------------------------------------------------------------------------
# Official ratio split
# -----------------------------------------------------------------------------


def list_stems(folder: Path, suffix: str) -> List[str]:
    if not folder.is_dir():
        raise FileNotFoundError(f"Image directory not found: {folder}")

    return sorted(
        p.name[: -len(suffix)]
        for p in folder.iterdir()
        if p.is_file() and p.name.endswith(suffix)
    )


def official_ratio_split(
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
            f"Expected exactly {expected_total} training images, "
            f"but found {len(names)} in {image_dir}."
        )

    if labeled_count > len(names):
        raise ValueError(
            f"labeled_count={labeled_count} > dataset_size={len(names)}"
        )

    # Same NumPy selection style as released Noisy-COD split_data.py.
    np.random.seed(seed)
    sampled_idx = np.random.choice(
        len(names),
        labeled_count,
        replace=False,
    )

    names_np = np.asarray(names)
    labeled = names_np[sampled_idx].tolist()
    labeled_set = set(labeled)
    unlabeled = [name for name in names if name not in labeled_set]

    return labeled, unlabeled


def write_split(path: Path, names: Sequence[str]):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(names) + "\n", encoding="utf-8")


def _normalize_list_name(raw: str) -> str:
    """Accept stem, filename, or a path and return the image stem."""
    raw = raw.strip()
    if not raw or raw.startswith("#"):
        return ""
    # If a line has extra columns, use the first token.
    raw = raw.split()[0]
    name = Path(raw).name
    # Strip one common image/mask extension.
    for suf in (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"):
        if name.lower().endswith(suf):
            return name[: -len(suf)]
    return Path(name).stem if "." in name else name


def read_sample_list(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(f"Training list not found: {path}")

    names = []
    seen = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        name = _normalize_list_name(line)
        if not name or name in seen:
            continue
        seen.add(name)
        names.append(name)

    if not names:
        raise RuntimeError(f"No valid sample names found in: {path}")
    return names


def validate_clean_training_files(names: Sequence[str], cfg: dict):
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]

    image_dir = Path(data_cfg["image_dir"])
    box_dir = Path(data_cfg["box_dir"])
    mask_dir = Path(train_cfg["target_mask_dir"])

    image_suffix = data_cfg["image_suffix"]
    box_suffix = data_cfg["box_suffix"]
    mask_suffix = train_cfg.get("target_mask_suffix", ".png")

    missing_image, missing_box, missing_mask = [], [], []
    for name in names:
        if not (image_dir / f"{name}{image_suffix}").is_file():
            missing_image.append(name)
        if not (box_dir / f"{name}{box_suffix}").is_file():
            missing_box.append(name)
        if not (mask_dir / f"{name}{mask_suffix}").is_file():
            missing_mask.append(name)

    if missing_image or missing_box or missing_mask:
        raise RuntimeError(
            "Clean-list file check failed.\n"
            f"missing images: {len(missing_image)} {missing_image[:10]}\n"
            f"missing boxes: {len(missing_box)} {missing_box[:10]}\n"
            f"missing train masks: {len(missing_mask)} {missing_mask[:10]}"
        )


def prepare_training_names(cfg, paths, logger):
    """
    Two modes:
      1) ratio mode: original Noisy-COD GT split.
      2) clean-list mode: read training names from TXT and supervise them with
         target_mask_dir (typically filtered pseudo labels).

    In clean-list mode, validation/generation candidates are all dataset images
    that are NOT in the clean training TXT.
    """
    train_cfg = cfg["train"]
    sample_list = train_cfg.get("sample_list")

    if sample_list in (None, ""):
        labeled, unlabeled = prepare_splits(cfg, paths, logger)
        return labeled, unlabeled, "ratio_gt"

    target_mask_dir = train_cfg.get("target_mask_dir")
    if target_mask_dir in (None, ""):
        raise ValueError(
            "train.sample_list is set, but train.target_mask_dir is empty. "
            "Point target_mask_dir to your filtered pseudo-label folder."
        )

    train_names = read_sample_list(Path(sample_list))
    validate_clean_training_files(train_names, cfg)

    all_names = list_stems(
        Path(cfg["data"]["image_dir"]), cfg["data"]["image_suffix"]
    )
    all_set = set(all_names)
    unknown = [n for n in train_names if n not in all_set]
    if unknown:
        raise RuntimeError(
            f"TXT contains {len(unknown)} names not present in image_dir: {unknown[:10]}"
        )

    train_set = set(train_names)
    remaining = [n for n in all_names if n not in train_set]

    write_split(paths["split_dir"] / f"clean_train_{len(train_names)}.txt", train_names)
    write_split(paths["split_dir"] / f"remaining_{len(remaining)}.txt", remaining)

    logger.info(
        "Clean-pseudo mode | train_txt=%s | clean_train=%d | remaining=%d",
        sample_list, len(train_names), len(remaining)
    )
    logger.info("Training supervision masks: %s", target_mask_dir)
    return train_names, remaining, "clean_pseudo"


def prepare_splits(cfg, paths, logger):
    exp = cfg["experiment"]
    data_cfg = cfg["data"]

    ratio, expected_labeled, expected_unlabeled = validate_ratio_cfg(cfg)

    labeled, unlabeled = official_ratio_split(
        image_dir=Path(data_cfg["image_dir"]),
        image_suffix=data_cfg["image_suffix"],
        expected_total=int(exp["expected_total"]),
        labeled_count=expected_labeled,
        seed=int(exp["seed"]),
        strict_total=bool(exp["strict_total"]),
    )

    if len(labeled) != expected_labeled or len(unlabeled) != expected_unlabeled:
        raise RuntimeError(
            f"Split mismatch: expected {expected_labeled}/{expected_unlabeled}, "
            f"got {len(labeled)}/{len(unlabeled)}"
        )

    f_name = f"F{ratio}_labeled_{len(labeled)}.txt"
    u_name = f"U{100-ratio}_generate_{len(unlabeled)}.txt"

    write_split(paths["split_dir"] / f_name, labeled)
    write_split(paths["split_dir"] / u_name, unlabeled)

    logger.info(
        "Noisy-COD split | ratio=F%d | total=%d | GT=%d | pseudo=%d | seed=%d",
        ratio,
        len(labeled) + len(unlabeled),
        len(labeled),
        len(unlabeled),
        int(exp["seed"]),
    )
    logger.info("GT split file: %s", paths["split_dir"] / f_name)
    logger.info("Pseudo split file: %s", paths["split_dir"] / u_name)

    return labeled, unlabeled


# -----------------------------------------------------------------------------
# Box / edge helpers
# -----------------------------------------------------------------------------


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
        if not path.is_file():
            raise FileNotFoundError(path)
        return read_labelme_box(path, h, w)

    box = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if box is None:
        raise FileNotFoundError(path)

    if box.shape != (h, w):
        box = cv2.resize(box, (w, h), interpolation=cv2.INTER_NEAREST)

    return (box > 0).astype(np.uint8) * 255


def official_mask_to_edge(mask_u8: np.ndarray) -> np.ndarray:
    mask = torch.from_numpy(
        mask_u8.astype(np.float32) / 255.0
    ).unsqueeze(0).unsqueeze(0)

    boundary = F.max_pool2d(
        1 - mask,
        kernel_size=3,
        stride=1,
        padding=1,
    )
    boundary = boundary - (1 - mask)
    boundary = F.max_pool2d(
        boundary,
        kernel_size=3,
        stride=1,
        padding=1,
    ) * mask

    return np.clip(
        boundary.squeeze().numpy() * 255.0,
        0,
        255,
    ).astype(np.uint8)


# -----------------------------------------------------------------------------
# Augmentation
# -----------------------------------------------------------------------------


def build_official_augmentation(strict_v1=False):
    if A is None:
        raise ImportError(
            "Albumentations is required when train.augment=True. "
            "For closest released behavior: pip install albumentations==1.3.1"
        )

    version = getattr(A, "__version__", "unknown")
    if strict_v1 and not str(version).startswith("1."):
        raise RuntimeError(
            f"Expected albumentations 1.x, got {version}"
        )

    ops = [
        A.ColorJitter(0.5, 0.5, 0.5, 0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
    ]

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

    ops.extend(
        [
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
    )

    return A.Compose(
        ops,
        additional_targets={
            "image2": "image",
            "mask": "mask",
            "edge": "mask",
        },
    )


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


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
            if training and augment
            else None
        )

        self.image_transform = transforms.Compose(
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
                transforms.Resize(
                    (self.image_size, self.image_size),
                    interpolation=transforms.InterpolationMode.NEAREST,
                ),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]

        image_path = self.image_dir / f"{name}{self.image_suffix}"
        mask_path = self.mask_dir / f"{name}{self.mask_suffix}"
        box_path = self.box_dir / f"{name}{self.box_suffix}"

        with Image.open(image_path) as image_file:
            image_pil = image_file.convert("RGB")

        orig_w, orig_h = image_pil.size
        image_np = np.asarray(image_pil, dtype=np.uint8)

        box_mask = load_box_mask(
            box_path,
            h=orig_h,
            w=orig_w,
            fmt=self.box_format,
        )

        # Released ANet second branch input: RGB * filled box mask.
        box_image_np = (
            image_np.astype(np.float32)
            * (box_mask[..., None].astype(np.float32) / 255.0)
        ).astype(np.uint8)

        gt_np = None
        edge_np = None

        if self.training or self.return_gt:
            if not mask_path.is_file():
                raise FileNotFoundError(mask_path)

            with Image.open(mask_path) as gt_file:
                gt_np = np.asarray(gt_file.convert("L"), dtype=np.uint8)

            if gt_np.shape != image_np.shape[:2]:
                target_wh = (gt_np.shape[1], gt_np.shape[0])
                image_np = cv2.resize(
                    image_np,
                    target_wh,
                    interpolation=cv2.INTER_LINEAR,
                )
                box_image_np = cv2.resize(
                    box_image_np,
                    target_wh,
                    interpolation=cv2.INTER_LINEAR,
                )
                orig_h, orig_w = gt_np.shape

            if self.training:
                if self.edge_dir is not None:
                    edge_path = self.edge_dir / f"{name}{self.edge_suffix}"
                    edge_np = cv2.imread(str(edge_path), cv2.IMREAD_GRAYSCALE)
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

        out = {
            "image": self.image_transform(Image.fromarray(image_np)),
            "box_image": self.image_transform(Image.fromarray(box_image_np)),
            "name": name,
            "orig_h": int(orig_h),
            "orig_w": int(orig_w),
        }

        if gt_np is not None:
            out["gt"] = self.gt_transform(Image.fromarray(gt_np))

        if edge_np is not None:
            out["edge"] = self.gt_transform(Image.fromarray(edge_np))

        return out


# -----------------------------------------------------------------------------
# Model / loss / LR
# -----------------------------------------------------------------------------


def build_model(cfg, device, logger):
    model = NoisyCODANet(
        backbone_name=cfg["model"].get(
            "backbone_name",
            "convnext_base.fb_in22k_ft_in1k_384",
        ),
        channels=int(cfg["model"]["channels"]),
        ablation=str(cfg["model"].get("ablation", "A0")),
        router_hidden=int(cfg["model"].get("router_hidden", 32)),
        router_temperature=float(cfg["model"].get("router_temperature", 1.0)),
    ).to(device)

    params = sum(p.numel() for p in model.parameters())
    logger.info("ANet parameters: %.3f M", params / 1e6)
    logger.info("Backbone: %s", cfg["model"]["backbone_name"])
    logger.info("DWT ablation: %s", cfg["model"].get("ablation", "A0"))
    return model


def adjust_lr(
    now_epoch: int,
    top_epoch: int,
    max_epoch: int,
    init_lr: float,
    top_lr: float,
    min_lr: float,
    optimizer,
):
    del init_lr

    if now_epoch < top_epoch:
        lr = min_lr + abs(top_lr - min_lr) / top_epoch * now_epoch
    else:
        progress = (now_epoch - top_epoch) / max(max_epoch - top_epoch, 1)
        lr = min_lr + (top_lr - min_lr) * 0.5 * (
            1 + math.cos(math.pi * progress)
        )

    for group in optimizer.param_groups:
        group["lr"] = lr

    return lr


def compute_anet_loss(preds, gts, edges, step_idx, total_step, train_cfg):
    # Official behavior: UAL coefficient restarts every epoch.
    ual_coef = get_ual_coef(step_idx / float(max(total_step, 1)))
    ual_loss = cal_ual(preds[4], gts) * ual_coef

    loss_init = (
        structure_loss(preds[0], gts) * 0.0625
        + structure_loss(preds[1], gts) * 0.125
        + structure_loss(preds[2], gts) * 0.25
        + structure_loss(preds[3], gts) * 0.5
    )
    loss_final = structure_loss(preds[4], gts)

    loss_edge = (
        dice_loss(preds[6], edges) * 0.125
        + dice_loss(preds[7], edges) * 0.25
        + dice_loss(preds[8], edges) * 0.5
    )

    total = (
        loss_init
        + loss_final
        + float(train_cfg["edge_loss_weight"]) * loss_edge
        + float(train_cfg["ual_loss_weight"]) * ual_loss
    )

    return total, loss_init, loss_final, loss_edge, ual_loss, ual_coef


# -----------------------------------------------------------------------------
# Validation / generation
# -----------------------------------------------------------------------------


@torch.no_grad()
def validate_mae(model, loader, device):
    model.eval()
    maes = []

    for batch in tqdm(loader, desc="VAL", ncols=96):
        image = batch["image"].to(device, non_blocking=True)
        box_image = batch["box_image"].to(device, non_blocking=True)
        gt = batch["gt"].to(device, non_blocking=True)

        pred = torch.sigmoid(model(image, box_image)[4])
        mae = torch.mean(torch.abs(gt - pred), dim=(1, 2, 3))
        maes.extend(mae.detach().cpu().tolist())

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

    diagnostic_mae = []

    for batch in tqdm(loader, desc="GENERATE", ncols=96):
        image = batch["image"].to(device, non_blocking=True)
        box_image = batch["box_image"].to(device, non_blocking=True)

        outputs = model(image, box_image)
        masks = torch.sigmoid(outputs[4]).detach().cpu().numpy()[:, 0]
        edges = outputs[8].detach().cpu().numpy()[:, 0]

        if "gt" in batch:
            gt = batch["gt"]
            pred_t = torch.from_numpy(masks).unsqueeze(1)
            diagnostic_mae.extend(
                torch.mean(torch.abs(gt - pred_t), dim=(1, 2, 3)).tolist()
            )

        for i, name in enumerate(batch["name"]):
            h = int(batch["orig_h"][i])
            w = int(batch["orig_w"][i])

            mask = cv2.resize(
                masks[i],
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )
            edge = cv2.resize(
                edges[i],
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )

            cv2.imwrite(
                str(mask_dir / f"{name}.png"),
                np.clip(mask * 255.0, 0, 255).astype(np.uint8),
            )
            cv2.imwrite(
                str(edge_dir / f"{name}.png"),
                np.clip(edge * 255.0, 0, 255).astype(np.uint8),
            )

    if diagnostic_mae:
        return float(np.mean(diagnostic_mae))
    return None


# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------


def train(cfg, model, labeled, unlabeled, paths, device, logger):
    train_cfg = cfg["train"]

    # IMPORTANT: only the TRAIN dataset can override the supervision mask
    # directory. Validation still uses cfg["data"]["mask_dir"] = real GT.
    train_data_cfg = dict(cfg["data"])
    if train_cfg.get("target_mask_dir") not in (None, ""):
        train_data_cfg["mask_dir"] = train_cfg["target_mask_dir"]
        train_data_cfg["mask_suffix"] = train_cfg.get(
            "target_mask_suffix", train_data_cfg.get("mask_suffix", ".png")
        )

    train_set = ANetDataset(
        names=labeled,
        data_cfg=train_data_cfg,
        image_size=int(train_cfg["image_size"]),
        training=True,
        augment=bool(train_cfg["augment"]),
        strict_albumentations_v1=bool(
            train_cfg.get("strict_albumentations_v1", False)
        ),
        return_gt=True,
    )

    # The remaining split is not used for gradient updates.
    # GT is loaded only for validation/model selection, matching the released code.
    val_set = ANetDataset(
        names=unlabeled,
        data_cfg=cfg["data"],
        image_size=int(train_cfg["image_size"]),
        training=False,
        augment=False,
        return_gt=True,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=max(1, min(64, int(train_cfg["batch_size"]) * 4)),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=True,
    )

    if str(train_cfg["optimizer"]).lower() != "adam":
        raise ValueError("Faithful Noisy-COD ANet requires optimizer='adam'.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["init_lr"]),
    )

    amp_enabled = bool(train_cfg["amp"])
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    epochs = int(train_cfg["epochs"])
    total_step = len(train_loader)

    best_mae = float("inf")
    best_epoch = -1

    paths["ckpt_dir"].mkdir(parents=True, exist_ok=True)
    best_path = paths["ckpt_dir"] / "Net_epoch_best.pth"

    for epoch in range(1, epochs + 1):
        lr = adjust_lr(
            now_epoch=epoch,
            top_epoch=int(train_cfg["top_epoch"]),
            max_epoch=epochs,
            init_lr=float(train_cfg["init_lr"]),
            top_lr=float(train_cfg["top_lr"]),
            min_lr=float(train_cfg["min_lr"]),
            optimizer=optimizer,
        )

        model.train()
        running_loss = 0.0

        pbar = tqdm(
            enumerate(train_loader, start=1),
            total=total_step,
            desc=f"TRAIN {epoch:03d}/{epochs:03d}",
            ncols=120,
        )

        for step_idx, batch in pbar:
            images = batch["image"].to(device, non_blocking=True)
            box_images = batch["box_image"].to(device, non_blocking=True)
            gts = batch["gt"].to(device, non_blocking=True)
            edges = batch["edge"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                preds = model(images, box_images)
                (
                    loss,
                    loss_init,
                    loss_final,
                    loss_edge,
                    loss_ual,
                    ual_coef,
                ) = compute_anet_loss(
                    preds,
                    gts,
                    edges,
                    step_idx,
                    total_step,
                    train_cfg,
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.detach().item())

            pbar.set_postfix(
                lr=f"{lr:.2e}",
                loss=f"{loss.item():.4f}",
                init=f"{loss_init.item():.4f}",
                final=f"{loss_final.item():.4f}",
                edge=f"{loss_edge.item():.4f}",
                ual=f"{loss_ual.item():.4f}",
                uc=f"{ual_coef:.3f}",
            )

        logger.info(
            "Epoch %03d | lr=%.8f | mean_loss=%.6f",
            epoch,
            lr,
            running_loss / max(total_step, 1),
        )

        if epoch > epochs - int(train_cfg["save_last_epochs"]):
            torch.save(
                model.state_dict(),
                paths["ckpt_dir"] / f"Net_epoch_{epoch}.pth",
            )

        if epoch % int(train_cfg["validate_every"]) == 0:
            mae = validate_mae(model, val_loader, device)

            logger.info(
                "VAL epoch=%d | MAE=%.6f | best=%.6f @ epoch=%d",
                epoch,
                mae,
                best_mae,
                best_epoch,
            )

            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                torch.save(model.state_dict(), best_path)
                logger.info("Saved best ANet: %s", best_path)

    if not best_path.is_file():
        torch.save(model.state_dict(), best_path)

    logger.info(
        "Training complete | best MAE=%.6f @ epoch=%d",
        best_mae,
        best_epoch,
    )

    return best_path


# -----------------------------------------------------------------------------
# Checkpoint + generation
# -----------------------------------------------------------------------------


def load_checkpoint(model, path: Path, device):
    state = torch.load(path, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    cleaned = {}
    for key, value in state.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value

    model.load_state_dict(cleaned, strict=True)


def run_generate(
    cfg,
    model,
    unlabeled,
    paths,
    device,
    checkpoint,
    logger,
):
    load_checkpoint(model, Path(checkpoint), device)
    logger.info("Loaded checkpoint: %s", checkpoint)

    gen_cfg = cfg["generate"]

    gen_set = ANetDataset(
        names=unlabeled,
        data_cfg=cfg["data"],
        image_size=int(gen_cfg["image_size"]),
        training=False,
        augment=False,
        # GT is only used to report diagnostic MAE; it is never model input.
        return_gt=True,
    )

    gen_loader = DataLoader(
        gen_set,
        batch_size=int(gen_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(gen_cfg["num_workers"]),
        pin_memory=True,
    )

    mae = generate_pseudo_labels(
        model=model,
        loader=gen_loader,
        device=device,
        mask_dir=paths["pseudo_mask_dir"],
        edge_dir=paths["pseudo_edge_dir"],
    )

    logger.info("Pseudo masks: %s", paths["pseudo_mask_dir"])
    logger.info("Pseudo edges: %s", paths["pseudo_edge_dir"])

    if mae is not None:
        logger.info(
            "Diagnostic pseudo-label MAE: %.6f",
            mae,
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    args = parse_args()

    cfg = load_cfg(args.config)
    cfg = apply_cli_overrides(cfg, args)

    ratio, labeled_count, unlabeled_count = validate_ratio_cfg(cfg)

    # If user changed --ratio but did not specify --output-root, automatically
    # avoid writing F5/F10 into the default F20 directory.
    if args.ratio is not None and args.output_root is None:
        cfg["output"]["root"] = f"./ANet_outputs/NoisyCOD_ANet_F{ratio}"

    seed_everything(int(cfg["experiment"]["seed"]))

    paths = build_paths(cfg)
    paths["root"].mkdir(parents=True, exist_ok=True)
    logger = setup_logger(paths["log_file"])

    device = torch.device(
        args.device if torch.cuda.is_available() else "cpu"
    )

    logger.info("Device: %s", device)
    if cfg["train"].get("sample_list") in (None, ""):
        logger.info(
            "Experiment F%d | GT=%d | pseudo=%d | output=%s",
            ratio, labeled_count, unlabeled_count, paths["root"],
        )
    else:
        logger.info(
            "Experiment clean-pseudo TXT mode | list=%s | output=%s",
            cfg["train"]["sample_list"], paths["root"],
        )

    labeled, unlabeled, train_source = prepare_training_names(
        cfg, paths, logger
    )
    logger.info(
        "Training source=%s | train_samples=%d | remaining=%d",
        train_source, len(labeled), len(unlabeled)
    )

    model = build_model(cfg, device, logger)

    best_path = None

    if args.mode in ("train", "all"):
        best_path = train(
            cfg,
            model,
            labeled,
            unlabeled,
            paths,
            device,
            logger,
        )

    if args.mode in ("generate", "all"):
        checkpoint = args.checkpoint or best_path

        if checkpoint is None:
            checkpoint = paths["ckpt_dir"] / "Net_epoch_best.pth"

        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"Generation checkpoint not found: {checkpoint}"
            )

        run_generate(
            cfg,
            model,
            unlabeled,
            paths,
            device,
            checkpoint,
            logger,
        )


if __name__ == "__main__":
    main()
