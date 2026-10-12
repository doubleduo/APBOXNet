#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Independent trainer / pseudo-label generator for ConvNeXtB_ZoomNeXt_ANet.

Training:
    RGB + BoxPrompt + dense GT -> ANet

Pseudo-label generation:
    RGB + BoxPrompt -> dense pseudo mask + pseudo edge

This file intentionally does NOT use basemain.train because the current
basemain.py is coupled to the B2 curriculum/EMA trainer.

Examples
--------
Train ANet:
    python anet_main.py \
        --config configs/anet_convnextb_zoomnext.py \
        --data-cfg dataset.yaml \
        --pretrained

Generate pseudo labels:
    python anet_main.py \
        --config configs/anet_convnextb_zoomnext.py \
        --data-cfg dataset.yaml \
        --load-from ANet_outputs/.../pth/state_final.pth \
        --generate
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import math
import os
import random
import shutil
import time
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import albumentations as A
import colorlog
import cv2
import numpy as np
import torch
import yaml
from mmengine import Config
from torch.utils import data
from tqdm import tqdm
from methods.zoomnext.anet_zoomnext import build_anet_or_apnet

from utils import io, ops, pt_utils, py_utils, recorder


LOGGER = logging.getLogger("main")
LOGGER.propagate = False
LOGGER.setLevel(logging.DEBUG)
if not LOGGER.handlers:
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(
        colorlog.ColoredFormatter(
            "%(log_color)s[%(filename)s] %(reset)s%(message)s"
        )
    )
    LOGGER.addHandler(stream_handler)


def _stem_names(path: str, suffix: str) -> List[str]:
    if not os.path.isdir(path):
        raise FileNotFoundError(path)
    return sorted(
        p[: -len(suffix)]
        for p in os.listdir(path)
        if p.endswith(suffix)
    )


def _resolve_component(dataset_info: dict, key: str):
    item = dataset_info[key]
    return (
        os.path.join(dataset_info["root"], item["path"]),
        item["suffix"],
    )


def _parse_labelme_box_json(path: str, h: int, w: int) -> np.ndarray:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    box_mask = np.zeros((h, w), dtype=np.uint8)
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
            box_mask[y1:y2, x1:x2] = 1
    return box_mask


def _parse_txt_boxes(
    path: str,
    h: int,
    w: int,
    box_format: str = "xyxy",
) -> np.ndarray:
    """
    Supported line formats:
      xyxy : x1 y1 x2 y2
      xywh : x y width height
      yolo : cx cy width height (normalized 0..1)

    Extra tokens are tolerated. If a line has >4 numeric values, the LAST
    four are used, which supports common "class x1 y1 x2 y2" files.
    """
    box_mask = np.zeros((h, w), dtype=np.uint8)

    with open(path, "r", encoding="utf-8-sig") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip().replace(",", " ")
        if not line:
            continue

        vals = []
        for token in line.split():
            try:
                vals.append(float(token))
            except ValueError:
                pass
        if len(vals) < 4:
            continue
        a, b, c, d = vals[-4:]

        fmt = box_format.lower()
        if fmt == "xyxy":
            x1, y1, x2, y2 = a, b, c, d
        elif fmt == "xywh":
            x1, y1 = a, b
            x2, y2 = a + c, b + d
        elif fmt == "yolo":
            cx, cy, bw, bh = a, b, c, d
            x1 = (cx - bw / 2.0) * w
            y1 = (cy - bh / 2.0) * h
            x2 = (cx + bw / 2.0) * w
            y2 = (cy + bh / 2.0) * h
        else:
            raise ValueError(f"Unsupported box_format={box_format!r}")

        x1 = max(0, min(w, int(math.floor(x1))))
        y1 = max(0, min(h, int(math.floor(y1))))
        x2 = max(0, min(w, int(math.ceil(x2))))
        y2 = max(0, min(h, int(math.ceil(y2))))

        if x2 > x1 and y2 > y1:
            box_mask[y1:y2, x1:x2] = 1

    return box_mask


def _load_box_mask(
    box_path: str,
    h: int,
    w: int,
    box_format: str,
) -> np.ndarray:
    suffix = Path(box_path).suffix.lower()

    if suffix == ".json":
        return _parse_labelme_box_json(box_path, h=h, w=w)

    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}:
        box = cv2.imread(box_path, cv2.IMREAD_GRAYSCALE)
        if box is None:
            raise FileNotFoundError(box_path)
        if box.shape != (h, w):
            box = cv2.resize(box, (w, h), interpolation=cv2.INTER_NEAREST)
        return (box > 0).astype(np.uint8)

    return _parse_txt_boxes(
        box_path,
        h=h,
        w=w,
        box_format=box_format,
    )


class ANetTrainDataset(data.Dataset):
    """Image + dense mask + box prompt for the fully annotated ANet subset."""

    def __init__(
        self,
        dataset_infos: Dict[str, dict],
        shape: dict,
        sample_ratio: float = 1.0,
        seed: int = 112358,
        augment: bool = True,
        box_format: str = "xyxy",
        target_key: str = "mask",
        clean_list: str | None = None,
        soft_target: bool = False,
    ):
        super().__init__()
        self.shape = shape
        self.box_format = str(box_format)
        self.target_key = str(target_key)
        self.soft_target = bool(soft_target)
        self.total_data_paths = []

        clean_names = None
        if clean_list:
            clean_path = Path(clean_list)
            if not clean_path.is_absolute():
                clean_path = Path.cwd() / clean_path
            if not clean_path.exists():
                raise FileNotFoundError(f"clean_list not found: {clean_path}")
            clean_names = set()
            with open(clean_path, "r", encoding="utf-8-sig") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    token = line.split()[0].split(",")[0]
                    clean_names.add(Path(token).stem)
            LOGGER.info("Loaded clean pseudo list: %s (%d)", clean_path, len(clean_names))

        for dataset_name, info in dataset_infos.items():
            image_path, image_suffix = _resolve_component(info, "image")
            if self.target_key not in info:
                raise KeyError(
                    f"{dataset_name} is missing target key {self.target_key!r}. "
                    "Add it to dataset yaml, e.g. pseudo_mask: {path: ..., suffix: .png}"
                )
            mask_path, mask_suffix = _resolve_component(info, self.target_key)

            if "box" in info:
                box_path, box_suffix = _resolve_component(info, "box")
            elif "box_json" in info:
                box_path, box_suffix = _resolve_component(info, "box_json")
            else:
                raise KeyError(
                    f"{dataset_name} requires 'box' or 'box_json' in data yaml"
                )

            valid = sorted(
                set(_stem_names(image_path, image_suffix))
                & set(_stem_names(mask_path, mask_suffix))
                & set(_stem_names(box_path, box_suffix))
            )
            if clean_names is not None:
                valid = [n for n in valid if Path(n).stem in clean_names]

            paths = [
                (
                    os.path.join(image_path, n) + image_suffix,
                    os.path.join(mask_path, n) + mask_suffix,
                    os.path.join(box_path, n) + box_suffix,
                    n,
                )
                for n in valid
            ]
            LOGGER.info(
                "ANet dataset %s: %d image/mask/box triplets",
                dataset_name,
                len(paths),
            )
            self.total_data_paths.extend(paths)

        ratio = float(sample_ratio)
        if not (0.0 < ratio <= 1.0):
            raise ValueError("train.sample_ratio must be in (0, 1].")

        if ratio < 1.0:
            rng = random.Random(int(seed))
            rng.shuffle(self.total_data_paths)
            keep = max(1, int(round(len(self.total_data_paths) * ratio)))
            self.total_data_paths = self.total_data_paths[:keep]
            LOGGER.info(
                "ANet full-mask subset: ratio=%.4f, samples=%d",
                ratio,
                keep,
            )

        self.augment = bool(augment)
        self.transforms = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.Rotate(
                    limit=30,
                    p=0.35,
                    interpolation=cv2.INTER_LINEAR,
                    border_mode=cv2.BORDER_REPLICATE,
                ),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.5,
                ),
                A.HueSaturationValue(
                    hue_shift_limit=5,
                    sat_shift_limit=10,
                    val_shift_limit=10,
                    p=0.4,
                ),
            ],
            additional_targets={"box_mask": "mask"},
        )

    def __len__(self):
        return len(self.total_data_paths)

    def __getitem__(self, index):
        image_path, mask_path, box_path, name = self.total_data_paths[index]

        image = io.read_color_array(image_path)
        if self.soft_target:
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(mask_path)
            mask = mask.astype(np.float32) / 255.0
        else:
            mask = io.read_gray_array(mask_path, thr=0).astype(np.float32)

        ih, iw = image.shape[:2]
        box_mask = _load_box_mask(
            box_path,
            h=ih,
            w=iw,
            box_format=self.box_format,
        ).astype(np.float32)

        if image.shape[:2] != mask.shape:
            mh, mw = mask.shape
            image = ops.resize(image, height=mh, width=mw)
            box_mask = cv2.resize(
                box_mask,
                (mw, mh),
                interpolation=cv2.INTER_NEAREST,
            )

        if self.augment:
            transformed = self.transforms(
                image=image,
                mask=mask,
                box_mask=box_mask,
            )
            image = transformed["image"]
            mask = transformed["mask"]
            box_mask = transformed["box_mask"]

        h = int(self.shape["h"])
        w = int(self.shape["w"])
        image = ops.resize(image, height=h, width=w)
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        box_mask = cv2.resize(
            box_mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

        image_m = torch.from_numpy(image).div(255).permute(2, 0, 1).float()
        if self.soft_target:
            mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        else:
            mask_t = torch.from_numpy((mask > 0).astype(np.float32)).unsqueeze(0)
        box_t = torch.from_numpy((box_mask > 0).astype(np.float32)).unsqueeze(0)

        return {
            "data": {
                "image_m": image_m,
                "mask": mask_t,
                "box_mask": box_t,
            },
            "info": {"name": name},
        }



class APNetWeakDataset(data.Dataset):
    """Box-only subset used by APNet V1.1 for online teacher->student distillation."""
    def __init__(self, dataset_infos, shape, exclude_names, augment=True, box_format="xyxy"):
        super().__init__()
        self.shape = shape
        self.box_format = str(box_format)
        self.total_data_paths = []
        exclude_names = set(exclude_names)
        for dataset_name, info in dataset_infos.items():
            image_path, image_suffix = _resolve_component(info, "image")
            if "box" in info:
                box_path, box_suffix = _resolve_component(info, "box")
            elif "box_json" in info:
                box_path, box_suffix = _resolve_component(info, "box_json")
            else:
                raise KeyError(f"{dataset_name} requires 'box' or 'box_json'")
            valid = sorted(set(_stem_names(image_path, image_suffix)) & set(_stem_names(box_path, box_suffix)))
            valid = [n for n in valid if Path(n).stem not in exclude_names]
            self.total_data_paths.extend([
                (os.path.join(image_path, n) + image_suffix,
                 os.path.join(box_path, n) + box_suffix, n)
                for n in valid
            ])
        self.augment = bool(augment)
        self.transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            # V1.1 deliberately removes rotation: box-only online targets are more stable this way.
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.5),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=10, val_shift_limit=10, p=0.4),
        ], additional_targets={"box_mask": "mask"})

    def __len__(self):
        return len(self.total_data_paths)

    def __getitem__(self, index):
        image_path, box_path, name = self.total_data_paths[index]
        image = io.read_color_array(image_path)
        ih, iw = image.shape[:2]
        box_mask = _load_box_mask(box_path, h=ih, w=iw, box_format=self.box_format).astype(np.float32)
        if self.augment:
            t = self.transforms(image=image, box_mask=box_mask)
            image, box_mask = t["image"], t["box_mask"]
        h, w = int(self.shape["h"]), int(self.shape["w"])
        image = ops.resize(image, height=h, width=w)
        box_mask = cv2.resize(box_mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return {
            "data": {
                "image_m": torch.from_numpy(image).div(255).permute(2, 0, 1).float(),
                "box_mask": torch.from_numpy((box_mask > 0).astype(np.float32)).unsqueeze(0),
            },
            "info": {"name": name},
        }


class ANetPromptDataset(data.Dataset):
    """Image + box only, used to generate dense pseudo labels."""

    def __init__(
        self,
        dataset_infos: Dict[str, dict],
        shape: dict,
        box_format: str = "xyxy",
    ):
        super().__init__()
        self.shape = shape
        self.box_format = str(box_format)
        self.total_data_paths = []

        for dataset_name, info in dataset_infos.items():
            image_path, image_suffix = _resolve_component(info, "image")
            if "box" in info:
                box_path, box_suffix = _resolve_component(info, "box")
            elif "box_json" in info:
                box_path, box_suffix = _resolve_component(info, "box_json")
            else:
                raise KeyError(
                    f"{dataset_name} requires 'box' or 'box_json'"
                )

            valid = sorted(
                set(_stem_names(image_path, image_suffix))
                & set(_stem_names(box_path, box_suffix))
            )

            self.total_data_paths.extend(
                [
                    (
                        os.path.join(image_path, n) + image_suffix,
                        os.path.join(box_path, n) + box_suffix,
                        n,
                    )
                    for n in valid
                ]
            )
            LOGGER.info(
                "ANet generation dataset %s: %d image/box pairs",
                dataset_name,
                len(valid),
            )

    def __len__(self):
        return len(self.total_data_paths)

    def __getitem__(self, index):
        image_path, box_path, name = self.total_data_paths[index]
        image = io.read_color_array(image_path)
        ih, iw = image.shape[:2]
        box_mask = _load_box_mask(
            box_path,
            h=ih,
            w=iw,
            box_format=self.box_format,
        ).astype(np.float32)

        h = int(self.shape["h"])
        w = int(self.shape["w"])
        image_resized = ops.resize(image, height=h, width=w)
        box_resized = cv2.resize(
            box_mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

        return {
            "data": {
                "image_m": torch.from_numpy(image_resized)
                .div(255)
                .permute(2, 0, 1)
                .float(),
                "box_mask": torch.from_numpy(
                    (box_resized > 0).astype(np.float32)
                ).unsqueeze(0),
            },
            "info": {
                "name": name,
                "orig_h": ih,
                "orig_w": iw,
            },
        }

class ANetValDataset(data.Dataset):
    """
    Validation dataset for ANet.

    Standard COD test sets normally provide RGB + GT mask but no box annotation.
    For validation ONLY, bounding-box prompts are deterministically derived from
    the GT mask by connected components. The GT mask is never passed into the
    model; it is used only to construct the validation box prompt and to compute
    metrics after prediction.
    """

    def __init__(
        self,
        dataset_info: dict,
        shape: dict,
        min_component_area: int = 4,
    ):
        super().__init__()
        self.shape = shape
        self.min_component_area = int(min_component_area)

        image_path, image_suffix = _resolve_component(dataset_info, "image")
        mask_path, mask_suffix = _resolve_component(dataset_info, "mask")

        valid = sorted(
            set(_stem_names(image_path, image_suffix))
            & set(_stem_names(mask_path, mask_suffix))
        )
        self.total_data_paths = [
            (
                os.path.join(image_path, n) + image_suffix,
                os.path.join(mask_path, n) + mask_suffix,
                n,
            )
            for n in valid
        ]

    def __len__(self):
        return len(self.total_data_paths)

    def _mask_to_component_boxes(self, mask: np.ndarray) -> np.ndarray:
        binary = (mask > 0).astype(np.uint8)
        h, w = binary.shape
        box_mask = np.zeros((h, w), dtype=np.uint8)

        num_labels, _, stats, _ = cv2.connectedComponentsWithStats(
            binary,
            connectivity=8,
        )

        for label_id in range(1, num_labels):
            x, y, bw, bh, area = stats[label_id]
            if int(area) < self.min_component_area:
                continue
            x2 = min(int(x + bw), w)
            y2 = min(int(y + bh), h)
            box_mask[int(y):y2, int(x):x2] = 1

        # Safety fallback for a non-empty GT whose tiny components were filtered.
        if box_mask.max() == 0 and binary.max() > 0:
            ys, xs = np.where(binary > 0)
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            box_mask[y1:y2, x1:x2] = 1

        return box_mask

    def __getitem__(self, index):
        image_path, mask_path, name = self.total_data_paths[index]
        image = io.read_color_array(image_path)

        mask_u8 = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask_u8 is None:
            raise FileNotFoundError(mask_path)
        mask_u8 = (mask_u8 > 0).astype(np.uint8)

        if image.shape[:2] != mask_u8.shape:
            h0, w0 = mask_u8.shape
            image = ops.resize(image, height=h0, width=w0)

        box_mask = self._mask_to_component_boxes(mask_u8)

        h = int(self.shape["h"])
        w = int(self.shape["w"])

        image_resized = ops.resize(image, height=h, width=w)
        box_resized = cv2.resize(
            box_mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

        return {
            "data": {
                "image_m": torch.from_numpy(image_resized)
                .div(255)
                .permute(2, 0, 1)
                .float(),
                "box_mask": torch.from_numpy(
                    (box_resized > 0).astype(np.float32)
                ).unsqueeze(0),
            },
            "info": {
                "name": name,
                "mask_path": mask_path,
            },
        }


def _append_val_csv(csv_path: str, epoch: int, dataset_name: str, metrics: dict):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.isfile(csv_path)

    def pick(*keys, default=float("nan")):
        for key in keys:
            if key in metrics:
                return float(metrics[key])
        return float(default)

    row = [
        int(epoch),
        str(dataset_name),
        pick("S", "sm"),
        pick("Fw", "wfm", "wFmeasure"),
        pick("maxem", "E", "em"),
        pick("MAE", "mae"),
    ]

    import csv
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(["epoch", "dataset", "S", "Fw", "E", "MAE"])
        writer.writerow(row)


@torch.no_grad()
def validate_anet(model, cfg, epoch: int):
    """
    Evaluate ANet with RGB + GT-derived box prompts on COD validation sets.

    Returns:
        results_by_dataset, average_mae
    """
    model.eval()
    results = {}
    save_vis_n = int(cfg.val.get("save_vis_n", 12))

    for dataset_name in cfg.val.data.names:
        dataset_info = cfg.dataset_infos[dataset_name]
        dataset = ANetValDataset(
            dataset_info=dataset_info,
            shape=cfg.val.data.shape,
            min_component_area=int(cfg.val.get("min_component_area", 4)),
        )
        loader = data.DataLoader(
            dataset,
            batch_size=int(cfg.val.batch_size),
            num_workers=int(cfg.val.num_workers),
            shuffle=False,
            pin_memory=True,
        )

        metric_recorder = recorder.GroupedMetricRecorder(
            metric_names=list(cfg.metric_names)
        )

        vis_dir = os.path.join(
            cfg.path.pth_log,
            "val_pred",
            f"epoch_{int(epoch):03d}",
            dataset_name,
        )
        if save_vis_n > 0:
            os.makedirs(vis_dir, exist_ok=True)

        saved = 0
        for batch in tqdm(
            loader,
            desc=f"[ANet VAL {dataset_name}]",
            ncols=92,
        ):
            batch_data = pt_utils.to_device(
                batch["data"],
                device=cfg.device,
            )

            logits = model(data=batch_data)
            probs = torch.sigmoid(logits)

            # Keep evaluation consistent with the repository's COD evaluator:
            # per-image min-max normalization before metric computation.
            pmin = probs.amin(dim=(2, 3), keepdim=True)
            pmax = probs.amax(dim=(2, 3), keepdim=True)
            probs = (probs - pmin) / (pmax - pmin + 1e-8)
            probs_np = probs[:, 0].detach().cpu().numpy()

            mask_paths = batch["info"]["mask_path"]
            names = batch["info"]["name"]

            for i, pred in enumerate(probs_np):
                mask_path = mask_paths[i]
                gt = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                if gt is None:
                    raise FileNotFoundError(mask_path)
                gt[gt > 0] = 255
                gh, gw = gt.shape

                pred = cv2.resize(
                    pred,
                    (gw, gh),
                    interpolation=cv2.INTER_LINEAR,
                )
                pred_u8 = np.clip(pred * 255.0, 0, 255).astype(np.uint8)

                metric_recorder.step(
                    group_name="image",
                    pre=pred_u8,
                    gt=gt,
                    gt_path=mask_path,
                )

                if saved < save_vis_n:
                    cv2.imwrite(
                        os.path.join(vis_dir, f"{names[i]}.png"),
                        pred_u8,
                    )
                    saved += 1

        metrics = metric_recorder.show()
        results[dataset_name] = metrics

        metric_str = ", ".join(
            f"{k}:{float(v):.4f}" for k, v in metrics.items()
        )
        LOGGER.info(
            "ANet VAL | Epoch %03d | %s | %s",
            int(epoch),
            dataset_name,
            metric_str,
        )

        _append_val_csv(
            csv_path=os.path.join(cfg.path.pth_log, "anet_val_metrics.csv"),
            epoch=int(epoch),
            dataset_name=dataset_name,
            metrics=metrics,
        )

    maes = []
    for metrics in results.values():
        for key in ("MAE", "mae"):
            if key in metrics:
                maes.append(float(metrics[key]))
                break

    avg_mae = float(np.mean(maes)) if maes else float("inf")
    LOGGER.info(
        "ANet VAL | Epoch %03d | AVG_MAE: %.6f",
        int(epoch),
        avg_mae,
    )
    return results, avg_mae



def _set_lr(optimizer, base_lrs: Sequence[float], factor: float):
    for group, base_lr in zip(optimizer.param_groups, base_lrs):
        group["lr"] = float(base_lr) * float(factor)


def _lr_factor(
    step: int,
    total_steps: int,
    warmup_steps: int,
    min_ratio: float,
):
    if warmup_steps > 0 and step < warmup_steps:
        return max((step + 1) / float(warmup_steps), 1e-3)

    denom = max(total_steps - warmup_steps, 1)
    p = (step - warmup_steps) / float(denom)
    p = min(max(p, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * p))
    return float(min_ratio + (1.0 - min_ratio) * cosine)


def _make_optimizer(model, cfg):
    grouped = model.get_grouped_params()
    lr = float(cfg.train.lr)
    backbone_factor = float(cfg.train.get("backbone_lr_factor", 0.1))
    weight_decay = float(cfg.train.get("weight_decay", 1e-4))

    param_groups = [
        {
            "params": grouped["pretrained"],
            "lr": lr * backbone_factor,
            "weight_decay": weight_decay,
        },
        {
            "params": grouped["retrained"],
            "lr": lr,
            "weight_decay": weight_decay,
        },
    ]
    param_groups = [g for g in param_groups if g["params"]]

    mode = str(cfg.train.get("optimizer", "adamw")).lower()
    if mode == "adam":
        return torch.optim.Adam(param_groups)
    if mode == "sgd":
        return torch.optim.SGD(
            param_groups,
            momentum=0.9,
            nesterov=True,
        )
    return torch.optim.AdamW(param_groups)


def train_anet_original(model, cfg):
    dataset_infos = {
        name: cfg.dataset_infos[name]
        for name in cfg.train.data.names
    }
    dataset = ANetTrainDataset(
        dataset_infos=dataset_infos,
        shape=cfg.train.data.shape,
        sample_ratio=float(cfg.train.get("sample_ratio", 1.0)),
        seed=int(cfg.base_seed),
        augment=bool(cfg.train.get("augment", True)),
        box_format=str(cfg.train.get("box_format", "xyxy")),
        target_key=str(cfg.train.get("target_key", "mask")),
        clean_list=cfg.train.get("clean_list", None),
        soft_target=bool(cfg.train.get("soft_target", False)),
    )

    loader = data.DataLoader(
        dataset,
        batch_size=int(cfg.train.batch_size),
        num_workers=int(cfg.train.num_workers),
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        worker_init_fn=(
            pt_utils.customized_worker_init_fn
            if bool(cfg.use_custom_worker_init)
            else None
        ),
    )
    if len(loader) == 0:
        raise RuntimeError(
            "ANet training DataLoader has zero batches. "
            "Reduce batch_size or check image/mask/box intersections."
        )

    optimizer = _make_optimizer(model, cfg)
    base_lrs = [float(g["lr"]) for g in optimizer.param_groups]

    epochs = int(cfg.train.num_epochs)
    grad_acc = max(1, int(cfg.train.get("grad_acc_step", 1)))
    total_steps = epochs * len(loader)
    warmup_steps = int(cfg.train.get("warmup_steps", len(loader)))
    min_lr_ratio = float(cfg.train.get("min_lr_ratio", 0.01))

    amp_enabled = bool(cfg.train.use_amp) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    val_enabled = bool(cfg.get("val", {}).get("enable", True))
    val_start = int(cfg.val.get("start_epoch", 5)) if val_enabled else 10**9
    val_interval = int(cfg.val.get("interval", 5)) if val_enabled else 10**9

    best_avg_mae = float("inf")
    best_per_dataset = {}

    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    train_start = time.perf_counter()

    LOGGER.info(
        "ANet train: samples=%d, batches/epoch=%d, epochs=%d, total_steps=%d",
        len(dataset),
        len(loader),
        epochs,
        total_steps,
    )
    if val_enabled:
        LOGGER.info(
            "ANet validation enabled: start=%d, interval=%d, datasets=%s",
            val_start,
            val_interval,
            list(cfg.val.data.names),
        )

    for epoch in range(1, epochs + 1):
        model.train()
        sums = {
            "total": 0.0,
            "structure": 0.0,
            "edge": 0.0,
            "ual": 0.0,
        }
        epoch_start = time.perf_counter()

        for batch_idx, batch in enumerate(loader, start=1):
            factor = _lr_factor(
                global_step,
                total_steps,
                warmup_steps,
                min_lr_ratio,
            )
            _set_lr(optimizer, base_lrs, factor)

            batch_data = pt_utils.to_device(
                batch["data"],
                device=cfg.device,
            )
            progress = global_step / max(total_steps - 1, 1)

            with torch.cuda.amp.autocast(enabled=amp_enabled):
                outputs = model(
                    data=batch_data,
                    iter_percentage=progress,
                )
                raw_loss = outputs["loss"]
                loss = raw_loss / grad_acc

            scaler.scale(loss).backward()

            if batch_idx % grad_acc == 0 or batch_idx == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            items = outputs["loss_items"]
            sums["total"] += float(raw_loss.detach().item())
            for key in ("structure", "edge", "ual"):
                sums[key] += float(items[key].detach().item())

            if (
                batch_idx == 1
                or batch_idx % int(cfg.log_interval) == 0
                or batch_idx == len(loader)
            ):
                elapsed = max(time.perf_counter() - train_start, 1e-6)
                eta_sec = elapsed / max(global_step + 1, 1) * max(
                    total_steps - global_step - 1,
                    0,
                )
                lrs = ",".join(
                    f"{g['lr']:.2e}" for g in optimizer.param_groups
                )
                LOGGER.info(
                    "ANet | ETA:%s | E%03d/%03d B%04d/%04d | LR:%s | %s",
                    str(datetime.timedelta(seconds=int(eta_sec))),
                    epoch,
                    epochs,
                    batch_idx,
                    len(loader),
                    lrs,
                    outputs["loss_str"],
                )

            if global_step < 3:
                recorder.plot_results(
                    {
                        "img": batch_data["image_m"],
                        "msk": batch_data["mask"],
                        **outputs["vis"],
                    },
                    save_path=os.path.join(
                        cfg.path.pth_log,
                        "img",
                        f"iter_{global_step}.png",
                    ),
                )

            global_step += 1

        n = max(len(loader), 1)
        LOGGER.info(
            "ANet Epoch %03d | %.1fs | L:%.4f STR:%.4f EDGE:%.4f UAL:%.4f",
            epoch,
            time.perf_counter() - epoch_start,
            sums["total"] / n,
            sums["structure"] / n,
            sums["edge"] / n,
            sums["ual"] / n,
        )

        # Always keep the latest weight.
        io.save_weight(
            model=model,
            save_path=cfg.path.final_state_net,
        )

        save_interval = int(cfg.train.get("save_interval", 10))
        if save_interval > 0 and epoch % save_interval == 0:
            io.save_weight(
                model=model,
                save_path=os.path.join(
                    cfg.path.pth,
                    f"state_epoch_{epoch:03d}.pth",
                ),
            )

        # ---- ANet validation ----
        should_validate = (
            val_enabled
            and epoch >= val_start
            and (epoch - val_start) % max(val_interval, 1) == 0
        )

        if should_validate:
            results, avg_mae = validate_anet(
                model=model,
                cfg=cfg,
                epoch=epoch,
            )

            # Save a global best checkpoint by average validation MAE.
            if avg_mae < best_avg_mae:
                best_avg_mae = avg_mae
                best_path = os.path.join(
                    cfg.path.pth,
                    "best_avg_mae.pth",
                )
                io.save_weight(model=model, save_path=best_path)
                LOGGER.info(
                    "ANet BEST AVG MAE -> %.6f @ epoch %d | %s",
                    best_avg_mae,
                    epoch,
                    best_path,
                )

            # Also save one best checkpoint for every validation dataset.
            for dataset_name, metrics in results.items():
                mae = None
                for key in ("MAE", "mae"):
                    if key in metrics:
                        mae = float(metrics[key])
                        break
                if mae is None:
                    continue

                old = best_per_dataset.get(dataset_name, float("inf"))
                if mae < old:
                    best_per_dataset[dataset_name] = mae
                    best_path = os.path.join(
                        cfg.path.pth,
                        f"best_{dataset_name}_mae.pth",
                    )
                    io.save_weight(model=model, save_path=best_path)
                    LOGGER.info(
                        "ANet BEST %s MAE -> %.6f @ epoch %d | %s",
                        dataset_name,
                        mae,
                        epoch,
                        best_path,
                    )

    LOGGER.info(
        "ANet training finished in %s | best_avg_mae=%.6f",
        str(
            datetime.timedelta(
                seconds=int(time.perf_counter() - train_start)
            )
        ),
        best_avg_mae,
    )


def _cycle_next(loader, iterator):
    try:
        return next(iterator), iterator
    except StopIteration:
        iterator = iter(loader)
        return next(iterator), iterator


def train_apnet_joint(model, cfg):
    """APNet V1.1: 10% GT+box + remaining 90% box-only in ONE optimization run."""
    infos = {name: cfg.dataset_infos[name] for name in cfg.train.data.names}
    fully = ANetTrainDataset(
        dataset_infos=infos,
        shape=cfg.train.data.shape,
        sample_ratio=float(cfg.train.get("sample_ratio", 0.10)),
        seed=int(cfg.base_seed), augment=bool(cfg.train.get("augment", True)),
        box_format=str(cfg.train.get("box_format", "xyxy")),
        target_key=str(cfg.train.get("target_key", "mask")),
        clean_list=cfg.train.get("clean_list", None), soft_target=False,
    )
    fully_names = {Path(x[3]).stem for x in fully.total_data_paths}
    weak = APNetWeakDataset(
        dataset_infos=infos, shape=cfg.train.data.shape, exclude_names=fully_names,
        augment=bool(cfg.train.get("augment", True)),
        box_format=str(cfg.train.get("box_format", "xyxy")),
    )
    fbs = int(cfg.train.get("fully_batch_size", 2))
    wbs = int(cfg.train.get("weak_batch_size", 6))
    common = dict(num_workers=int(cfg.train.num_workers), drop_last=True, pin_memory=True,
                  worker_init_fn=(pt_utils.customized_worker_init_fn if bool(cfg.use_custom_worker_init) else None))
    fully_loader = data.DataLoader(fully, batch_size=fbs, shuffle=True, **common)
    weak_loader = data.DataLoader(weak, batch_size=wbs, shuffle=True, **common)
    if len(fully_loader) == 0 or len(weak_loader) == 0:
        raise RuntimeError(f"APNet loader empty: fully={len(fully_loader)}, weak={len(weak_loader)}")

    optimizer = _make_optimizer(model, cfg)
    base_lrs = [float(g["lr"]) for g in optimizer.param_groups]
    epochs = int(cfg.train.num_epochs)
    steps_per_epoch = len(weak_loader)  # cover the 90% weak pool every epoch
    total_steps = epochs * steps_per_epoch
    warmup_steps = int(cfg.train.get("warmup_steps", 200))
    min_lr_ratio = float(cfg.train.get("min_lr_ratio", 0.05))
    weak_loss_weight = float(cfg.train.get("joint_weak_loss_weight", 1.0))
    grad_acc = max(1, int(cfg.train.get("grad_acc_step", 1)))
    amp_enabled = bool(cfg.train.use_amp) and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    val_enabled = bool(cfg.get("val", {}).get("enable", True))
    val_start = int(cfg.val.get("start_epoch", 10)) if val_enabled else 10**9
    val_interval = int(cfg.val.get("interval", 5)) if val_enabled else 10**9
    best_avg_mae, best_per_dataset = float("inf"), {}
    global_step = 0
    train_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)

    LOGGER.info("APNet V1.1 joint train: fully=%d weak=%d | batch=%d+%d | steps/epoch=%d epochs=%d total_steps=%d",
                len(fully), len(weak), fbs, wbs, steps_per_epoch, epochs, total_steps)

    for epoch in range(1, epochs + 1):
        model.train()
        f_iter = iter(fully_loader)
        sums = {"total":0.0, "full":0.0, "weak":0.0, "wkd":0.0, "out":0.0}
        epoch_start = time.perf_counter()
        for batch_idx, weak_batch in enumerate(weak_loader, start=1):
            full_batch, f_iter = _cycle_next(fully_loader, f_iter)
            factor = _lr_factor(global_step, total_steps, warmup_steps, min_lr_ratio)
            _set_lr(optimizer, base_lrs, factor)
            full_data = pt_utils.to_device(full_batch["data"], device=cfg.device)
            weak_data = pt_utils.to_device(weak_batch["data"], device=cfg.device)
            progress = global_step / max(total_steps - 1, 1)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                out_f = model(data=full_data, iter_percentage=progress)
                out_w = model(data=weak_data, iter_percentage=progress)
                raw_loss = out_f["loss"] + weak_loss_weight * out_w["loss"]
                loss = raw_loss / grad_acc
            scaler.scale(loss).backward()
            if batch_idx % grad_acc == 0 or batch_idx == steps_per_epoch:
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)

            sums["total"] += float(raw_loss.detach())
            sums["full"] += float(out_f["loss"].detach())
            sums["weak"] += float(out_w["loss"].detach())
            sums["wkd"] += float(out_w["loss_items"].get("weak_kd", torch.tensor(0.)).detach())
            sums["out"] += float(out_w["loss_items"].get("outside", torch.tensor(0.)).detach())

            if batch_idx == 1 or batch_idx % int(cfg.log_interval) == 0 or batch_idx == steps_per_epoch:
                lrs = ",".join(f"{g['lr']:.2e}" for g in optimizer.param_groups)
                LOGGER.info("APNet | E%03d/%03d B%04d/%04d | LR:%s | %s | %s",
                            epoch, epochs, batch_idx, steps_per_epoch, lrs, out_f["loss_str"], out_w["loss_str"])
            global_step += 1

        n = max(steps_per_epoch, 1)
        LOGGER.info("APNet Epoch %03d | %.1fs | L:%.4f FULL:%.4f WEAK:%.4f WKD:%.4f OUT:%.4f",
                    epoch, time.perf_counter()-epoch_start, sums["total"]/n, sums["full"]/n,
                    sums["weak"]/n, sums["wkd"]/n, sums["out"]/n)
        io.save_weight(model=model, save_path=cfg.path.final_state_net)
        save_interval = int(cfg.train.get("save_interval", 10))
        if save_interval > 0 and epoch % save_interval == 0:
            io.save_weight(model=model, save_path=os.path.join(cfg.path.pth, f"state_epoch_{epoch:03d}.pth"))

        if val_enabled and epoch >= val_start and (epoch-val_start) % max(val_interval,1) == 0:
            results, avg_mae = validate_anet(model=model, cfg=cfg, epoch=epoch)
            if avg_mae < best_avg_mae:
                best_avg_mae = avg_mae
                bp = os.path.join(cfg.path.pth, "best_avg_mae.pth")
                io.save_weight(model=model, save_path=bp)
                LOGGER.info("APNet BEST AVG MAE -> %.6f @ epoch %d", best_avg_mae, epoch)
            for dataset_name, metrics in results.items():
                mae = float(metrics.get("mae", metrics.get("MAE", float("inf"))))
                if mae < best_per_dataset.get(dataset_name, float("inf")):
                    best_per_dataset[dataset_name] = mae
                    io.save_weight(model=model, save_path=os.path.join(cfg.path.pth, f"best_{dataset_name}_mae.pth"))

    LOGGER.info("APNet V1.1 finished in %s | best_avg_mae=%.6f",
                str(datetime.timedelta(seconds=int(time.perf_counter()-train_start))), best_avg_mae)


def train(model, cfg):
    if str(cfg.get("model_type", "anet")).lower() == "apnet":
        return train_apnet_joint(model, cfg)
    return train_anet_original(model, cfg)


@torch.no_grad()
def generate_pseudo(model, cfg):
    names = list(cfg.generate.data.names)
    infos = {name: cfg.dataset_infos[name] for name in names}
    dataset = ANetPromptDataset(
        dataset_infos=infos,
        shape=cfg.generate.data.shape,
        box_format=str(cfg.generate.get("box_format", "xyxy")),
    )
    loader = data.DataLoader(
        dataset,
        batch_size=int(cfg.generate.batch_size),
        num_workers=int(cfg.generate.num_workers),
        shuffle=False,
        pin_memory=True,
    )

    mask_dir = os.path.join(cfg.path.output_dir, "pseudo_mask")
    edge_dir = os.path.join(cfg.path.output_dir, "pseudo_edge")
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(edge_dir, exist_ok=True)

    model.eval()
    for batch in tqdm(loader, desc="[ANet pseudo]", ncols=88):
        batch_data = pt_utils.to_device(
            batch["data"],
            device=cfg.device,
        )
        out = model(
            data=batch_data,
            return_aux=True,
        )

        mask_prob = torch.sigmoid(out["logits"]).cpu().numpy()
        edge_prob = torch.sigmoid(out["edge_logits"][-1]).cpu().numpy()

        names_batch = batch["info"]["name"]
        orig_h = batch["info"]["orig_h"]
        orig_w = batch["info"]["orig_w"]

        for i, name in enumerate(names_batch):
            h = int(orig_h[i])
            w = int(orig_w[i])

            mask = mask_prob[i, 0]
            edge = edge_prob[i, 0]

            mask = cv2.resize(
                mask,
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )
            edge = cv2.resize(
                edge,
                (w, h),
                interpolation=cv2.INTER_LINEAR,
            )

            cv2.imwrite(
                os.path.join(mask_dir, f"{name}.png"),
                np.clip(mask * 255.0, 0, 255).astype(np.uint8),
            )
            cv2.imwrite(
                os.path.join(edge_dir, f"{name}.png"),
                np.clip(edge * 255.0, 0, 255).astype(np.uint8),
            )

    LOGGER.info("Pseudo masks: %s", mask_dir)
    LOGGER.info("Pseudo edges: %s", edge_dir)


def parse_cfg():
    parser = argparse.ArgumentParser("ConvNeXtB ZoomNeXt ANet")
    parser.add_argument(
        "--config",
        default="configs/anet_convnextb_zoomnext.py",
    )
    parser.add_argument(
        "--data-cfg",
        default="./dataset.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="ANet_outputs",
    )
    parser.add_argument("--load-from", type=str)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--generate",dest="do_generate", action="store_true")
    parser.add_argument("--use-checkpoint", action="store_true")
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(vars(args))

    with open(cfg.data_cfg, "r", encoding="utf-8") as f:
        cfg.dataset_infos = yaml.safe_load(f)

    cfg.proj_root = os.path.dirname(os.path.abspath(__file__))
    cfg.model_name = "ConvNeXtB_ZoomNeXt_ANet"
    cfg.exp_name = py_utils.construct_exp_name(
        model_name=cfg.model_name,
        cfg=cfg,
    )
    cfg.output_dir = os.path.join(cfg.proj_root, cfg.output_dir)
    cfg.path = py_utils.construct_path(
        output_dir=cfg.output_dir,
        exp_name=cfg.exp_name,
    )
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    py_utils.pre_mkdir(cfg.path)
    with open(cfg.path.cfg_copy, "w", encoding="utf-8") as f:
        f.write(cfg.pretty_text)
    shutil.copy(__file__, cfg.path.trainer_copy)

    file_handler = logging.FileHandler(cfg.path.log)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("[%(filename)s] %(message)s")
    )
    LOGGER.addHandler(file_handler)
    LOGGER.info(cfg.pretty_text)

    return cfg


def main():
    cfg = parse_cfg()
    pt_utils.initialize_seed_cudnn(
        seed=int(cfg.base_seed),
        deterministic=bool(cfg.deterministic),
    )

    model_kwargs = dict(cfg.get("model", {}))
    model_type = str(cfg.get("model_type", "anet"))

    model = build_anet_or_apnet(
        model_type=model_type,
        pretrained=bool(cfg.pretrained),
        use_checkpoint=bool(cfg.use_checkpoint),
        **model_kwargs,
    ).to(cfg.device)
    

    LOGGER.info(
        "Number of Parameters: %.3fM",
        sum(p.numel() for p in model.parameters()) / 1e6,
    )

    if cfg.load_from:
        io.load_weight(
            model=model,
            load_path=cfg.load_from,
            strict=True,
        )
        LOGGER.info("Loaded weight: %s", cfg.load_from)

    if cfg.do_generate:
        if not cfg.load_from:
            raise ValueError(
                "--generate requires --load-from with a trained ANet weight."
            )
        generate_pseudo(model, cfg)
        return

    train(model, cfg)


if __name__ == "__main__":
    main()
