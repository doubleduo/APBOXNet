#!/usr/bin/env python3
"""Train and test the clean RGB-only ZoomNeXt PNet.

PNet consumes the exact split written by ``anet_main.py``. It never creates a
new 20% split and never reads Box annotations.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import logging
import math
import random
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from .pnet_model import ZoomNeXtPNet, compute_pnet_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("clean_pnet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Clean RGB-only ZoomNeXt PNet")
    parser.add_argument("--config", default="clean_pipeline/pnet_config.py")
    parser.add_argument("--mode", choices=("train", "test", "all"), default="train")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--edge-mode", choices=("none", "aux", "sobel"), default=None)
    return parser.parse_args()


def load_config(path: str) -> dict:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = PROJECT_ROOT / path_obj
    spec = importlib.util.spec_from_file_location("clean_pnet_config", path_obj)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import config: {path_obj}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "cfg"):
        raise KeyError(f"Config has no `cfg` dictionary: {path_obj}")
    return copy.deepcopy(module.cfg)


def resolve_path(value) -> Path | None:
    if value in (None, ""):
        return None
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_ROOT / path


def apply_overrides(cfg: dict, args: argparse.Namespace) -> dict:
    if args.output_dir:
        cfg["experiment"]["output_dir"] = args.output_dir
    if args.edge_mode:
        cfg["model"]["edge_mode"] = args.edge_mode
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    LOGGER.setLevel(logging.INFO)
    LOGGER.handlers.clear()
    formatter = logging.Formatter("[%(asctime)s] %(message)s", "%Y-%m-%d %H:%M:%S")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    LOGGER.addHandler(stream)
    file_handler = logging.FileHandler(output_dir / "train.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    LOGGER.addHandler(file_handler)


def read_names(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    names = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        item = raw.strip()
        if item and not item.startswith("#"):
            names.append(Path(item.split()[0]).stem)
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate names in {path}")
    return names


def list_stems(folder: Path, suffix: str) -> List[str]:
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return sorted(
        path.name[: -len(suffix)]
        for path in folder.iterdir()
        if path.is_file() and path.name.endswith(suffix)
    )


def load_training_split(cfg: dict) -> Tuple[Dict[str, List[str]], dict]:
    split_dir = resolve_path(cfg["data"]["split_dir"])
    manifest_path = split_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"ANet split manifest not found: {manifest_path}. Run anet_main.py first."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split = {
        key: read_names(split_dir / f"{key}.txt")
        for key in ("labeled_all", "labeled_train", "labeled_val", "unlabeled")
    }
    labeled_all = set(split["labeled_all"])
    if set(split["labeled_train"]) | set(split["labeled_val"]) != labeled_all:
        raise ValueError("Invalid ANet split: train + val != labeled_all")
    if set(split["labeled_train"]) & set(split["labeled_val"]):
        raise ValueError("Invalid ANet split: train and val overlap")
    if labeled_all & set(split["unlabeled"]):
        raise ValueError("Invalid ANet split: labeled and unlabeled overlap")
    if int(manifest["labeled_count"]) != len(split["labeled_all"]):
        raise ValueError("Split manifest labeled_count mismatch")
    if int(manifest["unlabeled_count"]) != len(split["unlabeled"]):
        raise ValueError("Split manifest unlabeled_count mismatch")
    return split, manifest


def mask_to_edge(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    binary = (mask > 0.5).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return (cv2.dilate(binary, kernel) - cv2.erode(binary, kernel)).astype(np.float32)


def read_gray(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image.astype(np.float32) / 255.0


def augment_sample(
    image: np.ndarray,
    target: np.ndarray,
    confidence: np.ndarray,
    edge: np.ndarray,
    soft_target: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        target = np.ascontiguousarray(target[:, ::-1])
        confidence = np.ascontiguousarray(confidence[:, ::-1])
        edge = np.ascontiguousarray(edge[:, ::-1])
    if random.random() < 0.35:
        height, width = target.shape
        angle = random.uniform(-30.0, 30.0)
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
        image = cv2.warpAffine(
            image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
        target = cv2.warpAffine(
            target,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR if soft_target else cv2.INTER_NEAREST,
        )
        confidence = cv2.warpAffine(confidence, matrix, (width, height), flags=cv2.INTER_LINEAR)
        edge = cv2.warpAffine(edge, matrix, (width, height), flags=cv2.INTER_LINEAR)
    if random.random() < 0.5:
        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-12.0, 12.0)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return image, target, confidence, edge


class PNetDataset(Dataset):
    def __init__(
        self,
        names: Sequence[str],
        data_cfg: dict,
        image_size: int,
        source: str,
        augment: bool,
    ) -> None:
        if source not in {"labeled", "pseudo"}:
            raise ValueError("source must be labeled or pseudo")
        self.names = list(names)
        self.source = source
        self.image_dir = resolve_path(data_cfg["image_dir"])
        self.image_suffix = data_cfg["image_suffix"]
        self.target_dir = resolve_path(
            data_cfg["mask_dir"] if source == "labeled" else data_cfg["pseudo_mask_dir"]
        )
        self.target_suffix = (
            data_cfg["mask_suffix"] if source == "labeled" else data_cfg["pseudo_suffix"]
        )
        self.edge_dir = (
            None if source == "labeled" else resolve_path(data_cfg.get("pseudo_edge_dir"))
        )
        self.confidence_dir = (
            None
            if source == "labeled"
            else resolve_path(data_cfg.get("pseudo_confidence_dir"))
        )
        self.image_size = int(image_size)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict:
        name = self.names[index]
        image_path = self.image_dir / f"{name}{self.image_suffix}"
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        target = read_gray(self.target_dir / f"{name}{self.target_suffix}")
        soft_target = self.source == "pseudo"
        if not soft_target:
            target = (target > 0.5).astype(np.float32)
        if image.shape[:2] != target.shape:
            image = cv2.resize(image, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)

        confidence = np.ones_like(target, dtype=np.float32)
        if self.confidence_dir is not None:
            confidence_path = self.confidence_dir / f"{name}{self.target_suffix}"
            confidence = read_gray(confidence_path)
            if confidence.shape != target.shape:
                confidence = cv2.resize(
                    confidence, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR
                )

        edge = None
        if self.edge_dir is not None:
            edge_path = self.edge_dir / f"{name}{self.target_suffix}"
            if edge_path.is_file():
                edge = read_gray(edge_path)
                if edge.shape != target.shape:
                    edge = cv2.resize(edge, (target.shape[1], target.shape[0]), interpolation=cv2.INTER_LINEAR)
        if edge is None:
            edge = mask_to_edge(target)

        if self.augment:
            image, target, confidence, edge = augment_sample(
                image, target, confidence, edge, soft_target
            )
        size = (self.image_size, self.image_size)
        image = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)
        target = cv2.resize(
            target,
            size,
            interpolation=cv2.INTER_LINEAR if soft_target else cv2.INTER_NEAREST,
        )
        confidence = cv2.resize(confidence, size, interpolation=cv2.INTER_LINEAR)
        edge = cv2.resize(edge, size, interpolation=cv2.INTER_LINEAR)
        return {
            "image": torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255.0),
            "mask": torch.from_numpy(target.copy()).unsqueeze(0).float().clamp(0, 1),
            "confidence": torch.from_numpy(confidence.copy()).unsqueeze(0).float().clamp(0, 1),
            "edge": torch.from_numpy(edge.copy()).unsqueeze(0).float().clamp(0, 1),
            "is_pseudo": torch.tensor(soft_target, dtype=torch.bool),
            "name": name,
        }


class TestDataset(Dataset):
    def __init__(self, dataset_cfg: dict, image_size: int) -> None:
        self.image_dir = resolve_path(dataset_cfg["image_dir"])
        self.image_suffix = dataset_cfg["image_suffix"]
        self.mask_dir = resolve_path(dataset_cfg.get("mask_dir"))
        self.mask_suffix = dataset_cfg.get("mask_suffix", ".png")
        self.names = list_stems(self.image_dir, self.image_suffix)
        self.image_size = int(image_size)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict:
        name = self.names[index]
        image_path = self.image_dir / f"{name}{self.image_suffix}"
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        return {
            "image": torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255.0),
            "name": name,
            "height": height,
            "width": width,
        }


def verify_training_files(cfg: dict, split: Dict[str, List[str]]) -> None:
    data_cfg = cfg["data"]
    image_dir = resolve_path(data_cfg["image_dir"])
    mask_dir = resolve_path(data_cfg["mask_dir"])
    pseudo_dir = resolve_path(data_cfg["pseudo_mask_dir"])
    if not pseudo_dir.is_dir():
        raise FileNotFoundError(f"Pseudo-mask directory not found: {pseudo_dir}")
    image_names = set(list_stems(image_dir, data_cfg["image_suffix"]))
    mask_names = set(list_stems(mask_dir, data_cfg["mask_suffix"]))
    pseudo_names = set(list_stems(pseudo_dir, data_cfg["pseudo_suffix"]))
    missing_images = (set(split["labeled_all"]) | set(split["unlabeled"])) - image_names
    missing_gt = set(split["labeled_all"]) - mask_names
    missing_pseudo = set(split["unlabeled"]) - pseudo_names
    if missing_images:
        raise RuntimeError(f"Missing training images, e.g. {sorted(missing_images)[:5]}")
    if missing_gt:
        raise RuntimeError(f"Missing labeled GT, e.g. {sorted(missing_gt)[:5]}")
    if missing_pseudo:
        raise RuntimeError(f"Missing pseudo masks, e.g. {sorted(missing_pseudo)[:5]}")
    extra = pseudo_names - set(split["unlabeled"])
    if extra:
        LOGGER.warning("Ignoring %d pseudo masks not present in unlabeled.txt", len(extra))


def merge_batches(labeled: dict, pseudo: dict, device: torch.device) -> dict:
    keys = ("image", "mask", "edge", "confidence", "is_pseudo")
    return {
        key: torch.cat((labeled[key], pseudo[key]), dim=0).to(device, non_blocking=True)
        for key in keys
    }


def build_model(cfg: dict, device: torch.device) -> ZoomNeXtPNet:
    model = ZoomNeXtPNet(**cfg["model"]).to(device)
    LOGGER.info(
        "PNet parameters: %.3f M | edge_mode=%s | input=RGB only",
        sum(p.numel() for p in model.parameters()) / 1e6,
        cfg["model"]["edge_mode"],
    )
    return model


def lr_at_step(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return min_lr + 0.5 * (base_lr - min_lr) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(model: ZoomNeXtPNet, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    errors = []
    for batch in tqdm(loader, desc="PNet val", ncols=100):
        image = batch["image"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        error = (model.predict(image) - mask).abs().mean((1, 2, 3))
        errors.extend(error.cpu().tolist())
    return float(np.mean(errors))


def save_checkpoint(path: Path, model, optimizer, epoch: int, split_digest: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict() if optimizer is not None else None,
            "epoch": int(epoch),
            "split_digest": split_digest,
        },
        path,
    )


def load_checkpoint(
    path: Path,
    model: ZoomNeXtPNet,
    device: torch.device,
    expected_split_digest: str | None = None,
) -> dict:
    payload = torch.load(path, map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    if expected_split_digest and isinstance(payload, dict):
        actual = payload.get("split_digest")
        if actual and actual != expected_split_digest:
            raise RuntimeError("Checkpoint and current ANet split have different digests")
    return payload if isinstance(payload, dict) else {}


def train_pnet(
    cfg: dict,
    split: Dict[str, List[str]],
    manifest: dict,
    output_dir: Path,
    model: ZoomNeXtPNet,
    device: torch.device,
) -> Path:
    train_cfg = cfg["train"]
    final_fit = bool(train_cfg["final_fit"])
    labeled_names = split["labeled_all"] if final_fit else split["labeled_train"]
    val_names = [] if final_fit else split["labeled_val"]
    labeled_set = PNetDataset(
        labeled_names, cfg["data"], train_cfg["image_size"], "labeled", train_cfg["augment"]
    )
    pseudo_set = PNetDataset(
        split["unlabeled"], cfg["data"], train_cfg["image_size"], "pseudo", train_cfg["augment"]
    )
    val_set = PNetDataset(
        val_names, cfg["data"], train_cfg["image_size"], "labeled", False
    )
    if not len(labeled_set) or not len(pseudo_set):
        raise RuntimeError("PNet requires both labeled and pseudo datasets")

    labeled_loader = DataLoader(
        labeled_set,
        batch_size=int(train_cfg["batch_size_labeled"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    pseudo_loader = DataLoader(
        pseudo_set,
        batch_size=int(train_cfg["batch_size_pseudo"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=int(train_cfg["batch_size_pseudo"]),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    ) if len(val_set) else None

    backbone_params = list(model.encoder.parameters())
    decoder_params = [
        parameter
        for name, parameter in model.named_parameters()
        if not name.startswith("encoder.")
    ]
    optimizer_name = str(train_cfg["optimizer"]).lower()
    if optimizer_name not in {"adam", "adamw"}:
        raise ValueError("PNet optimizer must be adam or adamw")
    optimizer_cls = torch.optim.AdamW if optimizer_name == "adamw" else torch.optim.Adam
    optimizer = optimizer_cls(
        [
            {
                "params": backbone_params,
                "lr": float(train_cfg["lr"]) * float(train_cfg["backbone_lr_scale"]),
                "lr_scale": float(train_cfg["backbone_lr_scale"]),
            },
            {"params": decoder_params, "lr": float(train_cfg["lr"]), "lr_scale": 1.0},
        ],
        weight_decay=float(train_cfg["weight_decay"]),
    )
    amp = bool(train_cfg["amp"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    total_steps = int(train_cfg["epochs"]) * len(pseudo_loader)
    warmup_steps = int(train_cfg["warmup_epochs"]) * len(pseudo_loader)
    global_step = 0
    best_mae = float("inf")
    checkpoint_dir = output_dir / "checkpoints"
    best_path = checkpoint_dir / "best.pth"
    last_path = checkpoint_dir / "last.pth"

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        labeled_iter = iter(labeled_loader)
        running = {"total": 0.0, "mask": 0.0, "labeled": 0.0, "pseudo": 0.0, "edge": 0.0, "ual": 0.0}
        loader = tqdm(pseudo_loader, desc=f"PNet {epoch:03d}", ncols=125)
        for step, pseudo_batch in enumerate(loader, 1):
            try:
                labeled_batch = next(labeled_iter)
            except StopIteration:
                labeled_iter = iter(labeled_loader)
                labeled_batch = next(labeled_iter)
            batch = merge_batches(labeled_batch, pseudo_batch, device)
            lr = lr_at_step(
                global_step,
                total_steps,
                warmup_steps,
                float(train_cfg["lr"]),
                float(train_cfg["min_lr"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr * float(group["lr_scale"])
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(batch["image"])
                losses = compute_pnet_loss(
                    outputs,
                    mask=batch["mask"],
                    edge=batch["edge"],
                    is_pseudo=batch["is_pseudo"],
                    confidence=batch["confidence"],
                    mask_level_weights=train_cfg["mask_level_weights"],
                    labeled_weight=train_cfg["labeled_loss_weight"],
                    pseudo_weight=train_cfg["pseudo_loss_weight"],
                    edge_weight=train_cfg["edge_loss_weight"],
                    pseudo_edge_weight=train_cfg["pseudo_edge_weight"],
                    ual_weight=train_cfg["ual_loss_weight"],
                    progress=(global_step + 1) / max(total_steps, 1),
                )
            scaler.scale(losses["total"]).backward()
            if float(train_cfg["grad_clip"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            for key in running:
                running[key] += float(losses[key].detach())
            loader.set_postfix(loss=f"{running['total'] / step:.4f}", lr=f"{lr:.2e}")

        save_checkpoint(last_path, model, optimizer, epoch, manifest["split_digest"])
        LOGGER.info(
            "Epoch %03d | %s",
            epoch,
            " ".join(f"{key}={value / len(pseudo_loader):.5f}" for key, value in running.items()),
        )
        if val_loader is not None and epoch % int(train_cfg["validate_every"]) == 0:
            mae = validate(model, val_loader, device)
            LOGGER.info("Labeled validation | epoch=%d MAE=%.6f best=%.6f", epoch, mae, best_mae)
            if mae < best_mae:
                best_mae = mae
                save_checkpoint(best_path, model, optimizer, epoch, manifest["split_digest"])

    if val_loader is None:
        save_checkpoint(best_path, model, optimizer, int(train_cfg["epochs"]), manifest["split_digest"])
    return best_path


@torch.no_grad()
def test_pnet(
    cfg: dict,
    output_dir: Path,
    model: ZoomNeXtPNet,
    device: torch.device,
    checkpoint: Path,
    split_digest: str | None,
) -> None:
    load_checkpoint(checkpoint, model, device, split_digest)
    model.eval()
    test_cfg = cfg["test"]
    for dataset_name, dataset_cfg in test_cfg["datasets"].items():
        dataset = TestDataset(dataset_cfg, test_cfg["image_size"])
        loader = DataLoader(
            dataset,
            batch_size=int(test_cfg["batch_size"]),
            shuffle=False,
            num_workers=int(test_cfg["num_workers"]),
            pin_memory=device.type == "cuda",
        )
        prediction_dir = output_dir / "predictions" / dataset_name
        prediction_dir.mkdir(parents=True, exist_ok=True)
        errors = []
        for batch in tqdm(loader, desc=f"Test {dataset_name}", ncols=100):
            probabilities = model.predict(batch["image"].to(device, non_blocking=True)).cpu().numpy()[:, 0]
            for index, name in enumerate(batch["name"]):
                height = int(batch["height"][index])
                width = int(batch["width"][index])
                prediction = cv2.resize(
                    probabilities[index], (width, height), interpolation=cv2.INTER_LINEAR
                ).clip(0, 1)
                cv2.imwrite(
                    str(prediction_dir / f"{name}.png"), (prediction * 255).astype(np.uint8)
                )
                if dataset.mask_dir is not None:
                    gt_path = dataset.mask_dir / f"{name}{dataset.mask_suffix}"
                    if gt_path.is_file():
                        gt = read_gray(gt_path)
                        if gt.shape != prediction.shape:
                            prediction_eval = cv2.resize(
                                prediction, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR
                            )
                        else:
                            prediction_eval = prediction
                        errors.append(float(np.abs(prediction_eval - (gt > 0.5)).mean()))
        if errors:
            LOGGER.info("%s | images=%d | MAE=%.6f | predictions=%s", dataset_name, len(dataset), np.mean(errors), prediction_dir)
        else:
            LOGGER.info("%s | images=%d | predictions=%s", dataset_name, len(dataset), prediction_dir)


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    output_dir = resolve_path(cfg["experiment"]["output_dir"])
    setup_logging(output_dir)
    seed_everything(int(cfg["experiment"]["seed"]))
    split, manifest = load_training_split(cfg)
    if args.mode in {"train", "all"}:
        verify_training_files(cfg, split)
    (output_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    LOGGER.info(
        "Device=%s | labeled=%d | pseudo=%d | split=%s",
        device,
        len(split["labeled_all"]),
        len(split["unlabeled"]),
        manifest["split_digest"][:12],
    )
    model = build_model(cfg, device)
    trained_checkpoint = None
    if args.mode in {"train", "all"}:
        trained_checkpoint = train_pnet(cfg, split, manifest, output_dir, model, device)
    if args.mode in {"test", "all"}:
        checkpoint = resolve_path(args.checkpoint) if args.checkpoint else trained_checkpoint
        if checkpoint is None:
            checkpoint = output_dir / "checkpoints" / "best.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"PNet checkpoint not found: {checkpoint}")
        test_pnet(cfg, output_dir, model, device, checkpoint, manifest["split_digest"])


if __name__ == "__main__":
    main()
