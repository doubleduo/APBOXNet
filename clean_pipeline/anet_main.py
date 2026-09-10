#!/usr/bin/env python3
"""Train Box-aware ANet and generate pseudo labels.

Run from the repository root:

    python -m clean_pipeline.anet_main --mode all
    python -m clean_pipeline.anet_main --mode split --split-mode txt \
        --labeled-txt lists/F20_labeled.txt --unlabeled-txt lists/F20_unlabeled.txt
"""

from __future__ import annotations

import argparse
import copy
import hashlib
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

from .anet_model import NoisyCODANet, compute_anet_loss


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGGER = logging.getLogger("clean_anet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Clean Box-aware ANet")
    parser.add_argument("--config", default="clean_pipeline/anet_config.py")
    parser.add_argument("--mode", choices=("split", "train", "generate", "all"), default="all")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--split-mode", choices=("txt", "random"), default=None)
    parser.add_argument("--labeled-txt", default=None)
    parser.add_argument("--unlabeled-txt", default=None)
    parser.add_argument("--val-txt", default=None)
    parser.add_argument("--labeled-count", type=int, default=None)
    parser.add_argument("--random-ratio", type=float, default=None)
    parser.add_argument("--force-split", action="store_true")
    return parser.parse_args()


def load_config(path: str) -> dict:
    path_obj = Path(path)
    if not path_obj.is_absolute():
        path_obj = PROJECT_ROOT / path_obj
    spec = importlib.util.spec_from_file_location("clean_anet_config", path_obj)
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
    for key, value in (
        ("mode", args.split_mode),
        ("labeled_txt", args.labeled_txt),
        ("unlabeled_txt", args.unlabeled_txt),
        ("val_txt", args.val_txt),
        ("labeled_count", args.labeled_count),
    ):
        if value is not None:
            cfg["split"][key] = value
    if args.random_ratio is not None:
        cfg["split"]["random_ratio"] = args.random_ratio
        # An explicit ratio overrides the config's fixed F20 count unless the
        # caller also supplied --labeled-count.
        if args.labeled_count is None:
            cfg["split"]["labeled_count"] = None
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


def list_stems(folder: Path, suffix: str) -> List[str]:
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    return sorted(
        path.name[: -len(suffix)]
        for path in folder.iterdir()
        if path.is_file() and path.name.endswith(suffix)
    )


def read_names(path: Path) -> List[str]:
    if not path.is_file():
        raise FileNotFoundError(path)
    names = []
    for raw in path.read_text(encoding="utf-8-sig").splitlines():
        item = raw.strip()
        if not item or item.startswith("#"):
            continue
        # Support either bare stems or paths such as Imgs/name.jpg.
        names.append(Path(item.split()[0]).stem)
    if len(names) != len(set(names)):
        raise ValueError(f"Duplicate names in {path}")
    return names


def write_names(path: Path, names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{name}\n" for name in names), encoding="utf-8")


def digest_names(names: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode("utf-8")).hexdigest()


def validate_split(
    all_names: Sequence[str],
    labeled_all: Sequence[str],
    labeled_train: Sequence[str],
    labeled_val: Sequence[str],
    unlabeled: Sequence[str],
) -> None:
    universe = set(all_names)
    labeled_set = set(labeled_all)
    train_set = set(labeled_train)
    val_set = set(labeled_val)
    unlabeled_set = set(unlabeled)
    if len(labeled_set) != len(labeled_all) or len(unlabeled_set) != len(unlabeled):
        raise ValueError("Split contains duplicate names")
    if train_set & val_set:
        raise ValueError("labeled_train and labeled_val overlap")
    if train_set | val_set != labeled_set:
        raise ValueError("labeled_train + labeled_val must equal labeled_all")
    if labeled_set & unlabeled_set:
        raise ValueError("labeled and unlabeled splits overlap")
    unknown = (labeled_set | unlabeled_set) - universe
    if unknown:
        raise ValueError(f"Split contains unknown names, e.g. {sorted(unknown)[:5]}")
    if labeled_set | unlabeled_set != universe:
        missing = sorted(universe - labeled_set - unlabeled_set)
        raise ValueError(f"Split does not cover the dataset, e.g. {missing[:5]}")


def prepare_split(cfg: dict, output_dir: Path, force: bool = False) -> Dict[str, List[str]]:
    data_cfg = cfg["data"]
    split_cfg = cfg["split"]
    image_dir = resolve_path(data_cfg["image_dir"])
    all_names = list_stems(image_dir, data_cfg["image_suffix"])
    expected_total = int(split_cfg.get("expected_total", len(all_names)))
    if bool(split_cfg.get("strict_total", False)) and len(all_names) != expected_total:
        raise RuntimeError(f"Expected {expected_total} images, found {len(all_names)} in {image_dir}")

    split_dir = output_dir / "splits"
    files = {
        key: split_dir / f"{key}.txt"
        for key in ("labeled_all", "labeled_train", "labeled_val", "unlabeled")
    }
    manifest_path = split_dir / "manifest.json"
    if (
        not force
        and bool(split_cfg.get("reuse_saved", True))
        and manifest_path.is_file()
        and all(path.is_file() for path in files.values())
    ):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("dataset_digest") != digest_names(all_names):
            raise RuntimeError("Saved split does not match the current image directory; use --force-split")
        result = {key: read_names(path) for key, path in files.items()}
        validate_split(all_names=all_names, **result)
        LOGGER.info("Reused split: %s", split_dir)
        return result

    mode = str(split_cfg["mode"]).lower()
    if mode == "txt":
        labeled_path = resolve_path(split_cfg.get("labeled_txt"))
        if labeled_path is None:
            raise ValueError("split.mode='txt' requires split.labeled_txt")
        labeled_all = read_names(labeled_path)
        unlabeled_path = resolve_path(split_cfg.get("unlabeled_txt"))
        if unlabeled_path is None:
            unlabeled = sorted(set(all_names) - set(labeled_all))
        else:
            unlabeled = read_names(unlabeled_path)
    elif mode == "random":
        count = split_cfg.get("labeled_count")
        if count is None:
            ratio = float(split_cfg["random_ratio"])
            if not 0.0 < ratio < 1.0:
                raise ValueError(f"random_ratio must be in (0, 1), got {ratio}")
            count = int(round(len(all_names) * ratio))
        count = int(count)
        if not 0 < count < len(all_names):
            raise ValueError(f"Invalid labeled_count={count} for total={len(all_names)}")
        rng = np.random.RandomState(int(split_cfg["seed"]))
        indices = rng.choice(len(all_names), count, replace=False)
        labeled_all = sorted(np.asarray(all_names)[indices].tolist())
        unlabeled = sorted(set(all_names) - set(labeled_all))
    else:
        raise ValueError("split.mode must be 'txt' or 'random'")

    # Train on every sample listed in labeled_txt.
    # Do not carve out an internal validation subset.
    # The remaining samples (unlabeled) are used as the validation set in train_anet().
    labeled_val = []
    labeled_train = sorted(labeled_all)
    unlabeled = sorted(unlabeled)

    result = dict(
        labeled_all=sorted(labeled_all),
        labeled_train=labeled_train,
        labeled_val=labeled_val,
        unlabeled=unlabeled,
    )
    validate_split(all_names=all_names, **result)
    split_dir.mkdir(parents=True, exist_ok=True)
    for key, names in result.items():
        write_names(files[key], names)
    manifest = {
        "version": 1,
        "mode": mode,
        "seed": int(split_cfg["seed"]),
        "dataset_size": len(all_names),
        "dataset_digest": digest_names(all_names),
        "labeled_count": len(labeled_all),
        "train_count": len(labeled_train),
        "val_count": len(labeled_val),
        "unlabeled_count": len(unlabeled),
        "split_digest": hashlib.sha256(
            (
                "LABELED\n"
                + "\n".join(sorted(labeled_all))
                + "\nUNLABELED\n"
                + "\n".join(sorted(unlabeled))
            ).encode("utf-8")
        ).hexdigest(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    LOGGER.info(
        "Created %s split | total=%d labeled=%d train=%d val=%d unlabeled=%d",
        mode,
        len(all_names),
        len(labeled_all),
        len(labeled_train),
        len(labeled_val),
        len(unlabeled),
    )
    return result


def load_box(path: Path, height: int, width: int, box_format: str) -> np.ndarray:
    if box_format == "mask":
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise FileNotFoundError(path)
        if mask.shape != (height, width):
            mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        return (mask > 0).astype(np.uint8)
    if box_format != "labelme_json":
        raise ValueError(f"Unsupported box_format={box_format}")
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    mask = np.zeros((height, width), dtype=np.uint8)
    for shape in payload.get("shapes", []):
        points = np.asarray(shape.get("points", []), dtype=np.float32)
        if points.size == 0:
            continue
        x1 = int(np.floor(points[:, 0].min()))
        y1 = int(np.floor(points[:, 1].min()))
        x2 = int(np.ceil(points[:, 0].max()))
        y2 = int(np.ceil(points[:, 1].max()))
        x1, x2 = np.clip((x1, x2), 0, width - 1)
        y1, y2 = np.clip((y1, y2), 0, height - 1)
        if x2 >= x1 and y2 >= y1:
            cv2.rectangle(mask, (int(x1), int(y1)), (int(x2), int(y2)), 1, -1)
    if not mask.any():
        raise ValueError(f"No valid boxes in {path}")
    return mask


def mask_to_edge(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    binary = (mask > 0.5).astype(np.uint8)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    return (cv2.dilate(binary, kernel) - cv2.erode(binary, kernel)).astype(np.float32)


def verify_anet_files(cfg: dict, split: Dict[str, List[str]], require_gt: bool) -> None:
    data_cfg = cfg["data"]
    image_dir = resolve_path(data_cfg["image_dir"])
    mask_dir = resolve_path(data_cfg["mask_dir"])
    box_dir = resolve_path(data_cfg["box_dir"])
    names = split["labeled_all"] + split["unlabeled"]
    missing_images = [
        name
        for name in names
        if not (image_dir / f"{name}{data_cfg['image_suffix']}").is_file()
    ]
    missing_boxes = [
        name
        for name in names
        if not (box_dir / f"{name}{data_cfg['box_suffix']}").is_file()
    ]
    missing_masks = []
    if require_gt:
        # Training uses labeled_all and validation uses the remaining/unlabeled set,
        # so GT masks are required for the full dataset.
        missing_masks = [
            name
            for name in names
            if not (mask_dir / f"{name}{data_cfg['mask_suffix']}").is_file()
        ]
    if missing_images:
        raise RuntimeError(f"Missing ANet images, e.g. {missing_images[:5]}")
    if missing_boxes:
        raise RuntimeError(f"Missing ANet boxes, e.g. {missing_boxes[:5]}")
    if missing_masks:
        raise RuntimeError(f"Missing ANet labeled GT, e.g. {missing_masks[:5]}")


def augment_sample(
    image: np.ndarray, mask: np.ndarray, box: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        image = np.ascontiguousarray(image[:, ::-1])
        mask = np.ascontiguousarray(mask[:, ::-1])
        box = np.ascontiguousarray(box[:, ::-1])
    if random.random() < 0.35:
        height, width = mask.shape
        angle = random.uniform(-30.0, 30.0)
        matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
        image = cv2.warpAffine(
            image, matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101
        )
        mask = cv2.warpAffine(mask, matrix, (width, height), flags=cv2.INTER_NEAREST)
        box = cv2.warpAffine(box, matrix, (width, height), flags=cv2.INTER_NEAREST)
    if random.random() < 0.5:
        alpha = random.uniform(0.9, 1.1)
        beta = random.uniform(-12.0, 12.0)
        image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
    return image, mask, box


class ANetDataset(Dataset):
    def __init__(
        self,
        names: Sequence[str],
        data_cfg: dict,
        image_size: int,
        training: bool,
        require_mask: bool,
    ) -> None:
        self.names = list(names)
        self.image_dir = resolve_path(data_cfg["image_dir"])
        self.mask_dir = resolve_path(data_cfg["mask_dir"])
        self.box_dir = resolve_path(data_cfg["box_dir"])
        self.image_suffix = data_cfg["image_suffix"]
        self.mask_suffix = data_cfg["mask_suffix"]
        self.box_suffix = data_cfg["box_suffix"]
        self.box_format = data_cfg["box_format"]
        self.image_size = int(image_size)
        self.training = bool(training)
        self.require_mask = bool(require_mask)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict:
        name = self.names[index]
        image_path = self.image_dir / f"{name}{self.image_suffix}"
        image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise FileNotFoundError(image_path)
        image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        orig_height, orig_width = image.shape[:2]
        box = load_box(
            self.box_dir / f"{name}{self.box_suffix}",
            orig_height,
            orig_width,
            self.box_format,
        )

        mask = None
        if self.require_mask:
            mask_path = self.mask_dir / f"{name}{self.mask_suffix}"
            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(mask_path)
            if mask.shape != image.shape[:2]:
                image = cv2.resize(image, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_LINEAR)
                box = cv2.resize(box, (mask.shape[1], mask.shape[0]), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0).astype(np.float32)
            if self.training:
                image, mask, box = augment_sample(image, mask, box)

        image = cv2.resize(image, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
        box = cv2.resize(box, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        sample = {
            "image": torch.from_numpy(image.copy()).permute(2, 0, 1).float().div(255.0),
            "box_mask": torch.from_numpy(box.copy()).unsqueeze(0).float().clamp(0, 1),
            "name": name,
            "orig_height": orig_height,
            "orig_width": orig_width,
        }
        if mask is not None:
            mask = cv2.resize(mask, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
            sample["mask"] = torch.from_numpy(mask.copy()).unsqueeze(0).float()
            sample["edge"] = torch.from_numpy(mask_to_edge(mask)).unsqueeze(0).float()
        return sample


def build_model(cfg: dict, device: torch.device) -> NoisyCODANet:
    model = NoisyCODANet(**cfg["model"]).to(device)
    LOGGER.info("ANet parameters: %.3f M", sum(p.numel() for p in model.parameters()) / 1e6)
    return model


def learning_rate(epoch: int, train_cfg: dict) -> float:
    warmup = int(train_cfg["warmup_epochs"])
    epochs = int(train_cfg["epochs"])
    peak = float(train_cfg["peak_lr"])
    minimum = float(train_cfg["min_lr"])
    if epoch <= warmup:
        start = float(train_cfg["init_lr"])
        return start + (peak - start) * epoch / max(warmup, 1)
    progress = (epoch - warmup) / max(epochs - warmup, 1)
    return minimum + 0.5 * (peak - minimum) * (1.0 + math.cos(math.pi * progress))


@torch.no_grad()
def validate(model: NoisyCODANet, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    errors = []
    for batch in tqdm(loader, desc="ANet val", ncols=100):
        image = batch["image"].to(device, non_blocking=True)
        box = batch["box_mask"].to(device, non_blocking=True)
        mask = batch["mask"].to(device, non_blocking=True)
        error = (model.predict(image, box) - mask).abs().mean((1, 2, 3))
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


def load_checkpoint(path: Path, model, device: torch.device) -> dict:
    payload = torch.load(path, map_location=device)
    state = payload.get("model", payload) if isinstance(payload, dict) else payload
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    return payload if isinstance(payload, dict) else {}


def train_anet(
    cfg: dict,
    split: Dict[str, List[str]],
    output_dir: Path,
    model: NoisyCODANet,
    device: torch.device,
) -> Path:
    train_cfg = cfg["train"]
    train_set = ANetDataset(
        split["labeled_train"], cfg["data"], train_cfg["image_size"], True, True
    )
    if not train_set:
        raise RuntimeError("labeled_train is empty")
    # Validate on every sample not present in labeled_txt.
    val_set = ANetDataset(
        split["unlabeled"], cfg["data"], train_cfg["image_size"], False, True
    )
    LOGGER.info(
        "ANet data | train=%d (all labeled_txt) | val=%d (remaining set)",
        len(train_set), len(val_set),
    )
    train_loader = DataLoader(
        train_set,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
        drop_last=False,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=max(1, int(train_cfg["batch_size"]) * 2),
        shuffle=False,
        num_workers=int(train_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    ) if len(val_set) else None

    optimizer_name = str(train_cfg["optimizer"]).lower()
    optimizer_cls = torch.optim.Adam if optimizer_name == "adam" else torch.optim.AdamW
    if optimizer_name not in {"adam", "adamw"}:
        raise ValueError("ANet optimizer must be adam or adamw")
    optimizer = optimizer_cls(
        model.parameters(),
        lr=float(train_cfg["init_lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    amp = bool(train_cfg["amp"] and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    checkpoint_dir = output_dir / "checkpoints"
    last_path = checkpoint_dir / "last.pth"
    best_path = checkpoint_dir / "best.pth"
    manifest = json.loads((output_dir / "splits" / "manifest.json").read_text())
    best_mae = float("inf")

    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        lr = learning_rate(epoch, train_cfg)
        for group in optimizer.param_groups:
            group["lr"] = lr
        model.train()
        running = {"total": 0.0, "mask": 0.0, "edge": 0.0, "ual": 0.0}
        loader = tqdm(train_loader, desc=f"ANet {epoch:03d}", ncols=120)
        for step, batch in enumerate(loader, 1):
            image = batch["image"].to(device, non_blocking=True)
            box = batch["box_mask"].to(device, non_blocking=True)
            mask = batch["mask"].to(device, non_blocking=True)
            edge = batch["edge"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                outputs = model(image, box)
                losses = compute_anet_loss(
                    outputs,
                    mask,
                    edge,
                    progress=step / max(len(train_loader), 1),
                    edge_weight=train_cfg["edge_loss_weight"],
                    ual_weight=train_cfg["ual_loss_weight"],
                )
            scaler.scale(losses["total"]).backward()
            if float(train_cfg["grad_clip"]) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
            scaler.step(optimizer)
            scaler.update()
            for key in running:
                running[key] += float(losses[key].detach())
            loader.set_postfix(loss=f"{running['total'] / step:.4f}", lr=f"{lr:.2e}")

        save_checkpoint(last_path, model, optimizer, epoch, manifest["split_digest"])
        message = " ".join(f"{key}={value / len(train_loader):.5f}" for key, value in running.items())
        LOGGER.info("Epoch %03d | lr=%.3e | %s", epoch, lr, message)
        if val_loader is not None and epoch % int(train_cfg["validate_every"]) == 0:
            mae = validate(model, val_loader, device)
            LOGGER.info("Remaining-set validation | epoch=%d MAE=%.6f best=%.6f", epoch, mae, best_mae)
            if mae < best_mae:
                best_mae = mae
                save_checkpoint(best_path, model, optimizer, epoch, manifest["split_digest"])

    if val_loader is None:
        save_checkpoint(best_path, model, optimizer, int(train_cfg["epochs"]), manifest["split_digest"])
    return best_path


@torch.no_grad()
def generate_pseudo(
    cfg: dict,
    names: Sequence[str],
    output_dir: Path,
    model: NoisyCODANet,
    device: torch.device,
    checkpoint: Path,
) -> None:
    load_checkpoint(checkpoint, model, device)
    model.eval()
    gen_cfg = cfg["generate"]
    dataset = ANetDataset(names, cfg["data"], gen_cfg["image_size"], False, False)
    loader = DataLoader(
        dataset,
        batch_size=int(gen_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(gen_cfg["num_workers"]),
        pin_memory=device.type == "cuda",
    )
    mask_dir = output_dir / "pseudo_mask"
    edge_dir = output_dir / "pseudo_edge"
    mask_dir.mkdir(parents=True, exist_ok=True)
    if bool(gen_cfg["save_edge"]):
        edge_dir.mkdir(parents=True, exist_ok=True)

    for batch in tqdm(loader, desc="Generate pseudo", ncols=100):
        image = batch["image"].to(device, non_blocking=True)
        box = batch["box_mask"].to(device, non_blocking=True)
        outputs = model(image, box)
        masks = torch.sigmoid(outputs["mask_logits"][-1]).cpu().numpy()[:, 0]
        edges = torch.sigmoid(outputs["edge_logits"][-1]).cpu().numpy()[:, 0]
        for idx, name in enumerate(batch["name"]):
            height = int(batch["orig_height"][idx])
            width = int(batch["orig_width"][idx])
            mask = cv2.resize(masks[idx], (width, height), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(str(mask_dir / f"{name}.png"), np.clip(mask * 255, 0, 255).astype(np.uint8))
            if bool(gen_cfg["save_edge"]):
                edge = cv2.resize(edges[idx], (width, height), interpolation=cv2.INTER_LINEAR)
                cv2.imwrite(str(edge_dir / f"{name}.png"), np.clip(edge * 255, 0, 255).astype(np.uint8))
    LOGGER.info("Generated %d pseudo masks in %s", len(dataset), mask_dir)


def main() -> None:
    args = parse_args()
    cfg = apply_overrides(load_config(args.config), args)
    output_dir = resolve_path(cfg["experiment"]["output_dir"])
    setup_logging(output_dir)
    seed_everything(int(cfg["experiment"]["seed"]))
    split = prepare_split(cfg, output_dir, force=args.force_split)
    (output_dir / "resolved_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if args.mode == "split":
        return

    verify_anet_files(cfg, split, require_gt=args.mode in {"train", "all"})

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    LOGGER.info("Device: %s", device)
    model = build_model(cfg, device)
    trained_checkpoint = None
    if args.mode in {"train", "all"}:
        trained_checkpoint = train_anet(cfg, split, output_dir, model, device)
    if args.mode in {"generate", "all"}:
        checkpoint = resolve_path(args.checkpoint) if args.checkpoint else trained_checkpoint
        if checkpoint is None:
            checkpoint = output_dir / "checkpoints" / "best.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(f"ANet checkpoint not found: {checkpoint}")
        generate_pseudo(cfg, split["unlabeled"], output_dir, model, device, checkpoint)


if __name__ == "__main__":
    main()
