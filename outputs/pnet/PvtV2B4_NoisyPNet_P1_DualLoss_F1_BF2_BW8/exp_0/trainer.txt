#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PNet trainer with P1 source-aware dual supervision.

The network/data protocol remains the same as the current PASAM PNet:
    fully annotated batch + weak ANet pseudo-label batch
    -> concatenate images/targets
    -> one PNet optimization step

P1 additionally preserves the supervision source of every sample:
    full GT  -> is_pseudo=False
    weak     -> is_pseudo=True

The model then computes:
    full GT mask loss  = StructureLoss
    weak pseudo loss   = NCLoss
    mask fusion        = normalized source-level weighting (default 1:1)

Edge loss and UAL stay unchanged for a clean P0 -> P1 ablation.
"""


import argparse
import copy
import csv
import logging
import os
import random
import shutil

import albumentations as A
import colorlog
import cv2
import numpy as np
import torch
import yaml
from mmengine import Config
from torch.utils import data

import basemain as base
from methods.noisycod_pnet import PvtV2B4_NoisyPNet, mask_to_boundary
from utils import io, ops, pipeline, pt_utils, py_utils, recorder


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


def _all_names(path, suffix):
    return sorted(
        p[: -len(suffix)]
        for p in os.listdir(path)
        if p.endswith(suffix)
    )


def _dataset_paths(dataset_info):
    root = dataset_info["root"]

    image_path = os.path.join(
        root,
        dataset_info["image"]["path"],
    )
    image_suffix = dataset_info["image"]["suffix"]

    mask_path = os.path.join(
        root,
        dataset_info["mask"]["path"],
    )
    mask_suffix = dataset_info["mask"]["suffix"]

    return image_path, image_suffix, mask_path, mask_suffix


def _resolve_path(path, proj_root):
    path = str(path)
    if os.path.isabs(path):
        return path
    return os.path.join(proj_root, path)


def _same_split_as_anet(names, ratio, seed):
    """
    Reconstruct the deterministic fully-labelled subset used by anet_main.py:
      sorted names -> random.Random(seed).shuffle -> first round(N*ratio)
    """
    names = list(sorted(names))
    rng = random.Random(int(seed))
    rng.shuffle(names)

    keep = max(1, int(round(len(names) * float(ratio))))
    full_names = sorted(names[:keep])
    weak_names = sorted(names[keep:])
    return full_names, weak_names


def _read_soft_mask(path):
    arr = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise FileNotFoundError(path)
    return arr.astype(np.float32) / 255.0


def _numpy_boundary(mask01, kernel_size=5):
    t = torch.from_numpy(mask01).float()[None, None]
    b = mask_to_boundary(t, kernel_size=kernel_size)
    return b[0, 0].numpy()


class PNetDataset(data.Dataset):
    def __init__(
        self,
        image_path,
        image_suffix,
        target_path,
        target_suffix,
        names,
        shape,
        is_pseudo=False,
        edge_path=None,
        edge_suffix=".png",
        augment=True,
    ):
        super().__init__()
        self.image_path = image_path
        self.image_suffix = image_suffix
        self.target_path = target_path
        self.target_suffix = target_suffix
        self.names = list(names)
        self.shape = shape
        self.is_pseudo = bool(is_pseudo)
        self.edge_path = edge_path
        self.edge_suffix = edge_suffix
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
            additional_targets={"edge": "mask"},
        )

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        name = self.names[index]

        image_file = os.path.join(
            self.image_path,
            name + self.image_suffix,
        )
        target_file = os.path.join(
            self.target_path,
            name + self.target_suffix,
        )

        image = io.read_color_array(image_file)

        if self.is_pseudo:
            mask = _read_soft_mask(target_file)
        else:
            mask = cv2.imread(target_file, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                raise FileNotFoundError(target_file)
            mask = (mask > 0).astype(np.float32)

        if self.edge_path:
            edge_file = os.path.join(
                self.edge_path,
                name + self.edge_suffix,
            )
            if os.path.isfile(edge_file):
                edge = _read_soft_mask(edge_file)
            else:
                edge = _numpy_boundary(mask)
        else:
            edge = _numpy_boundary(mask)

        if image.shape[:2] != mask.shape:
            mh, mw = mask.shape
            image = ops.resize(
                image,
                height=mh,
                width=mw,
            )

        if self.augment:
            transformed = self.transforms(
                image=image,
                mask=mask,
                edge=edge,
            )
            image = transformed["image"]
            mask = transformed["mask"]
            edge = transformed["edge"]

        h = int(self.shape["h"])
        w = int(self.shape["w"])

        image = ops.resize(image, height=h, width=w)

        # P1 intentionally keeps the current P0 resize behavior unchanged.
        # Soft-pseudo bilinear resize should be tested as a later ablation.
        mask = cv2.resize(
            mask,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )
        edge = cv2.resize(
            edge,
            (w, h),
            interpolation=cv2.INTER_NEAREST,
        )

        image_t = (
            torch.from_numpy(image)
            .float()
            .div(255.0)
            .permute(2, 0, 1)
        )
        mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0)
        edge_t = torch.from_numpy(edge.astype(np.float32)).unsqueeze(0)

        return {
            "data": {
                "image_m": image_t,
                "mask": mask_t.clamp(0.0, 1.0),
                "edge": edge_t.clamp(0.0, 1.0),
            },
            "info": {
                "name": name,
                "is_pseudo": self.is_pseudo,
            },
        }


def _prepare_pnet_datasets(cfg):
    if len(cfg.train.data.names) != 1:
        raise ValueError(
            "PNet patch currently expects one training dataset name."
        )

    dataset_name = cfg.train.data.names[0]
    dataset_info = cfg.dataset_infos[dataset_name]

    image_path, image_suffix, gt_path, gt_suffix = _dataset_paths(dataset_info)

    image_names = set(_all_names(image_path, image_suffix))
    gt_names = set(_all_names(gt_path, gt_suffix))
    all_names = sorted(image_names & gt_names)

    full_names, weak_names = _same_split_as_anet(
        all_names,
        ratio=float(cfg.train.full_ratio),
        seed=int(cfg.train.split_seed),
    )

    pseudo_mask_root = _resolve_path(
        cfg.train.pseudo_mask_root,
        cfg.proj_root,
    )
    pseudo_edge_root = cfg.train.get("pseudo_edge_root", None)
    if pseudo_edge_root:
        pseudo_edge_root = _resolve_path(
            pseudo_edge_root,
            cfg.proj_root,
        )

    pseudo_suffix = str(cfg.train.get("pseudo_suffix", ".png"))
    pseudo_names = set(_all_names(pseudo_mask_root, pseudo_suffix))
    weak_names = [n for n in weak_names if n in pseudo_names]

    if not full_names:
        raise RuntimeError("No fully-labelled PNet samples.")
    if not weak_names:
        raise RuntimeError(
            "No weak PNet samples found. Check pseudo_mask_root."
        )

    full_dataset = PNetDataset(
        image_path=image_path,
        image_suffix=image_suffix,
        target_path=gt_path,
        target_suffix=gt_suffix,
        names=full_names,
        shape=cfg.train.data.shape,
        is_pseudo=False,
        edge_path=None,
        augment=bool(cfg.train.get("augment", True)),
    )

    weak_dataset = PNetDataset(
        image_path=image_path,
        image_suffix=image_suffix,
        target_path=pseudo_mask_root,
        target_suffix=pseudo_suffix,
        names=weak_names,
        shape=cfg.train.data.shape,
        is_pseudo=True,
        edge_path=pseudo_edge_root,
        edge_suffix=pseudo_suffix,
        augment=bool(cfg.train.get("augment", True)),
    )

    LOGGER.info(
        "PNet split | total=%d | full=%d | weak=%d | ratio=%.4f",
        len(all_names),
        len(full_dataset),
        len(weak_dataset),
        float(cfg.train.full_ratio),
    )
    LOGGER.info("Pseudo masks: %s", pseudo_mask_root)

    return full_dataset, weak_dataset


def _concat_batch(fully_batch, weak_batch, device):
    """Concatenate inputs while preserving the supervision source."""
    full = pt_utils.to_device(
        fully_batch["data"],
        device=device,
    )
    weak = pt_utils.to_device(
        weak_batch["data"],
        device=device,
    )

    num_full = int(full["image_m"].shape[0])
    num_weak = int(weak["image_m"].shape[0])

    is_pseudo = torch.cat(
        [
            torch.zeros(
                num_full,
                dtype=torch.bool,
                device=device,
            ),
            torch.ones(
                num_weak,
                dtype=torch.bool,
                device=device,
            ),
        ],
        dim=0,
    )

    return {
        "image_m": torch.cat(
            [full["image_m"], weak["image_m"]],
            dim=0,
        ),
        "mask": torch.cat(
            [full["mask"], weak["mask"]],
            dim=0,
        ),
        "edge": torch.cat(
            [full["edge"], weak["edge"]],
            dim=0,
        ),
        "is_pseudo": is_pseudo,
    }


def _append_loss_csv(csv_path, epoch, sums, n):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    exists = os.path.isfile(csv_path)

    keys = [
        "epoch",
        "total",
        "mask",
        "full_mask",
        "weak_mask",
        "full_init",
        "full_final",
        "weak_init",
        "weak_final",
        "edge",
        "ual",
        "ual_raw",
        "ual_coef",
        "q",
    ]

    row = {"epoch": int(epoch)}
    for key in keys[1:]:
        row[key] = sums.get(key, 0.0) / max(n, 1)

    with open(
        csv_path,
        "a",
        encoding="utf-8",
        newline="",
    ) as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def train(model, cfg):
    full_dataset, weak_dataset = _prepare_pnet_datasets(cfg)

    full_loader = data.DataLoader(
        full_dataset,
        batch_size=int(cfg.train.batchsize_fully),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        drop_last=True,
        pin_memory=True,
        worker_init_fn=(
            pt_utils.customized_worker_init_fn
            if cfg.use_custom_worker_init
            else None
        ),
    )
    weak_loader = data.DataLoader(
        weak_dataset,
        batch_size=int(cfg.train.batchsize_weakly),
        shuffle=True,
        num_workers=int(cfg.train.num_workers),
        drop_last=True,
        pin_memory=True,
        worker_init_fn=(
            pt_utils.customized_worker_init_fn
            if cfg.use_custom_worker_init
            else None
        ),
    )

    if len(full_loader) == 0:
        raise RuntimeError(
            "Fully loader has zero batches. Reduce train.batchsize_fully."
        )
    if len(weak_loader) == 0:
        raise RuntimeError(
            "Weak loader has zero batches. Reduce train.batchsize_weakly."
        )

    steps_mode = str(cfg.train.get("steps_mode", "official_min"))
    if steps_mode == "official_min":
        steps_per_epoch = min(len(full_loader), len(weak_loader))
    elif steps_mode == "use_all_weak":
        steps_per_epoch = len(weak_loader)
    else:
        raise ValueError("steps_mode must be official_min or use_all_weak")

    counter = recorder.TrainingCounter(
        epoch_length=steps_per_epoch,
        epoch_based=True,
        num_epochs=int(cfg.train.num_epochs),
        num_total_iters=None,
    )

    optimizer = pipeline.construct_optimizer(
        model=model,
        initial_lr=cfg.train.lr,
        mode=cfg.train.optimizer.mode,
        group_mode=cfg.train.optimizer.group_mode,
        cfg=cfg.train.optimizer.cfg,
    )

    scheduler_cfg = copy.deepcopy(cfg.train.scheduler)
    if (
        str(scheduler_cfg.mode).lower() == "step"
        and str(scheduler_cfg.cfg.get("milestones", "")).lower() == "auto"
    ):
        scheduler_cfg.cfg.milestones = max(
            1,
            int(counter.num_total_iters * 2 / 3),
        )

    scheduler = pipeline.Scheduler(
        optimizer=optimizer,
        num_iters=counter.num_total_iters,
        epoch_length=counter.num_inner_iters,
        scheduler_cfg=scheduler_cfg,
        step_by_batch=cfg.train.sche_usebatch,
    )
    scheduler.record_lrs(param_groups=optimizer.param_groups)

    scaler = pipeline.Scaler(
        optimizer,
        cfg.train.use_amp,
        set_to_none=cfg.train.optimizer.set_to_none,
    )

    loss_csv = os.path.join(
        cfg.path.pth_log,
        "pnet_loss_metrics.csv",
    )

    LOGGER.info(
        "PNet loaders | full=%d batches | weak=%d batches | "
        "steps/epoch=%d | mode=%s",
        len(full_loader),
        len(weak_loader),
        steps_per_epoch,
        steps_mode,
    )

    for _ in range(counter.num_epochs):
        epoch_no = counter.curr_epoch + 1
        q_value = 1 if epoch_no > int(cfg.train.q_epoch) else 2

        model.train()
        sums = {}
        n_steps = 0

        full_iter = iter(full_loader)
        weak_iter = iter(weak_loader)

        for step_idx in range(steps_per_epoch):
            scheduler.step(curr_idx=counter.curr_iter)

            try:
                fully_batch = next(full_iter)
            except StopIteration:
                if steps_mode == "use_all_weak":
                    full_iter = iter(full_loader)
                    fully_batch = next(full_iter)
                else:
                    break

            try:
                weak_batch = next(weak_iter)
            except StopIteration:
                break

            data_batch = _concat_batch(
                fully_batch,
                weak_batch,
                device=cfg.device,
            )

            # Keep the released PNet behavior: UAL cosine schedule resets each epoch.
            iter_percentage = (step_idx + 1) / float(max(steps_per_epoch, 1))

            with torch.cuda.amp.autocast(enabled=cfg.train.use_amp):
                outputs = model(
                    data=data_batch,
                    iter_percentage=iter_percentage,
                    q_value=q_value,
                )

            loss = outputs["loss"] / cfg.train.grad_acc_step
            scaler.calculate_grad(loss=loss)

            if counter.every_n_iters(cfg.train.grad_acc_step):
                scaler.update_grad()

            items = {}
            for key, value in outputs["loss_items"].items():
                if torch.is_tensor(value):
                    value = float(value.detach().item())
                else:
                    value = float(value)
                items[key] = value
                sums[key] = sums.get(key, 0.0) + value

            n_steps += 1

            if (
                counter.is_first_inner_iter()
                or counter.is_last_inner_iter()
                or counter.every_n_iters(cfg.log_interval)
            ):
                num_full = int((~data_batch["is_pseudo"]).sum().item())
                num_weak = int(data_batch["is_pseudo"].sum().item())
                LOGGER.info(
                    "PNet-P1 | E%03d/%03d B%04d/%04d | %s | LR:%s | "
                    "FULL/WEAK=%d/%d | shape=%s",
                    epoch_no,
                    counter.num_epochs,
                    step_idx + 1,
                    steps_per_epoch,
                    outputs["loss_str"],
                    optimizer.lr_string(),
                    num_full,
                    num_weak,
                    tuple(data_batch["mask"].shape),
                )

                for key, value in items.items():
                    cfg.tb_logger.write_to_tb(
                        f"pnet_loss/{key}",
                        value,
                        counter.curr_iter,
                    )

            if counter.curr_iter < 3:
                recorder.plot_results(
                    {
                        "img": data_batch["image_m"],
                        "msk": data_batch["mask"],
                        **outputs["vis"],
                    },
                    save_path=os.path.join(
                        cfg.path.pth_log,
                        "img",
                        f"iter_{counter.curr_iter}.png",
                    ),
                )

            if counter.is_last_total_iter():
                break
            counter.update_iter_counter()

        if n_steps == 0:
            raise RuntimeError("PNet epoch ran zero steps.")

        _append_loss_csv(
            loss_csv,
            epoch_no,
            sums,
            n_steps,
        )

        LOGGER.info(
            "PNet-P1 Epoch %03d | L:%.4f | MASK:%.4f | "
            "FULL:%.4f (I:%.4f F:%.4f) | "
            "WEAK:%.4f (I:%.4f F:%.4f) | "
            "EDGE:%.4f | UAL:%.4f | Q:%d",
            epoch_no,
            sums.get("total", 0.0) / n_steps,
            sums.get("mask", 0.0) / n_steps,
            sums.get("full_mask", 0.0) / n_steps,
            sums.get("full_init", 0.0) / n_steps,
            sums.get("full_final", 0.0) / n_steps,
            sums.get("weak_mask", 0.0) / n_steps,
            sums.get("weak_init", 0.0) / n_steps,
            sums.get("weak_final", 0.0) / n_steps,
            sums.get("edge", 0.0) / n_steps,
            sums.get("ual", 0.0) / n_steps,
            q_value,
        )

        io.save_weight(
            model=model,
            save_path=cfg.path.final_state_net,
        )

        current_epoch = epoch_no
        val_start = int(cfg.train.get("val_start_epoch", 10))
        val_interval = int(cfg.train.get("val_interval", 5))
        should_validate = (
            current_epoch >= val_start
            and (current_epoch - val_start) % val_interval == 0
        )

        if should_validate:
            if bool(cfg.train.get("save_val_ckpt", True)):
                io.save_weight(
                    model=model,
                    save_path=os.path.join(
                        cfg.path.pth,
                        f"state_epoch_{current_epoch:03d}.pth",
                    ),
                )

            old_save = cfg.save_results
            cfg.save_results = False
            base.test(
                model=model,
                cfg=cfg,
                epoch=current_epoch,
            )
            cfg.save_results = old_save

        counter.update_epoch_counter()

    cfg.tb_logger.close_tb()
    io.save_weight(
        model=model,
        save_path=cfg.path.final_state_net,
    )


def parse_cfg():
    parser = argparse.ArgumentParser(
        "Noisy-COD PNet P1 Source-Aware Dual Loss"
    )
    parser.add_argument(
        "--config",
        default="configs/pnet_noisycod_p1_dualloss.py",
    )
    parser.add_argument(
        "--data-cfg",
        default="./dataset.yaml",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/GT/Pnet",
    )
    parser.add_argument("--load-from", type=str)
    parser.add_argument(
        "--pretrained",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--save-results", action="store_true")
    parser.add_argument("--use-checkpoint", action="store_true")
    parser.add_argument(
        "--metric-names",
        nargs="+",
        default=["sm", "wfm", "mae", "em"],
    )
    args = parser.parse_args()

    cfg = Config.fromfile(args.config)
    cfg.merge_from_dict(vars(args))

    with open(
        cfg.data_cfg,
        "r",
        encoding="utf-8",
    ) as f:
        cfg.dataset_infos = yaml.safe_load(f)

    cfg.proj_root = os.path.dirname(os.path.abspath(__file__))
    cfg.model_name = "PvtV2B4_NoisyPNet"

    model_cfg = dict(cfg.get("model", {}))
    loss_tag = (
        "P1_DualLoss"
        if bool(model_cfg.get("source_aware_loss", False))
        else "P0_Original"
    )
    ratio_tag = int(round(float(cfg.train.full_ratio) * 100))
    cfg.exp_name = (
        f"{cfg.model_name}_{loss_tag}_F{ratio_tag}"
        f"_BF{int(cfg.train.batchsize_fully)}"
        f"_BW{int(cfg.train.batchsize_weakly)}"
    )

    cfg.output_dir = os.path.join(
        cfg.proj_root,
        cfg.output_dir,
    )
    cfg.path = py_utils.construct_path(
        output_dir=cfg.output_dir,
        exp_name=cfg.exp_name,
    )
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    py_utils.pre_mkdir(cfg.path)

    with open(
        cfg.path.cfg_copy,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(cfg.pretty_text)

    shutil.copy(
        __file__,
        cfg.path.trainer_copy,
    )

    file_handler = logging.FileHandler(cfg.path.log)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("[%(filename)s] %(message)s")
    )
    LOGGER.addHandler(file_handler)
    LOGGER.info(cfg.pretty_text)

    cfg.tb_logger = recorder.TBLogger(tb_root=cfg.path.tb)
    return cfg


def main():
    cfg = parse_cfg()

    pt_utils.initialize_seed_cudnn(
        seed=cfg.base_seed,
        deterministic=cfg.deterministic,
    )

    model = PvtV2B4_NoisyPNet(
        pretrained=cfg.pretrained,
        use_checkpoint=cfg.use_checkpoint,
        **dict(cfg.get("model", {})),
    )
    model.to(cfg.device)

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

    if not cfg.evaluate:
        train(model, cfg)

    if cfg.evaluate or cfg.has_test:
        base.test(
            model=model,
            cfg=cfg,
            epoch=None,
        )


if __name__ == "__main__":
    main()
