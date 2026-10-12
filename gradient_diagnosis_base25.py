#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Checkpoint-based gradient diagnosis for:
Stageout/PvtV2B4_FPN_Baseline_BS8/base(25)

Purpose
-------
Use saved checkpoints to generate the two-panel diagnostic figure:

(a) Gradient Conflict: Cosine Similarity
    cos(g_clean, g_noisy)

(b) Information Dilution: Gradient Magnitude
    ||g_clean||_2 and ||g_noisy||_2

This script does NOT retrain the model. It loads each saved checkpoint,
uses fixed Clean/Noisy diagnostic batches, computes BCE gradients, saves
the statistics to CSV, and exports PNG/PDF figures.

Recommended use:
    python gradient_diagnosis_base25.py \
        --repo-root . \
        --exp-dir "Stageout/PvtV2B4_FPN_Baseline_BS8/576(23.6)" \
        --epochs 30:150:5 \
        --batch-size 4 \
        --num-batches 8

If checkpoints are actually stored in the original exp_3 directory recorded
inside config.py, the script will automatically try cfg.path.pth as a fallback.
"""

import argparse
import csv
import math
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from mmengine import Config
from torch.utils.data import DataLoader

import basemain as core


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    parser.add_argument(
        "--exp-dir",
        default="Stageout/PvtV2B4_FPN_Baseline_BS8/base(25)",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--ckpt-dir", default=None)
    parser.add_argument(
        "--epochs",
        default="30:150:5",
        help="Examples: 30:150:5 or 30,50,60,90,95,120,150 or all",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument(
        "--num-batches",
        type=int,
        default=8,
        help="Number of fixed Clean/Noisy batch pairs per checkpoint.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--param-scope",
        choices=["all", "encoder", "decoder", "predictor"],
        default="all",
        help="Which parameters are used to form the gradient vectors.",
    )
    parser.add_argument("--seed", type=int, default=112358)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <exp-dir>/gradient_diagnosis",
    )
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_epoch_spec(spec, available_epochs):
    available_epochs = sorted(set(int(x) for x in available_epochs))
    if spec.strip().lower() == "all":
        return available_epochs

    if ":" in spec:
        parts = [int(x) for x in spec.split(":")]
        if len(parts) != 3:
            raise ValueError("--epochs range must be start:end:step")
        start, end, step = parts
        requested = list(range(start, end + 1, step))
    else:
        requested = [int(x.strip()) for x in spec.split(",") if x.strip()]

    found = [e for e in requested if e in available_epochs]
    missing = [e for e in requested if e not in available_epochs]
    if missing:
        print("[WARN] Missing requested checkpoint epochs:", missing)
    if not found:
        raise RuntimeError(
            "None of the requested epochs exist in the checkpoint directory."
        )
    return found


def discover_checkpoints(ckpt_dir):
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.is_dir():
        return {}

    pattern = re.compile(r"state_epoch_(\d+)\.pth$")
    found = {}
    for p in ckpt_dir.glob("state_epoch_*.pth"):
        m = pattern.search(p.name)
        if m:
            found[int(m.group(1))] = p
    return dict(sorted(found.items()))


def resolve_paths(args):
    repo_root = Path(args.repo_root).expanduser().resolve()
    os.chdir(repo_root)

    exp_dir = Path(args.exp_dir)
    if not exp_dir.is_absolute():
        exp_dir = (repo_root / exp_dir).resolve()

    cfg_path = Path(args.config) if args.config else exp_dir / "config.py"
    if not cfg_path.is_absolute():
        cfg_path = (repo_root / cfg_path).resolve()

    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    cfg = Config.fromfile(str(cfg_path))

    candidates = []
    if args.ckpt_dir:
        p = Path(args.ckpt_dir)
        if not p.is_absolute():
            p = (repo_root / p).resolve()
        candidates.append(p)

    # First try the visible base(25)/pth directory.
    candidates.append(exp_dir / "pth")

    # Then try the original path saved in this run's config.py.
    try:
        candidates.append(Path(str(cfg.path.pth)).expanduser())
    except Exception:
        pass

    ckpt_dir = None
    ckpts = {}
    for candidate in candidates:
        candidate = candidate.resolve()
        current = discover_checkpoints(candidate)
        if current:
            ckpt_dir = candidate
            ckpts = current
            break

    if not ckpts:
        lines = "\n".join(f"  - {p}" for p in candidates)
        raise FileNotFoundError(
            "No state_epoch_XXX.pth checkpoints were found.\n"
            "Checked:\n"
            f"{lines}\n\n"
            "This GitHub folder does not contain pth files, so this script must "
            "be run on the machine where the original checkpoints are stored. "
            "You can also pass --ckpt-dir /your/checkpoint/path."
        )

    out_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else exp_dir / "gradient_diagnosis"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    return repo_root, exp_dir, cfg_path, cfg, ckpt_dir, ckpts, out_dir


def build_model(cfg, device):
    model_class = core.model_zoo.__dict__.get(cfg.model_name)
    if model_class is None:
        raise KeyError(f"Unknown model_name in config: {cfg.model_name}")

    model_kwargs = dict(cfg.get("model", {}))

    # Checkpoint will overwrite the parameters, so avoid unnecessary pretrained
    # weight initialization/download during diagnosis.
    model = model_class(
        num_frames=1,
        pretrained=False,
        use_checkpoint=False,
        **model_kwargs,
    )

    # Match the baseline optimizer's freeze behavior:
    # get_grouped_params() freezes encoder.patch_embed1.* for this model.
    if hasattr(model, "get_grouped_params"):
        model.get_grouped_params()

    model.to(device)
    return model


def select_params(model, scope):
    selected = []
    selected_names = []

    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue

        if scope == "encoder" and not name.startswith("encoder."):
            continue
        if scope == "decoder" and name.startswith("encoder."):
            continue
        if scope == "predictor" and not name.startswith("predictor."):
            continue

        selected.append(p)
        selected_names.append(name)

    if not selected:
        raise RuntimeError(f"No trainable parameters found for scope={scope}")

    n_params = sum(p.numel() for p in selected)
    print(
        f"[PARAMS] scope={scope}, tensors={len(selected)}, "
        f"elements={n_params:,}"
    )
    return selected, selected_names


def build_diag_loaders(cfg, batch_size, num_workers):
    pools = cfg.train.curriculum.pools

    clean_ds = core._build_train_subset(
        cfg=cfg,
        list_path=pools.clean,
        pool_name="clean_diag",
        augment=False,
    )
    noisy_ds = core._build_train_subset(
        cfg=cfg,
        list_path=pools.noisy,
        pool_name="noisy_diag",
        augment=False,
    )

    # shuffle=False + augment=False => same diagnostic batches at every epoch.
    common = dict(
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    clean_loader = DataLoader(clean_ds, **common)
    noisy_loader = DataLoader(noisy_ds, **common)

    print(f"[DATA] Clean={len(clean_ds)}, Noisy={len(noisy_ds)}")
    print(
        f"[DATA] batches: Clean={len(clean_loader)}, "
        f"Noisy={len(noisy_loader)}"
    )
    return clean_loader, noisy_loader


def move_batch(batch, device):
    return core.pt_utils.to_device(batch["data"], device=device)


def segmentation_loss_eval_mode(model, data):
    """
    The baseline returns logits directly in eval mode.
    We compute exactly the same BCE loss used by PvtV2B4_FPN_Baseline.
    """
    logits = model(data=data)
    if isinstance(logits, dict):
        logits = logits.get("logits", logits.get("pred"))
    if not torch.is_tensor(logits):
        raise TypeError("Could not obtain logits tensor from model output.")

    mask = data["mask"].float()
    if mask.shape[-2:] != logits.shape[-2:]:
        mask = F.interpolate(mask, size=logits.shape[-2:], mode="nearest")

    return F.binary_cross_entropy_with_logits(
        logits,
        mask,
        reduction="mean",
    )


def get_gradients(model, data, params):
    model.zero_grad(set_to_none=True)
    loss = segmentation_loss_eval_mode(model, data)
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=False,
        create_graph=False,
        allow_unused=True,
    )
    grads = tuple(
        None if g is None else g.detach()
        for g in grads
    )
    return float(loss.detach().item()), grads


def grad_pair_stats(clean_grads, noisy_grads):
    # Accumulate without concatenating a giant 60M+ parameter vector.
    device = next(
        g.device for g in clean_grads if g is not None
    )

    dot = torch.zeros((), device=device, dtype=torch.float64)
    clean_sq = torch.zeros((), device=device, dtype=torch.float64)
    noisy_sq = torch.zeros((), device=device, dtype=torch.float64)

    used = 0
    for gc, gn in zip(clean_grads, noisy_grads):
        if gc is None or gn is None:
            continue
        gc = gc.double()
        gn = gn.double()
        dot += torch.sum(gc * gn)
        clean_sq += torch.sum(gc * gc)
        noisy_sq += torch.sum(gn * gn)
        used += 1

    clean_norm = torch.sqrt(clean_sq)
    noisy_norm = torch.sqrt(noisy_sq)
    denom = clean_norm * noisy_norm

    if used == 0 or denom.item() <= 1e-30:
        cosine = float("nan")
    else:
        cosine = float((dot / denom).item())

    return cosine, float(clean_norm.item()), float(noisy_norm.item())


def read_noisy_weight(exp_dir, epoch):
    schedule_path = Path(exp_dir) / "continuous_sampling_schedule.csv"
    if not schedule_path.is_file():
        return float("nan")

    with schedule_path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            if int(row["epoch"]) == int(epoch):
                return float(row["noisy_weight"])
    return float("nan")


def summarize(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)), float(np.nanstd(arr))


def load_checkpoint(model, ckpt_path):
    # Use the repository's own loader so checkpoint format matches training.
    core.io.load_weight(
        model=model,
        load_path=str(ckpt_path),
        strict=True,
    )


def diagnose_one_epoch(
    model,
    params,
    clean_loader,
    noisy_loader,
    epoch,
    ckpt_path,
    device,
    num_batches,
    noisy_weight,
):
    load_checkpoint(model, ckpt_path)

    # Eval mode removes DropPath / BN-state mutation, while gradients are still enabled.
    model.eval()

    cosines = []
    clean_norms = []
    noisy_norms = []
    clean_losses = []
    noisy_losses = []

    with torch.enable_grad():
        for batch_idx, (clean_batch, noisy_batch) in enumerate(
            zip(clean_loader, noisy_loader)
        ):
            if batch_idx >= num_batches:
                break

            clean_data = move_batch(clean_batch, device)
            noisy_data = move_batch(noisy_batch, device)

            clean_loss, clean_grads = get_gradients(
                model, clean_data, params
            )
            noisy_loss, noisy_grads = get_gradients(
                model, noisy_data, params
            )

            cosine, clean_norm, noisy_norm = grad_pair_stats(
                clean_grads, noisy_grads
            )

            cosines.append(cosine)
            clean_norms.append(clean_norm)
            noisy_norms.append(noisy_norm)
            clean_losses.append(clean_loss)
            noisy_losses.append(noisy_loss)

            # Release large gradient tuples before the next batch.
            del clean_grads, noisy_grads, clean_data, noisy_data
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if not cosines:
        raise RuntimeError("No diagnostic batch pairs were processed.")

    cos_mean, cos_std = summarize(cosines)
    cgn_mean, cgn_std = summarize(clean_norms)
    ngn_mean, ngn_std = summarize(noisy_norms)
    cl_mean, cl_std = summarize(clean_losses)
    nl_mean, nl_std = summarize(noisy_losses)

    effective_noisy = (
        ngn_mean * noisy_weight
        if math.isfinite(noisy_weight)
        else float("nan")
    )

    print(
        f"[EPOCH {epoch:03d}] "
        f"cos={cos_mean:+.4f}±{cos_std:.4f} | "
        f"||gC||={cgn_mean:.4f} | "
        f"||gN||={ngn_mean:.4f} | "
        f"wN={noisy_weight:.4f}"
    )

    return {
        "epoch": epoch,
        "checkpoint": str(ckpt_path),
        "num_batch_pairs": len(cosines),
        "cosine_mean": cos_mean,
        "cosine_std": cos_std,
        "clean_grad_norm_mean": cgn_mean,
        "clean_grad_norm_std": cgn_std,
        "noisy_grad_norm_mean": ngn_mean,
        "noisy_grad_norm_std": ngn_std,
        "noisy_weight": noisy_weight,
        "effective_noisy_grad_norm": effective_noisy,
        "clean_loss_mean": cl_mean,
        "clean_loss_std": cl_std,
        "noisy_loss_mean": nl_mean,
        "noisy_loss_std": nl_std,
    }


def save_csv(rows, path):
    fields = list(rows[0].keys())
    with Path(path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_figure(rows, out_dir):
    epochs = np.asarray([r["epoch"] for r in rows], dtype=float)

    cos_m = np.asarray([r["cosine_mean"] for r in rows], dtype=float)
    cos_s = np.asarray([r["cosine_std"] for r in rows], dtype=float)

    clean_m = np.asarray(
        [r["clean_grad_norm_mean"] for r in rows], dtype=float
    )
    clean_s = np.asarray(
        [r["clean_grad_norm_std"] for r in rows], dtype=float
    )
    noisy_m = np.asarray(
        [r["noisy_grad_norm_mean"] for r in rows], dtype=float
    )
    noisy_s = np.asarray(
        [r["noisy_grad_norm_std"] for r in rows], dtype=float
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.0))
    ax1, ax2 = axes

    # Left: cross-pool gradient alignment.
    ax1.plot(
        epochs,
        cos_m,
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label="Clean vs Noisy (mean)",
    )
    ax1.fill_between(
        epochs,
        cos_m - cos_s,
        cos_m + cos_s,
        alpha=0.15,
        label="Batch-pair std.",
    )
    ax1.axhline(
        0.0,
        linestyle="--",
        linewidth=1.2,
        label="y = 0 (conflict line)",
    )
    ax1.set_title(
        "(a) Gradient Conflict: Cosine Similarity",
        fontweight="bold",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cosine Similarity")
    ax1.grid(True, linestyle="--", alpha=0.3)
    ax1.legend(frameon=True, fontsize=8)

    # Right: gradient magnitudes.
    ax2.plot(
        epochs,
        clean_m,
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label=r"Clean Samples Norm ($||g_C||_2$)",
    )
    ax2.fill_between(
        epochs,
        np.maximum(0.0, clean_m - clean_s),
        clean_m + clean_s,
        alpha=0.12,
    )
    ax2.plot(
        epochs,
        noisy_m,
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label=r"Noisy Samples Norm ($||g_N||_2$)",
    )
    ax2.fill_between(
        epochs,
        np.maximum(0.0, noisy_m - noisy_s),
        noisy_m + noisy_s,
        alpha=0.12,
    )
    ax2.set_title(
        "(b) Information Dilution: Gradient Magnitude",
        fontweight="bold",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("L2 Norm of Gradient")
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.legend(frameon=True, fontsize=8)

    # base(25): noisy weight is exactly 1.0 through epoch 91,
    # and begins to decrease from epoch 92 in the saved schedule.
    reweight_epoch = 92
    if epochs.min() <= reweight_epoch <= epochs.max():
        for ax in axes:
            ax.axvline(reweight_epoch, linestyle=":", linewidth=1.0)
        ax1.annotate(
            "Noisy reweighting starts",
            xy=(reweight_epoch, np.interp(reweight_epoch, epochs, cos_m)),
            xytext=(reweight_epoch + 4, np.nanmax(cos_m)),
            arrowprops=dict(arrowstyle="->", linewidth=0.8),
            fontsize=8,
        )

    fig.tight_layout()
    png_path = Path(out_dir) / "gradient_conflict_magnitude_base25.png"
    pdf_path = Path(out_dir) / "gradient_conflict_magnitude_base25.pdf"
    fig.savefig(png_path, dpi=600, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    return png_path, pdf_path


def main():
    args = parse_args()
    seed_everything(args.seed)

    (
        repo_root,
        exp_dir,
        cfg_path,
        cfg,
        ckpt_dir,
        ckpts,
        out_dir,
    ) = resolve_paths(args)

    print(f"[REPO]   {repo_root}")
    print(f"[EXP]    {exp_dir}")
    print(f"[CONFIG] {cfg_path}")
    print(f"[CKPTS]  {ckpt_dir}")
    print(f"[OUT]    {out_dir}")

    cfg.device = args.device
    # Make relative pool/data paths resolve from the current checkout.
    cfg.proj_root = str(repo_root)

    epochs = parse_epoch_spec(args.epochs, ckpts.keys())
    print("[EPOCHS]", epochs)

    device = torch.device(args.device)
    model = build_model(cfg, device)
    params, _ = select_params(model, args.param_scope)

    clean_loader, noisy_loader = build_diag_loaders(
        cfg=cfg,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    rows = []
    for epoch in epochs:
        noisy_weight = read_noisy_weight(exp_dir, epoch)
        row = diagnose_one_epoch(
            model=model,
            params=params,
            clean_loader=clean_loader,
            noisy_loader=noisy_loader,
            epoch=epoch,
            ckpt_path=ckpts[epoch],
            device=device,
            num_batches=args.num_batches,
            noisy_weight=noisy_weight,
        )
        rows.append(row)

    csv_path = out_dir / "gradient_diagnosis_base25.csv"
    save_csv(rows, csv_path)

    png_path, pdf_path = plot_figure(rows, out_dir)

    print("\n[DONE]")
    print("CSV :", csv_path)
    print("PNG :", png_path)
    print("PDF :", pdf_path)


if __name__ == "__main__":
    main()
