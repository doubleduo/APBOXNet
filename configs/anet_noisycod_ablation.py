# -*- coding: utf-8 -*-
"""
One configuration file for all A0-A8 ANet ablations.

Only change:
    experiment = "A0"  -> "A1"... "A8"

Recommended first:
    A0 A1 A2 A3
"""

# ---------------------------------------------------------------------------
# experiment
# ---------------------------------------------------------------------------
experiment = "A8"

ablation_note = dict(
    A0="Noisy-COD base: Box + DWT(HH/LL) + GPM + Edge + UAL",
    A1="w/o Box prompt: second branch receives RGB, capacity unchanged",
    A2="w/o DWT: keep ETM/GCM3 projections but remove frequency injection",
    A3="w/o GPM: replace global prior with simple 1x1 prior from deepest fused feature",
    A4="w/o Edge loss",
    A5="w/o UAL",
    A6="All high-frequency: aggregate LH+HL+HH instead of HH only",
    A7="Relation fusion: [R,B,|R-B|,R*B] -> 1x1",
    A8="A6 + A7",
)

base_seed = 112358
deterministic = True
use_custom_worker_init = True
log_interval = 20

model = dict(
    ablation=experiment,
    channels=64,
    input_norm=True,
    edge_loss_weight=4.0,
    ual_loss_weight=2.0,
    ual_start=0.0,
    ual_full=1.0,
)

train = dict(
    batch_size=4,
    num_workers=8,
    use_amp=True,
    num_epochs=40,
    grad_acc_step=1,

    # ---- your supervision ----
    # For strict Noisy-COD reproduction use target_key="mask".
    # For your clean pseudo subset use target_key="pseudo_mask".
    target_key="pseudo_mask",
    clean_list="data/pseudo_pool/shape/clean.txt",
    soft_target=False,
    sample_ratio=1.0,

    augment=True,
    box_format="xyxy",

    optimizer="adamw",
    lr=1e-4,
    backbone_lr_factor=0.1,
    weight_decay=1e-4,
    warmup_steps=500,
    min_lr_ratio=0.01,

    save_interval=5,
    data=dict(
        shape=dict(h=384, w=384),
        names=["combined_tr"],
    ),
)

val = dict(
    enable=True,
    start_epoch=5,
    interval=5,
    batch_size=8,
    num_workers=4,
    save_vis_n=12,
    min_component_area=4,
    data=dict(
        shape=dict(h=384, w=384),
        names=["camo_te", "cod10k_te"],
    ),
)

generate = dict(
    batch_size=8,
    num_workers=4,
    box_format="xyxy",
    data=dict(
        shape=dict(h=384, w=384),
        names=["combined_tr"],
    ),
)

metric_names = ["sm", "wfm", "mae", "em"]
