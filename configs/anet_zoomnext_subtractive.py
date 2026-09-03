# Original ANetZoomNeXt subtractive ablations.
# Only change this:
experiment = "Z8"

notes = dict(
    Z0="Full original ANetZoomNeXt",
    Z1="w/o Box prompt information; second encoder receives RGB instead of RGB*Box",
    Z2="w/o Box branch contribution; box feature is zero and box encoder forward is skipped",
    Z3="w/o top-down cross-scale propagation between RGPU stages",
    Z4="w/o deep mask supervision; only final prediction gets structure loss",
    Z5="w/o edge supervision",
    Z6="w/o UAL",
    Z7="w/o deepest SimpleASPP; use plain 3x3 ConvBNReLU",
    Z8="w/o RGPU; replace each RGPU with plain 3x3 ConvBNReLU",
)

base_seed = 112358
deterministic = True
use_custom_worker_init = True
log_interval = 20

model = dict(
    ablation=experiment,
    mid_dim=64,
    hmu_groups=6,
    edge_loss_weight=4.0,
    ual_loss_weight=2.0,
    ual_start=0.0,
    ual_full=1.0,
)

# Keep these exactly the same as your current strong ANet run.
train = dict(
    batch_size=4,
    num_workers=8,
    use_amp=True,
    num_epochs=40,
    grad_acc_step=1,
    sample_ratio=1.0,
    target_key="pseudo_mask",
    clean_list="data/pseudo_pool/shape/clean.txt",
    soft_target=False,
    augment=True,
    box_format="xyxy",
    lr=1e-4,
    backbone_lr_factor=0.1,
    weight_decay=1e-4,
    optimizer="adamw",
    warmup_steps=500,
    min_lr_ratio=0.01,
    save_interval=5,
    data=dict(shape=dict(h=384, w=384), names=["combined_tr"]),
)

generate = dict(
    batch_size=8,
    num_workers=4,
    box_format="xyxy",
    data=dict(shape=dict(h=384, w=384), names=["combined_tr"]),
)

val = dict(
    enable=True,
    start_epoch=5,
    interval=5,
    batch_size=8,
    num_workers=4,
    save_vis_n=12,
    min_component_area=4,
    data=dict(shape=dict(h=384, w=384), names=["camo_te", "cod10k_te"]),
)

metric_names = ["sm", "wfm", "mae", "em"]
