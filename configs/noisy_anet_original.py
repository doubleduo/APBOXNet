# Noisy-COD original-style ANet adapted for APBOXNet.

base_seed = 112358
deterministic = True
use_custom_worker_init = True
log_interval = 20
model_type = "noisy_anet"

model = dict(
    channels=64,
    input_norm=True,
    edge_loss_weight=4.0,
    ual_loss_weight=2.0,
    copy_box_encoder_init=True,
)

train = dict(
    batch_size=4,
    num_workers=8,
    use_amp=True,
    num_epochs=40,
    grad_acc_step=1,

    # "mask" for true dense GT; "pseudo_mask" for your clean pseudo pool.
    target_key="pseudo_mask",
    clean_list="data/pseudo_pool/Top400.txt",
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

    data=dict(
        shape=dict(h=384, w=384),
        names=["combined_tr"],
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

# Same ANet validation protocol as APBOXNet/anet_main.py:
# Test GT is used only to derive box prompts and compute metrics.
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

metric_names = ["sm", "wfm", "mae", "em"]
