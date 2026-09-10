# -*- coding: utf-8 -*-
"""
C32: 60 + 80 + 30

Stage I:
    epoch 1-60
    Clean:Noisy weight = 1.0:1.0

Stage II:
    epoch 61-140
    noisy weight cosine 1.0 -> 0.0

Stage III:
    global epoch 141-170
    CLEAN-only promotion

Set promotion.mode to:
    "clean"      : all Clean, no SBFT
    "acp"        : hard Clean top-20%, no SBFT
    "sbft"       : all Clean + SBFT
    "acp_sbft"   : hard Clean top-20% + SBFT
"""

has_test = True
deterministic = True
use_custom_worker_init = True

log_interval = 20
base_seed = 112358

__BATCHSIZE = 8

# Stage I + II only.
__PHASE12_EPOCHS = 140
__SAMPLES_PER_EPOCH = 4040
__ITER_PER_EPOCH = __SAMPLES_PER_EPOCH // __BATCHSIZE
__NUM_ITERS = __PHASE12_EPOCHS * __ITER_PER_EPOCH


train = dict(
    batch_size=__BATCHSIZE,
    num_workers=4,
    use_amp=True,

    # Main curriculum stops exactly at 140.
    num_epochs=__PHASE12_EPOCHS,
    epoch_based=True,
    num_iters=None,

    lr=1e-4,
    grad_acc_step=1,

    val_start_epoch=30,
    val_interval=5,
    save_val_ckpt=True,

    # Keep all old B2 confounds OFF.
    ema_kd=dict(
        enable=False,
        lambda_kd=0.0,
    ),

    curriculum=dict(
        enable=True,

        pools=dict(
            clean="./data/pseudo_pool/shape/clean.txt",
            noisy="./data/pseudo_pool/shape/noisy.txt",
        ),

        num_samples_per_epoch=__SAMPLES_PER_EPOCH,

        reliability=dict(
            clean_weight=1.0,

            # 1-60 = 1:1
            broad_end_epoch=60,
            noisy_start_weight=1.0,

            # 61-140 cosine 1 -> 0
            anneal_end_epoch=140,
            noisy_end_weight=0.0,
        ),

        # ------------------------------------------------------
        # OFF by default for the clean 60+80+30 validation.
        # Enable later for PESF-style PSCL ablation.
        # ------------------------------------------------------
        difficulty=dict(
            enable=False,

            # If enabled:
            source="mask_area",   # or "csv"
            csv=None,             # e.g. difficulty_box.csv
            start_epoch=1,
            end_epoch=120,
            min_fraction=0.60,
            pacing="log",

            # Gaussian perturbs ONLY difficulty boundary.
            gaussian_enable=False,
            gaussian_alpha=0.10,
            gaussian_std=0.10,
        ),
    ),

    # ----------------------------------------------------------
    # Stage III: 141-170
    # Change ONLY mode for T0/T1/T2/T3 ablation.
    # ----------------------------------------------------------
    promotion=dict(
        enable=True,

        # Recommended first run:
        #   "acp"
        #
        # Alternatives:
        #   "clean"
        #   "sbft"
        #   "acp_sbft"
        mode="sbft",

        epochs=30,

        # ACP only uses this when mode contains "acp".
        hard_ratio=0.20,

        # Independent fine-tuning optimizer for all Stage-III modes.
        lr=1e-5,

        eval_batch_size=8,
        val_interval=5,

        # Used only for sbft/acp_sbft.
        sbft_prob_start=0.10,
        sbft_prob_end=0.50,
        sbft_kernel=11,
        sbft_sigma=3.0,
    ),

    optimizer=dict(
        mode="adam",
        set_to_none=False,
        group_mode="finetune",
        cfg=dict(
            weight_decay=0,
            diff_factor=0.1,
        ),
    ),

    sche_usebatch=True,

    # IMPORTANT:
    # no LR drop inside 1-140, so reliability annealing is not
    # confounded with the old epoch-120 LR step.
    scheduler=dict(
        warmup=dict(
            num_iters=0,
            initial_coef=0.01,
            mode="linear",
        ),
        mode="step",
        cfg=dict(
            milestones=__NUM_ITERS + 1,
            gamma=0.1,
        ),
    ),

    bn=dict(
        freeze_status=True,
        freeze_affine=True,
        freeze_encoder=False,
    ),

    data=dict(
        shape=dict(h=384, w=384),
        names=["combined_tr"],
    ),
)


test = dict(
    batch_size=__BATCHSIZE,
    num_workers=8,
    clip_range=None,

    data=dict(
        shape=dict(h=384, w=384),
        names=[
            "camo_te",
            "cod10k_te",
        ],
    ),
)
