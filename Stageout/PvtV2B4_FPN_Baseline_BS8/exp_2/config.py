base_seed = 112358
config = 'configs/C31/icod_c32_T3_acp_sbft.py'
data_cfg = './dataset.yaml'
dataset_infos = dict(
    camo_te=dict(
        image=dict(path='Image', suffix='.jpg'),
        mask=dict(path='Mask', suffix='.png'),
        root='data/Test/CAMO-TE'),
    chameleon=dict(
        image=dict(path='Image', suffix='.jpg'),
        mask=dict(path='Mask', suffix='.png'),
        root='data/Test/CHAMELEON'),
    cod10k_te=dict(
        image=dict(path='Image', suffix='.jpg'),
        mask=dict(path='Mask', suffix='.png'),
        root='data/Test/COD10K-TE'),
    combined_tr=dict(
        box_json=dict(path='./data/Train/box_label', suffix='.json'),
        image=dict(path='./data/Train/Imgs', suffix='.jpg'),
        mask=dict(path='./data/Train/S1_GT', suffix='.png'),
        pseudo_mask=dict(path='./data/Train/S1_GT', suffix='.png'),
        root='.'),
    nc4k=dict(
        image=dict(path='Image', suffix='.jpg'),
        mask=dict(path='Mask', suffix='.png'),
        root='data/Test/NC4K'))
deterministic = True
device = 'cuda:0'
evaluate = False
exp_name = 'PvtV2B4_FPN_Baseline_BS8'
has_test = True
info = None
load_from = None
log_interval = 20
metric_names = [
    'sm',
    'wfm',
    'mae',
    'em',
]
model_name = 'PvtV2B4_FPN_Baseline'
output_dir = '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout'
path = dict(
    cfg_copy=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout/PvtV2B4_FPN_Baseline_BS8/exp_2/config.py',
    excel=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout/PvtV2B4_FPN_Baseline_BS8/exp_2/results.xlsx',
    final_full_net=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout/PvtV2B4_FPN_Baseline_BS8/exp_2/pth/checkpoint_final.pth',
    final_state_net=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout/PvtV2B4_FPN_Baseline_BS8/exp_2/pth/state_final.pth',
    log=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout/PvtV2B4_FPN_Baseline_BS8/exp_2/log_2026-09-10.txt',
    output_dir=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout',
    pth=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout/PvtV2B4_FPN_Baseline_BS8/exp_2/pth',
    pth_log=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout/PvtV2B4_FPN_Baseline_BS8/exp_2',
    save=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout/PvtV2B4_FPN_Baseline_BS8/exp_2/pre',
    tb=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout/PvtV2B4_FPN_Baseline_BS8/exp_2/tb',
    trainer_copy=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout/PvtV2B4_FPN_Baseline_BS8/exp_2/trainer.txt'
)
pretrained = True
proj_root = '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet'
save_results = False
test = dict(
    batch_size=8,
    clip_range=None,
    data=dict(names=[
        'camo_te',
        'cod10k_te',
    ], shape=dict(h=384, w=384)),
    num_workers=8)
train = dict(
    batch_size=8,
    bn=dict(freeze_affine=True, freeze_encoder=False, freeze_status=True),
    curriculum=dict(
        difficulty=dict(
            csv=None,
            enable=False,
            end_epoch=120,
            gaussian_alpha=0.1,
            gaussian_enable=False,
            gaussian_std=0.1,
            min_fraction=0.6,
            pacing='log',
            source='mask_area',
            start_epoch=1),
        enable=True,
        num_samples_per_epoch=4040,
        pools=dict(
            clean='./data/pseudo_pool/shape/clean.txt',
            noisy='./data/pseudo_pool/shape/noisy.txt'),
        reliability=dict(
            anneal_end_epoch=140,
            broad_end_epoch=60,
            clean_weight=1.0,
            noisy_end_weight=0.0,
            noisy_start_weight=1.0)),
    data=dict(names=[
        'combined_tr',
    ], shape=dict(h=384, w=384)),
    ema_kd=dict(enable=False, lambda_kd=0.0),
    epoch_based=True,
    grad_acc_step=1,
    lr=0.0001,
    num_epochs=140,
    num_iters=None,
    num_workers=4,
    optimizer=dict(
        cfg=dict(diff_factor=0.1, weight_decay=0),
        group_mode='finetune',
        mode='adam',
        set_to_none=False),
    promotion=dict(
        enable=True,
        epochs=30,
        eval_batch_size=8,
        hard_ratio=0.2,
        lr=1e-05,
        mode='acp_sbft',
        sbft_kernel=11,
        sbft_prob_end=0.5,
        sbft_prob_start=0.1,
        sbft_sigma=3.0,
        val_interval=5),
    save_val_ckpt=True,
    sche_usebatch=True,
    scheduler=dict(
        cfg=dict(gamma=0.1, milestones=70701),
        mode='step',
        warmup=dict(initial_coef=0.01, mode='linear', num_iters=0)),
    use_amp=True,
    val_interval=5,
    val_start_epoch=30)
use_checkpoint = False
use_custom_worker_init = True
