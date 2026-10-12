base_seed = 112358
config = 'configs/pnet_noisycod_p1_dualloss.py'
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
        mask=dict(path='./data/Train/GT', suffix='.png'),
        pseudo_mask=dict(path='./data/Train/GT', suffix='.png'),
        root='.'),
    nc4k=dict(
        image=dict(path='Image', suffix='.jpg'),
        mask=dict(path='Mask', suffix='.png'),
        root='data/Test/NC4K'))
deterministic = True
device = 'cuda:0'
evaluate = False
exp_name = 'PvtV2B4_NoisyPNet_P1_DualLoss_F1_BF2_BW8'
has_test = True
load_from = None
log_interval = 20
metric_names = [
    'sm',
    'wfm',
    'mae',
    'em',
]
model = dict(
    channels=64,
    full_loss_type='structure',
    full_loss_weight=1.0,
    mask_level_weights=(
        0.0625,
        0.125,
        0.25,
        0.5,
        1.0,
    ),
    source_aware_loss=True,
    ual_weight=2.0,
    weak_loss_type='nc',
    weak_loss_weight=1.0)
model_name = 'PvtV2B4_NoisyPNet'
output_dir = '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/PNet_outputs'
path = dict(
    cfg_copy=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/PNet_outputs/PvtV2B4_NoisyPNet_P1_DualLoss_F1_BF2_BW8/exp_0/config.py',
    excel=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/PNet_outputs/PvtV2B4_NoisyPNet_P1_DualLoss_F1_BF2_BW8/exp_0/results.xlsx',
    final_full_net=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/PNet_outputs/PvtV2B4_NoisyPNet_P1_DualLoss_F1_BF2_BW8/exp_0/pth/checkpoint_final.pth',
    final_state_net=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/PNet_outputs/PvtV2B4_NoisyPNet_P1_DualLoss_F1_BF2_BW8/exp_0/pth/state_final.pth',
    log=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/PNet_outputs/PvtV2B4_NoisyPNet_P1_DualLoss_F1_BF2_BW8/exp_0/log_2026-08-31.txt',
    output_dir=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/PNet_outputs',
    pth=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/PNet_outputs/PvtV2B4_NoisyPNet_P1_DualLoss_F1_BF2_BW8/exp_0/pth',
    pth_log=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/PNet_outputs/PvtV2B4_NoisyPNet_P1_DualLoss_F1_BF2_BW8/exp_0',
    save=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/PNet_outputs/PvtV2B4_NoisyPNet_P1_DualLoss_F1_BF2_BW8/exp_0/pre',
    tb=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/PNet_outputs/PvtV2B4_NoisyPNet_P1_DualLoss_F1_BF2_BW8/exp_0/tb',
    trainer_copy=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/PNet_outputs/PvtV2B4_NoisyPNet_P1_DualLoss_F1_BF2_BW8/exp_0/trainer.txt'
)
pretrained = True
proj_root = '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet'
save_results = False
test = dict(
    batch_size=16,
    clip_range=None,
    data=dict(
        names=[
            'chameleon',
            'camo_te',
            'cod10k_te',
            'nc4k',
        ],
        shape=dict(h=384, w=384)),
    num_workers=8)
train = dict(
    augment=True,
    batchsize_fully=2,
    batchsize_weakly=8,
    data=dict(names=[
        'combined_tr',
    ], shape=dict(h=384, w=384)),
    full_ratio=0.01,
    grad_acc_step=1,
    lr=0.0001,
    num_epochs=100,
    num_workers=4,
    optimizer=dict(
        cfg=dict(diff_factor=1.0, weight_decay=0),
        group_mode='finetune',
        mode='adam',
        set_to_none=False),
    pseudo_edge_root='ANet_outputs/pseudo_edge',
    pseudo_mask_root='ANet_outputs/pseudo_mask',
    pseudo_suffix='.png',
    q_epoch=60,
    save_val_ckpt=True,
    sche_usebatch=True,
    scheduler=dict(
        cfg=dict(gamma=0.1, milestones='auto'),
        mode='step',
        warmup=dict(initial_coef=0.01, mode='linear', num_iters=0)),
    split_seed=112358,
    steps_mode='use_all_weak',
    use_amp=True,
    val_interval=5,
    val_start_epoch=10)
use_checkpoint = False
use_custom_worker_init = True
