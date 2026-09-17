base_seed = 112358
config = 'configs/curablation/fpn_u_a2_teacher_iou.py'
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
exp_name = 'PvtV2B4_FPN_Unvalue_BS8_INFOa2_teacher_iou'
has_test = True
info = 'A2_teacher_iou'
load_from = None
log_interval = 20
metric_names = [
    'sm',
    'wfm',
    'mae',
    'em',
]
model = dict(
    aux_p3_weight=0.0,
    box_dilate_kernel=9,
    clean_iou_weight=0.2,
    consistency_weight=0.05,
    decoder_norm='bn',
    dynamic_loss_mode='normalized_abs',
    dynamic_mix_max=0.5,
    dynamic_weight=0.25,
    fpn_dim=64,
    gn_groups=8,
    min_teacher_foreground=16,
    noisy_mask_weight=1.0,
    outside_weight=0.2,
    teacher_confidence=0.9,
    teacher_disagreement=0.05,
    teacher_target_mode='teacher')
model_name = 'PvtV2B4_FPN_Unvalue'
output_dir = '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout'
path = dict(
    cfg_copy=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout/PvtV2B4_FPN_Unvalue_BS8_INFOa2_teacher_iou/exp_1/config.py',
    excel=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout/PvtV2B4_FPN_Unvalue_BS8_INFOa2_teacher_iou/exp_1/results.xlsx',
    final_full_net=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout/PvtV2B4_FPN_Unvalue_BS8_INFOa2_teacher_iou/exp_1/pth/checkpoint_final.pth',
    final_state_net=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout/PvtV2B4_FPN_Unvalue_BS8_INFOa2_teacher_iou/exp_1/pth/state_final.pth',
    log=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout/PvtV2B4_FPN_Unvalue_BS8_INFOa2_teacher_iou/exp_1/log_2026-09-17.txt',
    output_dir=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout',
    pth=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout/PvtV2B4_FPN_Unvalue_BS8_INFOa2_teacher_iou/exp_1/pth',
    pth_log=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout/PvtV2B4_FPN_Unvalue_BS8_INFOa2_teacher_iou/exp_1',
    save=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout/PvtV2B4_FPN_Unvalue_BS8_INFOa2_teacher_iou/exp_1/pre',
    tb=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout/PvtV2B4_FPN_Unvalue_BS8_INFOa2_teacher_iou/exp_1/tb',
    trainer_copy=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/Stageout/PvtV2B4_FPN_Unvalue_BS8_INFOa2_teacher_iou/exp_1/trainer.txt'
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
        batch_schedule=[
            dict(
                batch=dict(clean=4, noisy=4, unvalue=0),
                end_epoch=40,
                name='mask_warmup'),
            dict(
                batch=dict(clean=4, noisy=3, unvalue=1),
                end_epoch=60,
                name='box_introduction'),
            dict(
                batch=dict(clean=3, noisy=3, unvalue=2),
                end_epoch=100,
                name='teacher_recovery'),
            dict(
                batch=dict(clean=4, noisy=2, unvalue=2),
                end_epoch=120,
                name='hard_consolidation'),
            dict(
                batch=dict(clean=5, noisy=2, unvalue=1),
                end_epoch=150,
                name='reliable_finetune'),
        ],
        continuous_schedule=dict(enable=True),
        enable=True,
        num_samples_per_epoch=4040,
        pools=dict(
            clean='./data/pseudo_pool/shape/clean.txt',
            noisy='./data/pseudo_pool/shape/noisy.txt',
            unvalue='./data/pseudo_pool/shape/unvalue.txt')),
    data=dict(names=[
        'combined_tr',
    ], shape=dict(h=384, w=384), use_box=True),
    epoch_based=True,
    grad_acc_step=1,
    lr=0.0001,
    num_epochs=150,
    num_iters=None,
    num_workers=4,
    optimizer=dict(
        cfg=dict(diff_factor=0.1, weight_decay=0),
        group_mode='finetune',
        mode='adam',
        set_to_none=False),
    save_val_ckpt=True,
    sche_usebatch=True,
    scheduler=dict(
        cfg=dict(gamma=0.1, milestones=60600),
        mode='step',
        warmup=dict(initial_coef=0.01, mode='linear', num_iters=0)),
    unvalue_teacher=dict(
        ema_decay=0.99, freeze_epoch=120, hflip=True, start_epoch=41),
    use_amp=True,
    val_interval=5,
    val_start_epoch=30)
use_checkpoint = False
use_custom_worker_init = True
