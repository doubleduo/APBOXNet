base_seed = 112358
config = 'configs/anet_convnextb_zoomnext.py'
data_cfg = 'dataset.yaml'
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
do_generate = True
exp_name = 'ConvNeXtB_ZoomNeXt_ANet_BS4'
generate = dict(
    batch_size=8,
    box_format='xyxy',
    data=dict(names=[
        'combined_tr',
    ], shape=dict(h=384, w=384)),
    num_workers=4)
load_from = 'outputs/shape/ConvNeXtB_ZoomNeXt_ANet_BS4/GT/pth/state_epoch_040.pth'
log_interval = 20
metric_names = [
    'sm',
    'wfm',
    'mae',
    'em',
]
model = dict(
    edge_loss_weight=4.0,
    hmu_groups=6,
    mid_dim=64,
    ual_full=1.0,
    ual_loss_weight=2.0,
    ual_start=0.0)
model_name = 'ConvNeXtB_ZoomNeXt_ANet'
model_type = 'anet'
output_dir = '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/shape'
path = dict(
    cfg_copy=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/shape/ConvNeXtB_ZoomNeXt_ANet_BS4/exp_1/config.py',
    excel=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/shape/ConvNeXtB_ZoomNeXt_ANet_BS4/exp_1/results.xlsx',
    final_full_net=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/shape/ConvNeXtB_ZoomNeXt_ANet_BS4/exp_1/pth/checkpoint_final.pth',
    final_state_net=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/shape/ConvNeXtB_ZoomNeXt_ANet_BS4/exp_1/pth/state_final.pth',
    log=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/shape/ConvNeXtB_ZoomNeXt_ANet_BS4/exp_1/log_2026-08-31.txt',
    output_dir=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/shape',
    pth=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/shape/ConvNeXtB_ZoomNeXt_ANet_BS4/exp_1/pth',
    pth_log=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/shape/ConvNeXtB_ZoomNeXt_ANet_BS4/exp_1',
    save=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/shape/ConvNeXtB_ZoomNeXt_ANet_BS4/exp_1/pre',
    tb=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/shape/ConvNeXtB_ZoomNeXt_ANet_BS4/exp_1/tb',
    trainer_copy=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/shape/ConvNeXtB_ZoomNeXt_ANet_BS4/exp_1/trainer.txt'
)
pretrained = True
proj_root = '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet'
train = dict(
    augment=True,
    backbone_lr_factor=0.1,
    batch_size=4,
    box_format='xyxy',
    clean_list='data/pseudo_pool/shape/clean.txt',
    data=dict(names=[
        'combined_tr',
    ], shape=dict(h=384, w=384)),
    grad_acc_step=1,
    lr=0.0001,
    min_lr_ratio=0.01,
    num_epochs=40,
    num_workers=8,
    optimizer='adamw',
    sample_ratio=1.0,
    save_interval=5,
    soft_target=False,
    target_key='pseudo_mask',
    use_amp=True,
    warmup_steps=500,
    weight_decay=0.0001)
use_checkpoint = False
use_custom_worker_init = True
val = dict(
    batch_size=8,
    data=dict(names=[
        'camo_te',
        'cod10k_te',
    ], shape=dict(h=384, w=384)),
    enable=True,
    interval=5,
    min_component_area=4,
    num_workers=4,
    save_vis_n=12,
    start_epoch=5)
