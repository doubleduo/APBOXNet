base_seed = 112358
config = 'configs/anet_ratio.py'
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
        mask=dict(path='./data/Train/GT', suffix='.png'),
        pseudo_mask=dict(path='./data/Train/GT', suffix='.png'),
        root='.'),
    nc4k=dict(
        image=dict(path='Image', suffix='.jpg'),
        mask=dict(path='Mask', suffix='.png'),
        root='data/Test/NC4K'))
deterministic = True
device = 'cuda:0'
exp_name = 'ConvNeXtB_ZoomNeXt_ANet_BS4'
generate = False
load_from = None
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
output_dir = '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/ratio/20'
path = dict(
    cfg_copy=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/ratio/20/ConvNeXtB_ZoomNeXt_ANet_BS4/exp_0/config.py',
    excel=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/ratio/20/ConvNeXtB_ZoomNeXt_ANet_BS4/exp_0/results.xlsx',
    final_full_net=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/ratio/20/ConvNeXtB_ZoomNeXt_ANet_BS4/exp_0/pth/checkpoint_final.pth',
    final_state_net=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/ratio/20/ConvNeXtB_ZoomNeXt_ANet_BS4/exp_0/pth/state_final.pth',
    log=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/ratio/20/ConvNeXtB_ZoomNeXt_ANet_BS4/exp_0/log_2026-09-03.txt',
    output_dir=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/ratio/20',
    pth=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/ratio/20/ConvNeXtB_ZoomNeXt_ANet_BS4/exp_0/pth',
    pth_log=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/ratio/20/ConvNeXtB_ZoomNeXt_ANet_BS4/exp_0',
    save=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/ratio/20/ConvNeXtB_ZoomNeXt_ANet_BS4/exp_0/pre',
    tb=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/ratio/20/ConvNeXtB_ZoomNeXt_ANet_BS4/exp_0/tb',
    trainer_copy=
    '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet/outputs/ratio/20/ConvNeXtB_ZoomNeXt_ANet_BS4/exp_0/trainer.txt'
)
pretrained = True
proj_root = '/media/cc/b5d7f4aa-fba2-44c6-9e72-aabbee8d6a73/AAA_base_hjw/usefulmodel/APboxnet'
train = dict(
    augment=True,
    backbone_lr_factor=0.1,
    batch_size=4,
    box_format='xyxy',
    data=dict(names=[
        'combined_tr',
    ], shape=dict(h=384, w=384)),
    grad_acc_step=1,
    lr=0.0001,
    min_lr_ratio=0.01,
    num_epochs=40,
    num_workers=8,
    optimizer='adamw',
    sample_ratio=0.2,
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
