"""A0: paired rerun of the original model/loss behavior."""

_base_ = ["./fpn_nc_unvalue_improved.py"]

model = dict(
    decoder_norm="bn",
    clean_iou_weight=0.0,
    aux_p3_weight=0.0,
    teacher_target_mode="blend",
    dynamic_loss_mode="normalized_abs",
)

