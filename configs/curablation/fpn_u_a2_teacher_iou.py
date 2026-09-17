"""A2: A1 plus a low-weight Soft-IoU term on clean masks."""

_base_ = ["./fpn_u_a1_teacher.py"]

model = dict(
    clean_iou_weight=0.20,
)

