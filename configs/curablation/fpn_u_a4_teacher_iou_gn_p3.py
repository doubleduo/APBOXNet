"""A4: A3 plus clean-only P3 auxiliary supervision."""

_base_ = ["./fpn_u_a3_teacher_iou_gn.py"]

model = dict(
    aux_p3_weight=0.10,
)

