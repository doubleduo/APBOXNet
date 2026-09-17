"""A3: A2 plus GroupNorm in the FPN decoder."""

_base_ = ["./fpn_u_a2_teacher_iou.py"]

model = dict(
    decoder_norm="gn",
    gn_groups=8,
)

