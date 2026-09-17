"""A1: only allow reliable teacher pixels to correct unvalue targets."""

_base_ = ["./fpn_u_a0_original.py"]

model = dict(
    teacher_target_mode="teacher",
)

