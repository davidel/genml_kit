"""Shared augmentation utilities (classification + self-supervised objectives).

Moved out of ``genml_kit.pretrain`` (v4.2): with the unified
``genml-kit-train`` CLI there is no separate "pretrain" program, so the
augmentation library is top-level.  These are objective-level transforms
returned by ``Method.build_transform``.
"""
