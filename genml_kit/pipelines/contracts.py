"""Batch/contract types shared by all pipelines."""

import collections

# One DataLoader step.  data is opaque to the loop; meta carries anything
# the method's train_step needs (labels, GT similarity, ...).
DataBlob = collections.namedtuple("DataBlob", ["data", "meta"])

# NOTE: do NOT define a generic "ModelOutput" here -- the name is taken by
# genml_kit.models.registry.ModelOutput (holds .logits, used by every custom
# model loader).  The loop passes the raw `model(...)` return straight to
# method.train_step, which decodes its own shape (dict for VO, tensor for
# classification, tuple for SimMIM, ...).

# What method.train_step returns.  loss is the (already averaged-across-the-batch),
# UNSCALED objective; metrics is dict[str, torch.Tensor] for logging.
LossOutput = collections.namedtuple("LossOutput", ["loss", "metrics"])

__all__ = ["DataBlob", "LossOutput"]
