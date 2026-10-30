"""Batch/contract types shared by all pipelines."""

import collections

# One DataLoader step.  data is opaque to the loop; meta carries anything
# the method's train_step needs (labels, GT similarity, ...).
#
# ``records`` carries pipeline-side, method-consumable metric records (e.g.
# work-amount facts such as the batch's sample/token count).  The method folds
# them into ``LossOutput.metrics`` so pipeline facts reach the metric sinks
# through the one existing channel.  It defaults to ``None`` so every existing
# constructor keeps working unchanged; code that *modifies* a blob (notably
# ``to_device``) must use ``blob._replace(...)`` so this field -- and any field
# added later -- flows through untouched.
DataBlob = collections.namedtuple("DataBlob", ["data", "meta", "records"],
                                  defaults=[None])

# NOTE: do NOT define a generic "ModelOutput" here -- the name is taken by
# genml_kit.models.registry.ModelOutput (holds .logits, used by every custom
# model loader).  The loop passes the raw `model(...)` return straight to
# method.train_step, which decodes its own shape (dict for VO, tensor for
# classification, tuple for SimMIM, ...).

# What method.train_step returns.  loss is the (already averaged-across-the-batch),
# UNSCALED objective; metrics is dict[str, torch.Tensor] for logging.
# For PER: td_errors (optional) contains per-sample TD errors for priority updates.
LossOutput = collections.namedtuple("LossOutput", ["loss", "metrics", "td_errors"],
                                    defaults=[None])

__all__ = ["DataBlob", "LossOutput"]
