"""Batch/contract types shared by all pipelines."""

import collections

# One DataLoader step.  data is opaque to the loop; meta carries anything
# the method's train_step needs (labels, GT similarity, ...).
#
# ``size`` is the batch's sample count (the batch dimension), published by the
# collator so consumers never re-derive it by sniffing the shape of ``data``
# (which may be a tensor, a tuple of views, or a dict of tensors).  It defaults
# to ``None`` for the few synthetic/test constructors that do not know it.
#
# ``records`` carries pipeline-side, method-consumable metric records (e.g.
# work-amount facts).  The method folds them into ``LossOutput.metrics`` so
# pipeline facts reach the metric sinks through the one existing channel.  It
# defaults to ``None``.
#
# Code that *modifies* a blob (notably ``to_device``) must use
# ``blob._replace(...)`` so every field -- and any field added later -- flows
# through untouched.
DataBlob = collections.namedtuple("DataBlob", ["data", "meta", "size", "records"],
                                  defaults=[None, None])

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

# What method.evaluate returns.  ``metrics`` are this rank's validation scalars
# (logged verbatim on a single rank); ``weight`` is the sample count the rank
# saw (the denominator for a sample-weighted merge across shards);
# ``accumulators`` carries opaque per-rank state for non-additive metrics
# (confusion matrices, prediction lists, PR curves), merged verbatim by
# ``Method.merge_validation``.
ValidationResult = collections.namedtuple("ValidationResult",
                                          ["metrics", "weight", "accumulators"],
                                          defaults=[0.0, None])

__all__ = ["DataBlob", "LossOutput", "ValidationResult"]
