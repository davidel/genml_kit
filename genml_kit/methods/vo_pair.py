"""VO pair method: supervised + photometric staged training.

The loss and the mce metric follow ``vo/README.md`` section 4; the
closed-form components come from ``genml_kit.geometry.similarity`` and
the staged objective from ``genml_kit.training.vo.train_vo`` (kept as
module functions for unit testing).

NOTE (deliberate deviation from plans/GENERIC_PIPELINE.md s 9): the plan
listed reshaping ``VOSimilarityNet.forward`` to accept the
``DataBlob.data`` ``(image_a, image_b)`` tuple directly.  That item was
NOT applied: ``evaluate_vo`` (kept per the plan) and ``test_vo_e2e`` call
``model(image_a, image_b)`` with two positional tensors, so a tuple-only
``forward`` would break them.  Instead, this method unpacks the tuple
from ``blob.data`` and calls ``model(image_a, image_b)`` -- the s 6
blob contract is honoured without changing the model's signature.
"""

import collections

from genml_kit.datasets.vo_pairs import VOPairMeta
from genml_kit.methods.base import Method
from genml_kit.methods.registry import METHODS
from genml_kit.models.registry import load_model
from genml_kit.pipelines.contracts import LossOutput
from genml_kit.training.vo.train_vo import STAGES, vo_losses

# What train_step/evaluate pass to vo_losses: the model's similarity
# estimate and its confidence head (namedtuple for ``pred.params`` access).
VOPrediction = collections.namedtuple("VOPrediction", ["params", "conf"])


@METHODS.register
class VOPairMethod(Method):
  """VO similarity objective (supervised or photometric self-supervised).

  ``METRIC_KEY`` is ``mce`` and ``METRIC_MINIMIZE`` is ``True``: the
  best-checkpoint value is stored under ``best_mce`` (positive, as measured).
  """

  NAME = "vo_pair"
  METRIC_KEY = "mce"
  # Mce is minimized.
  METRIC_MINIMIZE = True
  NEEDS_LABELS = True

  def __init__(self):
    super().__init__()
    self._cfg = None
    self._stage = STAGES["supervised"]

  @classmethod
  def add_args(cls, parser):
    group = parser.add_argument_group("vo_pair method")
    group.add_argument(
        "--vo_stage",
        type=str,
        choices=("supervised", "photometric"),
        default="supervised",
        help="VO loss stage: 'supervised' (mce + similarity deltas) or "
        "'photometric' (adds the photometric residual term).",
    )
    group.add_argument(
        "--vo_profile",
        type=str,
        default="npu-small",
        help="VO model profile registered in the model registry.",
    )
    group.add_argument(
        "--vo_loss_cfg",
        type=str,
        default=None,
        help="Comma-separated loss weights (w_log_s,w_theta,w_conf,w_photo).",
    )

  def build_model(self, args, device):
    """Build the :class:`VOSimilarityNet` and place it on *device*."""
    self._stage = STAGES[getattr(args, "vo_stage", "supervised")]
    self._cfg = self._parse_loss_cfg(getattr(args, "vo_loss_cfg", None))
    model = load_model(
        f"vo/{getattr(args, 'vo_profile', 'npu-small')}",
        num_labels=0,
        image_size=getattr(args, "image_size", 64),
        in_ch=getattr(args, "in_ch", 1),
    ).model
    model.to(device)
    # Extras (LoRA / freeze / checkpointing) apply to the VOSimilarityNet
    # itself; supported but exercised only by unit-level tests today.
    return self._apply_model_extras(args, model, device)

  def _parse_loss_cfg(self, spec):
    """Parse a comma-separated loss-weight string into a small namespace."""
    from genml_kit.training.vo.train_vo import _LossCfg

    cfg = _LossCfg()
    if not spec:
      return cfg
    parts = spec.split(",")
    names = ("w_log_s", "w_theta", "w_conf", "w_photo")
    for name, raw in zip(names, parts):
      setattr(cfg, name, float(raw))
    return cfg

  def train_step(self, model, blob, global_step, *, labels=None):
    # blob.data is the (image_a, image_b) tuple from VOPairPipeline;
    # VOSimilarityNet.forward deliberately keeps two positional args
    # (see module docstring: deviation from the plan's forward-tuple item).
    image_a, image_b = blob.data
    out = model(image_a, image_b)
    batch = {
        "image_a": image_a,
        "image_b": image_b,
        "meta": VOPairMeta(**blob.meta),
        "corners": out.corners,
        "dc": out.dc,
    }
    pred = VOPrediction(params=out.params, conf=out.conf)
    loss, parts = vo_losses(pred, batch, self._cfg, self._stage)
    metrics = {k: v.detach() for k, v in parts.items()}
    return LossOutput(loss=loss, metrics=metrics)

  def evaluate(self, model, loader, device, to_device):
    """Return ``{"mce": ...}`` over the loader (DataBlob items).

    Same aggregation as ``evaluate_vo``; both call
    ``vo_metrics_from_output`` so the metric definitions cannot drift.
    What differs is only how the GT is reached -- here the collated
    ``blob.meta`` dict, there the per-item ``VOPairMeta`` namedtuple.
    """
    import torch

    from genml_kit.training.model_utils import model_mode
    from genml_kit.training.vo.train_vo import (
        VOMetrics,
        vo_metrics_from_output,
    )

    sums = torch.zeros(4, dtype=torch.float64)
    count = 0
    with model_mode(model, "eval"), torch.no_grad():
      for blob in loader:
        blob = to_device(blob, device)
        image_a, image_b = blob.data
        out = model(image_a, image_b)
        metrics = vo_metrics_from_output(out, blob.meta["gt"], blob.meta["gt_residual"])
        sums += torch.tensor(
            [metrics.mce, metrics.dlog_s, metrics.dtheta, metrics.conf_mae],
            dtype=torch.float64)
        count += 1
    if count == 0:
      metrics = VOMetrics()
    else:
      sums /= count
      metrics = VOMetrics(*sums.tolist())
    return {"mce": float(metrics.mce)}

  def has_metric_improved(self, new_metric, best_metric):
    """Return True when the corner error improved (lower is better)."""
    return new_metric < best_metric

  def get_checkpoint_state(self, model, args):
    return {"method": "vo_pair", "stage": self._stage}
