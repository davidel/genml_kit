"""Built-in named losses registered in the LOSSES registry.

Two protocol levels are registered here:

* **Supervised criteria** -- ``focal`` / ``default``: a factory
  returning a callable ``(logits, targets) -> tensor`` (an
  ``nn.Module``).
* **RL loss bundles** -- ``ppo`` / ``sac`` / ``dqn``: each returns a
  dict of the plain callables used by the corresponding RL method.
  External scripts may override individual keys; ``load_loss`` merges
  ``{**builtin_bundle, **script_bundle}``.

Every factory tolerates unknown kwargs via ``**kwargs`` so that CLI
``--loss_args`` payloads never crash built-in losses.
"""

from genml_kit.losses.registry import LOSSES


@LOSSES.register("focal")
@LOSSES.register("default")
def focal_loss(*, weights=None, gamma=0.0, label_smoothing=0.0, **kwargs):
  """Cost-sensitive focal loss for soft targets (built-in default).

  Args:
    weights: Optional (C,) class-weight tensor (or ``None``).
    gamma:   Focal modulation exponent (0 disables focal weighting).
    label_smoothing: Label smoothing epsilon (0 disables).

  Returns:
    ``CombinedFocalLoss`` instance callable as ``(logits, targets)``.
  """
  from genml_kit.losses.focal import CombinedFocalLoss

  return CombinedFocalLoss(
      weights=weights,
      gamma=gamma,
      label_smoothing=label_smoothing,
  )


@LOSSES.register("supcon")
def supcon_loss(*, temperature=0.07, **kwargs):
  """Supervised contrastive loss (SupCon) factory.

  Returns the ``supcon_loss`` function itself; the method passes the
  gathered (features, labels) tensors directly.
  """
  from genml_kit.losses.contrastive import supcon_loss as _supcon

  return _supcon


@LOSSES.register("ppo")
def ppo_loss_bundle(*, clip_eps=0.2, vf_clip_eps=0.5, **kwargs):
  """Default PPO loss bundle (policy + value + entropy).

  Keys: ``policy(ratio, advantages, clip_eps)``,
  ``value(pred_v, returns, old_values, clip_eps)``,
  ``entropy(entropy)``.
  """
  from genml_kit.losses.rl import clipped_surrogate, entropy_bonus, value_loss

  return {
      "policy": clipped_surrogate,
      "value": value_loss,
      "entropy": entropy_bonus,
  }


@LOSSES.register("sac")
def sac_loss_bundle(**kwargs):
  """Default SAC loss bundle (q + policy + alpha).

  Keys: ``q(pred_q, soft_target)``,
  ``policy(log_probs, q_values, alpha)``,
  ``alpha(log_probs, target_entropy, coef)``.
  """
  from genml_kit.losses.rl import sac_alpha_loss, sac_policy_loss, sac_q_loss

  return {
      "q": sac_q_loss,
      "policy": sac_policy_loss,
      "alpha": sac_alpha_loss,
  }


@LOSSES.register("dqn")
def dqn_loss_bundle(*, reduction="mean", **kwargs):
  """Default DQN loss bundle (td).

  Key: ``td(pred_q, target, reduction)``.
  """
  from genml_kit.losses.rl import td_loss

  return {"td": td_loss}


# Import-time registration hook: re-exported by genml_kit.losses.
__all__ = [
    "dqn_loss_bundle",
    "focal_loss",
    "ppo_loss_bundle",
    "sac_loss_bundle",
    "supcon_loss",
]
