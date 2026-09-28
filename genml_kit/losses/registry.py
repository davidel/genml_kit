"""Loss registry -- built-in and user-supplied loss builders.

A loss spec is either:

* A registered name (e.g. ``"focal"``, ``"ppo"``, ``"sac"``).
* A ``.py`` path/URL whose script defines ``build_loss(**kwargs)``
  returning a loss callable.

The *returned* object depends on the protocol level:

* **Supervised criteria** (classification): a callable
  ``(logits, targets) -> tensor`` (typically an ``nn.Module``).
* **RL loss bundles** (PPO/SAC/DQN): a dict of callables keyed per
  method (e.g. ``{"policy": ..., "value": ..., "entropy": ...}`` for
  PPO).  External scripts may return a *partial* bundle; it is merged
  over the built-in bundle named by the caller's ``protocol`` argument
  (see ``load_loss``).

External scripts must be self-contained: the registry executes the
script and calls ``build_loss(**kwargs)``; unknown kwargs must be
tolerated by the script (use ``**kwargs`` in the signature).
"""

import logging

from genml_kit.utils.logging import fatal
from genml_kit.utils.registry import Registry
from genml_kit.utils.script import load_extern, is_script_spec

LOSSES = Registry("loss")

# Canonical built-in default (a registered name, not a sentinel).
DEFAULT_LOSS = "focal"


def _load_loss_script(spec, **kwargs):
  """Load a loss from an external script defining ``build_loss(**kwargs)``."""
  build_loss = load_extern(spec, "build_loss")
  logging.info("Loading external loss from %s", spec)
  return build_loss(**kwargs)


def load_loss(spec=DEFAULT_LOSS, *, protocol=None, **kwargs):
  """Build a loss from *spec* -- script, registered name, or default.

  Args:
    spec: Loss spec.  ``None`` or omitted maps to ``DEFAULT_LOSS``
        (``"focal"``).  A ``.py`` path/URL loads an external script
        defining ``build_loss(**kwargs)``.  A registered name builds
        the matching entry via ``LOSSES.build(spec, **kwargs)``.
    protocol: Optional registered name of the *built-in* bundle the
        result is merged over.  Used by RL methods: an external script
        returning a partial dict (e.g. only ``{"entropy": ...}``) is
        merged over the built-in bundle for that protocol (e.g.
        ``"ppo"``).  ``None`` (supervised criteria) returns the script
        result unchanged.
    **kwargs: Forwarded to ``build_loss(**kwargs)`` (script) or the
        registered factory ``cls(**kwargs)``.

  Returns:
    The loss built from *spec*: for supervised methods a callable
    ``(logits, targets) -> tensor``; for RL methods a merged bundle
    dict ``{**builtin, **script}`` when *protocol* is given.
  """
  if spec is None:
    spec = DEFAULT_LOSS
  if is_script_spec(spec):
    built = _load_loss_script(spec, **kwargs)
    return _merge_loss_bundle(built, protocol, kwargs)
  if LOSSES.contains(spec):
    logging.info("Loading loss '%s' from registry.", spec)
    return LOSSES.build(spec, **kwargs)
  available = ", ".join(LOSSES.list_names()) or "(none)"
  fatal(
      f"Unknown loss {spec!r}. Available: {available}. "
      f"Or provide a path to a .py file defining build_loss().",
      ValueError,
  )


def _merge_loss_bundle(built, protocol, kwargs):
  """Merge a script-returned loss bundle over the built-in bundle.

  If *built* is not a dict (a supervised criterion) or *protocol* is
  ``None``, *built* is returned unchanged.  Otherwise the script's
  entries are merged over ``LOSSES.build(protocol, **kwargs)``.
  """
  if not isinstance(built, dict) or protocol is None:
    return built
  if not LOSSES.contains(protocol):
    return built
  base = LOSSES.build(protocol, **kwargs)
  if not isinstance(base, dict):
    return built
  merged = dict(base)
  merged.update(built)
  return merged
