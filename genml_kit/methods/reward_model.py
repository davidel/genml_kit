"""Reward model: a scalar head + Bradley-Terry preference ranking.

Wraps a decoder-only backbone (built through the normal registry, so any model
works) with a single scalar output and trains it to score the chosen response
of a preference pair above the rejected one.

Design notes:

- The scalar head is attached **directly** to the loaded backbone as
  ``model.score`` (an ``nn.Linear``), *not* via a wrapper class.  This keeps
  the backbone's ``state_dict`` keys unprefixed, so ``--source_checkpoint``
  matches them exactly and only the new ``score.*`` keys are unmatched.
- The reward is the head applied to the **last non-pad token's** hidden
  state; :func:`sequence_reward` is the shared extraction helper.
- The backbone must expose last-layer hidden states (accept
  ``output_hidden_states=True`` and return ``.hidden_states``); HuggingFace
  models do, custom models opt in -- the same contract philosophy as
  ``GenerativeLM``.
"""

import torch
import torch.nn as nn

from genml_kit.losses import load_loss
from genml_kit.methods.base import Method
from genml_kit.methods.registry import METHODS
from genml_kit.pipelines.contracts import LossOutput
from genml_kit.utils.logging import fatal


def resolve_hidden_size(model):
  """Return the backbone hidden dimension for the reward head.

    Tries the common config attributes first (``hidden_size`` / ``n_embd`` /
    ``d_model``); if none is present, probes with a one-token forward and
    reads the last hidden dim.  The probe assumes the hidden-states contract,
    which the reward model requires anyway.
    """
  config = getattr(model, "config", None)
  if config is not None:
    for attr in ("hidden_size", "n_embd", "d_model"):
      value = getattr(config, attr, None)
      if isinstance(value, int) and value > 0:
        return value

  # Probe: a single dummy token is enough to learn the width.
  device = next(model.parameters()).device
  dummy = torch.zeros(1, 1, dtype=torch.long, device=device)
  try:
    out = model(input_ids=dummy, output_hidden_states=True)
    hidden = out.hidden_states[-1]
  except (TypeError, AttributeError, KeyError) as exc:
    fatal(
        "Cannot resolve the hidden size for the reward head: the model has "
        "no config.hidden_size/n_embd/d_model and does not expose "
        f"hidden_states ({exc}).", ValueError)
  return int(hidden.shape[-1])


def attach_reward_head(model, hidden_size=None):
  """Attach ``model.score`` (a ``Linear(D, 1)``) to *model* in place."""
  hidden_size = hidden_size or resolve_hidden_size(model)
  model.score = nn.Linear(hidden_size, 1)
  return model


def gather_last_token(hidden_states, attention_mask):
  """Select each row's last non-pad hidden state: ``(B, T, D) -> (B, D)``."""
  # Index of the last position whose mask is 1.
  lengths = attention_mask.sum(dim=1).clamp(min=1) - 1
  idx = lengths.clamp(max=hidden_states.size(1) - 1)
  rows = torch.arange(hidden_states.size(0), device=hidden_states.device)
  return hidden_states[rows, idx]


def sequence_reward(model, input_ids, attention_mask):
  """Scalar reward per sequence: pooled last-token hidden state -> ``score``."""
  out = model(input_ids=input_ids,
              attention_mask=attention_mask,
              output_hidden_states=True)
  hidden = out.hidden_states[-1]
  pooled = gather_last_token(hidden, attention_mask)
  return model.score(pooled).squeeze(-1)


@METHODS.register
class RewardModelMethod(Method):
  """Bradley-Terry reward model over chosen/rejected pairs."""

  NAME = "reward_model"
  METRIC_KEY = "accuracy"
  METRIC_MINIMIZE = False
  NEEDS_LABELS = True

  def __init__(self):
    super().__init__()
    self.tokenizer = None
    self._bt_loss = None

  @classmethod
  def add_args(cls, parser):
    group = parser.add_argument_group("reward_model method")
    # ``--tokenizer`` / ``--trust_remote_code`` / ``--attn_implementation`` are
    # owned by CausalLMMethod.add_args; read them via getattr, never redeclare.
    group.add_argument("--reward_head_hidden",
                       type=int,
                       help="Override the reward head's input size "
                       "(default: the backbone hidden size).")

  def prepare_transforms(self, args, device):
    """Resolve the tokenizer the preference pipeline reads (position 1)."""
    del device
    from genml_kit.models.tokenizer import load_tokenizer
    self.tokenizer = load_tokenizer(
        getattr(args, "tokenizer", None) or args.model,
        cache_dir=getattr(args, "cache_dir", None),
    )

  def build_model(self, args, device):
    """Build the backbone through the registry and attach the reward head."""
    from genml_kit.models import load_model

    self._bt_loss = load_loss(getattr(args, "loss", None) or "bradley_terry")
    model = load_model(
        args.model,
        task="causal_lm",
        num_labels=0,
        cache_dir=getattr(args, "cache_dir", None),
        device=device,
        trust_remote_code=getattr(args, "trust_remote_code", False),
        attn_implementation=getattr(args, "attn_implementation", None),
    )
    # Attach the head BEFORE model extras so freeze/LoRA/source-checkpoint
    # treat score.* like any other parameter.
    attach_reward_head(model, getattr(args, "reward_head_hidden", None))
    return self._apply_model_extras(args, model, device)

  def train_step(self, model, blob, global_step, *, labels=None):
    """Bradley-Terry loss for one preference batch."""
    del global_step, labels
    data = blob.data
    chosen = sequence_reward(model, data["chosen_input_ids"],
                             data["chosen_attention_mask"])
    rejected = sequence_reward(model, data["rejected_input_ids"],
                               data["rejected_attention_mask"])
    loss = self._bt_loss(chosen, rejected)
    return LossOutput(
        loss,
        {
            "loss": loss.detach(),
            "margin": (chosen - rejected).mean().detach(),
            "accuracy": (chosen > rejected).float().mean(),
        },
        None,
    )

  def evaluate(self, model, loader, device, to_device):
    """Mean Bradley-Terry loss and pairwise ranking accuracy over *loader*."""
    from genml_kit.training.model_utils import model_mode

    total_loss, total_acc, n = 0.0, 0.0, 0
    with model_mode(model, "eval"), torch.no_grad():
      for blob in loader:
        blob = to_device(blob, device)
        out = self.train_step(model, blob, 0)
        total_loss += float(out.loss)
        total_acc += float(out.metrics["accuracy"])
        n += 1
    return {
        "loss": total_loss / max(n, 1),
        "accuracy": total_acc / max(n, 1),
    }

  def get_checkpoint_state(self, model, args):
    return {
        "method": "reward_model",
        "tokenizer": getattr(args, "tokenizer", None) or args.model,
        "reward_head_hidden": getattr(args, "reward_head_hidden", None),
    }
