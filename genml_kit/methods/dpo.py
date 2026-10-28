"""DPO: Direct Preference Optimization (Rafailov et al. 2023).

Trains a policy to prefer the chosen response of a preference pair over the
rejected one, relative to a **frozen reference** model.  The policy is built
through the normal registry (``load_model``), so any decoder-only model works.

Two models live in memory at once:

- the **policy** -- trained, saved, has gradients;
- the **reference** -- frozen, held as a plain attribute (never an
  ``nn.Module`` submodule), reconstructed from ``--reference_model`` /
  ``--reference_checkpoint`` on resume, so it never enters ``state_dict``.

The DPO loss is the sigmoid objective over the implicit reward margin
(:func:`genml_kit.losses.alignment.dpo_loss`); the sequence score is the mean
per-token log-probability of the response tokens
(:func:`genml_kit.losses.alignment.sequence_logps`), so policy and reference
aggregate identically.
"""

import torch

from genml_kit.losses import load_loss
from genml_kit.losses.alignment import sequence_logps
from genml_kit.methods.base import Method
from genml_kit.methods.registry import METHODS
from genml_kit.pipelines.contracts import LossOutput
from genml_kit.training.text_collator import IGNORE_INDEX


def _shifted_logps(model, input_ids, attention_mask, labels):
  """Mean per-token log-prob of *labels* under *model* (next-token shift)."""
  out = model(input_ids=input_ids, attention_mask=attention_mask)
  logits = out.logits[:, :-1, :].contiguous()
  targets = labels[:, 1:].contiguous()
  return sequence_logps(logits, targets, ignore_index=IGNORE_INDEX)


@METHODS.register
class DPOMethod(Method):
  """Direct Preference Optimization over chosen/rejected pairs."""

  NAME = "dpo"
  METRIC_KEY = "accuracy"
  METRIC_MINIMIZE = False
  NEEDS_LABELS = True

  def __init__(self):
    super().__init__()
    self.tokenizer = None
    self.reference_model = None
    self._dpo_loss = None
    self._beta = 0.1

  @classmethod
  def add_args(cls, parser):
    group = parser.add_argument_group("dpo method")
    # ``--tokenizer`` / ``--trust_remote_code`` / ``--attn_implementation`` are
    # registered by CausalLMMethod.add_args (always run by register_all_owners)
    # and read via getattr below -- re-declaring them would collide.
    group.add_argument("--reference_model",
                       type=str,
                       help="Reference model id; defaults to --model.")
    group.add_argument("--reference_checkpoint",
                       type=str,
                       help="State-dict checkpoint for the reference model.")
    group.add_argument("--beta",
                       type=float,
                       default=0.1,
                       help="DPO temperature (KL strength); default: 0.1.")

  def prepare_transforms(self, args, device):
    """Resolve the tokenizer the preference pipeline reads (position 1)."""
    del device
    from genml_kit.models.tokenizer import load_tokenizer
    self.tokenizer = load_tokenizer(
        args.tokenizer or args.model,
        cache_dir=args.cache_dir,
    )

  def _build_lm(self, args, device, model_name, checkpoint):
    """Load a causal LM through the registry, optionally restoring weights."""
    from genml_kit.models import load_model

    model = load_model(
        model_name,
        task="causal_lm",
        num_labels=0,
        cache_dir=args.cache_dir,
        device=device,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
    )
    if checkpoint:
      from genml_kit.training.llm_model import load_checkpoint_into_model
      model = load_checkpoint_into_model(model, checkpoint, device)
    return model

  def build_model(self, args, device):
    """Build the policy (with model extras) and the frozen reference."""
    self._dpo_loss = load_loss(args.loss or "dpo")
    self._beta = getattr(args, "beta", 0.1)
    policy = self._build_lm(args, device, args.model, None)
    policy = self._apply_model_extras(args, policy, device)

    reference = self._build_lm(args, device,
                               getattr(args, "reference_model", None) or args.model,
                               getattr(args, "reference_checkpoint", None))
    reference.to(device)
    reference.eval()
    for param in reference.parameters():
      param.requires_grad_(False)
    # Plain attribute (not a submodule): keeps the reference out of the
    # policy's state_dict and out of every checkpoint.
    self.reference_model = reference
    return policy

  def train_step(self, model, blob, global_step, *, labels=None):
    """DPO loss for one preference batch."""
    del global_step, labels
    data = blob.data
    beta = self._beta
    policy_chosen = _shifted_logps(model, data["chosen_input_ids"],
                                   data["chosen_attention_mask"],
                                   blob.meta["chosen_labels"])
    policy_rejected = _shifted_logps(model, data["rejected_input_ids"],
                                     data["rejected_attention_mask"],
                                     blob.meta["rejected_labels"])
    with torch.no_grad():
      ref_chosen = _shifted_logps(self.reference_model, data["chosen_input_ids"],
                                  data["chosen_attention_mask"],
                                  blob.meta["chosen_labels"])
      ref_rejected = _shifted_logps(self.reference_model, data["rejected_input_ids"],
                                    data["rejected_attention_mask"],
                                    blob.meta["rejected_labels"])
    loss, chosen_rewards, rejected_rewards = self._dpo_loss(policy_chosen,
                                                            policy_rejected, ref_chosen,
                                                            ref_rejected, beta)
    margin = (chosen_rewards - rejected_rewards).mean()
    return LossOutput(
        loss,
        {
            "loss": loss.detach(),
            "margin": margin,
            "accuracy": (chosen_rewards > rejected_rewards).float().mean(),
        },
        None,
    )

  def evaluate(self, model, loader, device, to_device):
    """Mean DPO loss and pairwise preference accuracy over *loader*."""
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
    """Record the reference spec so it can be rebuilt on resume."""
    return {
        "method": "dpo",
        "tokenizer": args.tokenizer or args.model,
        "reference_model": getattr(args, "reference_model", None) or args.model,
        "reference_checkpoint": getattr(args, "reference_checkpoint", None),
        "beta": getattr(args, "beta", 0.1),
    }
