"""Causal-LM method: next-token cross-entropy + perplexity.

Builds a decoder-only model via ``AutoModelForCausalLM`` (or a registered
custom model), computes the shifted-token cross-entropy during training, and
reports perplexity as the validation metric.  The tokenizer is resolved in
``prepare_transforms`` (lifecycle position 1) and read by the ``text``
pipeline, mirroring how the classification method resolves its processor.
"""

import math

import torch
import torch.nn.functional as F

from genml_kit.methods.base import Method
from genml_kit.methods.registry import METHODS
from genml_kit.pipelines.contracts import LossOutput
from genml_kit.training.text_collator import IGNORE_INDEX
from genml_kit.training.train_reporting import derived_metric
from genml_kit.utils.logging import fatal


def _perplexity(ctx):
  """Perplexity of the reported line: ``exp(loss)``.

  ``ctx["loss"]`` is the loss printed on the current line -- the window
  mean in the per-step log, the epoch mean in the summary -- so the
  printed ``ppl`` always matches the ``loss`` beside it, and equals the
  validation metric (``exp`` of the mean shifted cross-entropy).
  """
  return math.exp(ctx["loss"])


@METHODS.register
class CausalLMMethod(Method):
  """Decoder-only language-model fine-tuning (SFT / LM pre-training)."""

  NAME = "causal_lm"
  METRIC_KEY = "ppl"
  METRIC_MINIMIZE = True
  NEEDS_LABELS = True

  def __init__(self):
    super().__init__()
    self.tokenizer = None

  @classmethod
  def add_args(cls, parser):
    group = parser.add_argument_group("causal_lm method")
    group.add_argument("--tokenizer",
                       type=str,
                       help="Tokenizer name/path; defaults to --model.")
    group.add_argument("--trust_remote_code",
                       action="store_true",
                       help="Allow custom code from the model hub.")
    group.add_argument("--attn_implementation",
                       type=str,
                       help="Attention backend (e.g. 'sdpa', 'eager', "
                       "'flash_attention_2').")

  def prepare_transforms(self, args, device):
    """Resolve the tokenizer the text pipeline reads (lifecycle position 1).

        The tokenizer is stored on the method (never stuffed into ``args``),
        exactly as the classification method keeps its transforms.
        """
    del device
    from genml_kit.models.tokenizer import load_tokenizer
    self.tokenizer = load_tokenizer(
        getattr(args, "tokenizer", None) or args.model,
        cache_dir=getattr(args, "cache_dir", None),
    )

  def build_model(self, args, device):
    """Load a causal LM and apply the shared model extras (LoRA/freeze/...)."""
    from genml_kit.models import load_model

    model = load_model(
        args.model,
        task="causal_lm",
        num_labels=0,
        cache_dir=getattr(args, "cache_dir", None),
        device=device,
        trust_remote_code=getattr(args, "trust_remote_code", False),
        attn_implementation=getattr(args, "attn_implementation", None),
        **getattr(args, "model_arg", {}),
    )
    return self._apply_model_extras(args, model, device)

  def train_step(self, model, blob, global_step, *, labels=None):
    """Forward, shifted cross-entropy over ``labels`` (ignore index -100)."""
    del global_step, labels
    targets = blob.meta.get("labels")
    if targets is None:
      fatal(
          "CausalLMMethod.train_step requires meta['labels'] "
          "(the text collator provides them).", ValueError)
    out = model(input_ids=blob.data["input_ids"],
                attention_mask=blob.data["attention_mask"])
    logits = out.logits
    # Shift: token t is predicted from positions < t.
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = targets[:, 1:].contiguous()
    loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                           shift_labels.view(-1),
                           ignore_index=IGNORE_INDEX)
    # ``ppl`` is DERIVED from the reported loss at render time (never
    # averaged): its value is exp(window loss) on the step line and
    # exp(epoch mean loss) in the summary.
    return LossOutput(loss, {
        "loss": loss.detach(),
        "ppl": derived_metric("ppl", _perplexity),
    }, None)

  def evaluate(self, model, loader, device, to_device):
    """Mean token cross-entropy and perplexity over *loader*."""
    from genml_kit.training.model_utils import model_mode

    total_loss, total_tokens = 0.0, 0
    with torch.no_grad(), model_mode(model, "eval"):
      for blob in loader:
        blob = to_device(blob, device)
        targets = blob.meta["labels"]
        out = model(input_ids=blob.data["input_ids"],
                    attention_mask=blob.data["attention_mask"])
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = targets[:, 1:].contiguous()
        n_tokens = (shift_labels != IGNORE_INDEX).sum().item()
        if n_tokens == 0:
          continue
        loss_sum = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1),
                                   ignore_index=IGNORE_INDEX,
                                   reduction="sum")
        total_loss += loss_sum.item()
        total_tokens += n_tokens
    mean_loss = total_loss / max(total_tokens, 1)
    return {
        "loss": mean_loss,
        "ppl": float(torch.exp(torch.tensor(mean_loss))),
    }

  def get_checkpoint_state(self, model, args):
    return {
        "method": "causal_lm",
        "tokenizer": getattr(args, "tokenizer", None) or args.model,
        "max_seq_len": getattr(args, "max_seq_len", None)
    }
