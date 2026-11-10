"""Causal-LM method: next-token cross-entropy + perplexity.

Builds a decoder-only model via ``AutoModelForCausalLM`` (or a registered
custom model), computes the shifted-token cross-entropy during training, and
reports perplexity as the validation metric.  The tokenizer is resolved in
``prepare_transforms`` (lifecycle position 1) and read by the ``text``
pipeline, mirroring how the classification method resolves its processor.
"""

import logging
import math

import torch
import torch.nn.functional as F

from genml_kit.methods.base import Method
from genml_kit.methods.registry import METHODS
from genml_kit.pipelines.contracts import LossOutput, ValidationResult
from genml_kit.training.packed_attention import (build_block_diagonal_mask,
                                                 flash_varlen_available,
                                                 flash_varlen_layout)
from genml_kit.training.text_collator import IGNORE_INDEX
from genml_kit.utils.attr import MISSING, get_attribute
from genml_kit.utils.logging import fatal
from genml_kit.utils.shared_args import ARG_TOKENIZER_ARG, add_shared_arg


def _is_hf_model(model):
  """True for a HuggingFace ``PreTrainedModel`` (not a custom generative LM)."""
  try:
    from transformers import PreTrainedModel
  except Exception:
    return False
  return isinstance(model, PreTrainedModel)


def _hf_attention_backend(model):
  """Best-effort read of a HuggingFace model's resolved attention backend.

  NOTE: this peeks at ``config._attn_implementation``, which -- despite the
  leading underscore -- is the *only* place HF exposes the resolved backend
  (the public ``attn_implementation`` name is a constructor kwarg, not an
  attribute).  It is therefore not a stable public API; the read is defensive
  (dotted-path ``get_attribute`` returning ``MISSING``) and the caller falls
  back to the safe assumption when it cannot be resolved.  Prefers HF's own
  helper when importable, else the property directly.

  Returns the backend string, or ``MISSING`` if it cannot be determined.
  """
  # Preferred: HF's public classifier, which also recognises flash variants
  # beyond the literal "flash_attention_2" (e.g. flash_attention_3, kernels
  # package aliases).  Guarded: importing it or calling it may fail across
  # transformers versions, in which case we fall through to the raw property.
  try:
    from transformers.modeling_utils import is_flash_attention_requested
    config = get_attribute(model, "config")
    if config is not MISSING and is_flash_attention_requested(config=config):
      return "flash"
  except Exception:
    pass
  # Fallback: the private resolved-backend property (public alias first).
  backend = get_attribute(model, "config.attn_implementation")
  if backend is MISSING:
    backend = get_attribute(model, "config._attn_implementation")
  return backend


def _resolve_packed_capability(model):
  """How *model* keeps documents isolated in a packed window.

  Custom models declare :attr:`GenerativeLM.PACKED_ATTENTION` (default
  ``"none"``).  HuggingFace decoders are inferred from their attention
  backend: a flash kernel packs natively (varlen), the other backends
  (sdpa/eager) take a 4-D additive mask (block diagonal).
  """
  # A declared attribute wins (custom models always carry one via the base
  # class).  Resolved with the recursive (dotted-path) helper so a missing
  # attribute yields MISSING instead of raising.
  declared = get_attribute(model, "PACKED_ATTENTION")
  if declared is not MISSING:
    return declared
  # HuggingFace decoder: infer from the resolved attention backend.
  if _is_hf_model(model):
    backend = _hf_attention_backend(model)
    if backend is MISSING:
      # Backend unknown (older/newer transformers): assume the conservative,
      # kernel-free path.  block_diagonal is pure torch and always available.
      return "block_diagonal"
    # Any flash flavor consumes packed (varlen) boundaries.
    return "varlen" if "flash" in str(backend) else "block_diagonal"
  # Unknown module with no declaration: assume it cannot isolate documents.
  return "none"


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
    add_shared_arg(group, ARG_TOKENIZER_ARG)
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
        args.tokenizer or args.model,
        cache_dir=args.cache_dir,
        **(args.tokenizer_arg or {}),
    )

  def build_model(self, args, device):
    """Load a causal LM and apply the shared model extras (LoRA/freeze/...)."""
    from genml_kit.models import load_model

    model = load_model(
        args.model,
        task="causal_lm",
        num_labels=0,
        cache_dir=args.cache_dir,
        device=device,
        trust_remote_code=args.trust_remote_code,
        attn_implementation=args.attn_implementation,
        **args.model_arg,
    )
    model = self._apply_model_extras(args, model, device)
    # The model is the only authority on how it keeps documents isolated in a
    # packed window.  Resolved (and memoized) here -- the one point where both
    # the model and the packing intent are known.  Packing comes from the
    # compiled ``--sequence_mode`` (the single source of truth), not the raw
    # ``--packing`` flag.
    from genml_kit.training.sequence_mode import resolve_sequence_mode
    sequence = resolve_sequence_mode(args.sequence_mode, args.packing,
                                     args.length_bucket)
    self._packed_capability = _resolve_packed_capability(model)
    if sequence.windowing in ("wrapped", "bfd"):
      if self._packed_capability == "none":
        fatal(
            f"packing was requested but model {args.model!r} declares "
            f"PACKED_ATTENTION='none', so document boundaries cannot be "
            f"enforced (tokens would attend across documents).  Use a model "
            f"declaring 'block_diagonal' or 'varlen', or drop --packing.", ValueError)
      if self._packed_capability == "varlen" and not flash_varlen_available():
        fatal(
            f"packing with model {args.model!r} needs the optional 'flash_attn' "
            f"package (its attention backend is flash).  Install it "
            f"(pip install flash-attn) or select a non-flash backend "
            f"(--attn_implementation sdpa).", ValueError)
      logging.info("Packed attention: capability=%s.", self._packed_capability)
    return model

  def _forward(self, model, blob):
    """Call *model*, enforcing document boundaries when the batch is packed.

        The collator emits ``segment_ids`` (and ``position_ids``) only under
        ``--packing``.  When present, the batch is translated to whatever the
        model declared it can consume; otherwise the historical 2-D call is
        made verbatim.
        """
    data = blob.data
    # No boundary tensors => the batch was not packed: keep the historical call
    # exactly as it was (2-D key-padding mask only).
    if "segment_ids" not in data:
      return model(input_ids=data["input_ids"], attention_mask=data["attention_mask"])
    # Packed batch.  ``segment_ids`` is [B, L] (document run per token, -1 pad).
    segment_ids = data["segment_ids"]
    capability = getattr(self, "_packed_capability", None)
    if capability is None:
      # Resolve lazily for callers that never went through build_model (e.g. a
      # standalone evaluate on a freshly loaded model).
      capability = _resolve_packed_capability(model)
      self._packed_capability = capability
    if capability == "varlen":
      # HuggingFace flash decoders do not accept a 4-D additive mask: they take
      # the document boundaries as ``FlashAttentionKwargs`` on a padding-free
      # flattened [1, total] batch (attention_mask=None).  Flatten the kept
      # tokens row-major (documents stay contiguous), pass the derived
      # cu_seqlens / max_length, then scatter the flattened logits back to
      # [B, L, V] so the unchanged shifted cross-entropy lines up with labels.
      valid, cu_seqlens, max_length = flash_varlen_layout(data["attention_mask"],
                                                          segment_ids)
      input_ids = data["input_ids"]
      position_ids = data["position_ids"]
      out = model(input_ids=input_ids[valid].unsqueeze(0),
                  attention_mask=None,
                  position_ids=position_ids[valid].unsqueeze(0),
                  use_cache=False,
                  cu_seq_lens_q=cu_seqlens,
                  cu_seq_lens_k=cu_seqlens,
                  max_length_q=max_length,
                  max_length_k=max_length)
      b, length = input_ids.shape
      full = out.logits.new_zeros(b, length, out.logits.shape[-1])
      # ``full[valid]`` and ``input_ids[valid]`` both fill in row-major order,
      # so the flattened positions line up.
      full[valid] = out.logits[0]
      out.logits = full
      return out
    if capability == "block_diagonal":
      # Build the [B, 1, L, L] additive mask from the [B, L] segment ids and
      # pass it in place of the 2-D attention_mask; the model stays oblivious
      # to document structure beyond the mask it already understands.
      mask = build_block_diagonal_mask(segment_ids)
      return model(input_ids=data["input_ids"],
                   attention_mask=mask,
                   position_ids=data["position_ids"])
    if capability == "segment_ids":
      # A model that computes attention without materializing the L x L score
      # matrix (linear / streaming / recurrent) cannot consume a dense
      # block-diagonal mask -- doing so would defeat the very cost it saves.
      # Hand it the raw boundary tensors instead: the full untranslated batch
      # (the ordinary 2-D key-padding mask, ``position_ids``) plus the per-token
      # document runs, and let the model reset its state at each boundary.
      return model(input_ids=data["input_ids"],
                   attention_mask=data["attention_mask"],
                   position_ids=data["position_ids"],
                   segment_ids=segment_ids)
    fatal(
        "Packed batch received but the model cannot isolate documents "
        "(PACKED_ATTENTION='none').", ValueError)

  def train_step(self, model, blob, global_step, *, labels=None):
    """Forward, shifted cross-entropy over ``labels`` (ignore index -100)."""
    # Imported lazily: train_reporting pulls in optim_factory -> methods,
    # so a module-level import here would close a circular import
    # (train_reporting -> optim_factory -> methods -> causal_lm).
    from genml_kit.training.train_reporting import derived_metric

    del global_step, labels
    targets = blob.meta.get("labels")
    if targets is None:
      fatal(
          "CausalLMMethod.train_step requires meta['labels'] "
          "(the text collator provides them).", ValueError)
    out = self._forward(model, blob)
    logits = out.logits  # [B, L, V]
    # Shift: token t is predicted from positions < t.  Drop the last logit and
    # the first label so both align to positions 1..L-1: [B, L, V] -> [B, L-1, V]
    # and [B, L] -> [B, L-1].
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = targets[:, 1:].contiguous()
    # Flatten to a single (B*(L-1)) token axis for the cross-entropy.
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
    """This rank's token CE / perplexity as a ``ValidationResult``.

        Returns the raw summed loss as the ``loss`` metric and the token count
        as ``weight``; ``merge_validation`` divides once over the merged totals.
        Perplexity is nonlinear in the mean, so it is computed after the merge
        rather than per-rank.
        """
    from genml_kit.training.model_utils import model_mode

    total_loss, total_tokens = 0.0, 0
    with torch.no_grad(), model_mode(model, "eval"):
      for blob in loader:
        blob = to_device(blob, device)
        targets = blob.meta["labels"]
        out = self._forward(model, blob)
        # Same causal shift as train_step: align logits[..., :L-1, :] with
        # labels[..., 1:, ...] ([B, L, V] -> [B, L-1, V], [B, L] -> [B, L-1]).
        shift_logits = out.logits[:, :-1, :].contiguous()
        shift_labels = targets[:, 1:].contiguous()
        n_tokens = (shift_labels != IGNORE_INDEX).sum().item()
        if n_tokens == 0:
          continue
        # Sum (not mean) so the merge can weight by the true token count.
        loss_sum = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1),
                                   ignore_index=IGNORE_INDEX,
                                   reduction="sum")
        total_loss += loss_sum.item()
        total_tokens += n_tokens
    mean_loss = total_loss / max(total_tokens, 1)
    # ``metrics`` is this rank's own (correct) value; the raw sum + token count
    # ride along in ``accumulators`` so ranks merge by re-deriving the mean
    # rather than averaging means.
    return ValidationResult(
        {
            "loss": mean_loss,
            "ppl": float(torch.exp(torch.tensor(mean_loss))),
        }, float(total_tokens), (total_loss, total_tokens))

  def merge_validation(self, results):
    """Merge summed loss over ranks, then derive the mean CE and perplexity."""
    total_loss = sum(r.accumulators[0] for r in results)
    total_tokens = sum(r.accumulators[1] for r in results) or 1.0
    mean_loss = total_loss / total_tokens
    return {
        "loss": mean_loss,
        "ppl": float(torch.exp(torch.tensor(mean_loss))),
    }

  def get_checkpoint_state(self, model, args):
    return {
        "method": "causal_lm",
        "tokenizer": args.tokenizer or args.model,
        "max_seq_len": args.max_seq_len
    }
