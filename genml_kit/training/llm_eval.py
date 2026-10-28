"""``genml-kit-llm eval``: perplexity and reference-based quality metrics.

Metrics
-------

- **perplexity** -- teacher-forced, no generation.  Runs the model over a
  ``text`` column and reports ``exp(mean token cross-entropy)``, the same
  objective the ``causal_lm`` method trains with.
- **exact-match** -- fraction of greedy generations that match the reference
  exactly (after whitespace stripping).
- **token-f1** -- SQuAD-style token-overlap F1 between generation and
  reference, averaged over items.
- **pass@k** -- *optional* and pluggable.  A user-supplied scorer script
  (``--scorer``, a ``.py`` path/URL defining ``score(prompt, completion,
  reference) -> float``) is invoked ``k`` times per item and the pass rate
  is reported.  No code execution is built in; the scorer owns that.

The metrics live in small pure functions so they can be tested without a
model, and are assembled by :func:`run_eval`.
"""

import json
import logging
import os

import torch
import torch.nn.functional as F

from genml_kit.training.llm_model import load_lm_for_inference
from genml_kit.utils.args import add_logging_args
from genml_kit.utils.generation import (SamplingConfig, generate_from_prompts)
from genml_kit.utils.logging import fatal, setup_logging
from genml_kit.utils.script import is_script_spec, load_extern
from genml_kit.utils.shared_args import (ARG_ATTN_IMPLEMENTATION, ARG_BATCH_SIZE,
                                         ARG_CACHE_DIR, ARG_DATASET, ARG_MAX_ITEMS,
                                         ARG_MAX_SEQ_LEN, ARG_MODEL, ARG_SEED,
                                         ARG_SPLIT, ARG_TOKENIZER,
                                         ARG_TRUST_REMOTE_CODE, add_shared_arg)

_METRICS = ("ppl", "exact_match", "token_f1", "pass_at_k")


def add_eval_args(parser):
  """Add the ``eval`` subcommand arguments to *parser*."""
  group = parser.add_argument_group("model")
  add_shared_arg(
      group,
      ARG_MODEL,
      required=True,
      help="HuggingFace model id or registered custom model.",
  )
  add_shared_arg(group, ARG_TOKENIZER)
  group.add_argument("--checkpoint",
                     type=str,
                     help="State-dict checkpoint, or a <root> with "
                     "<root>_best.pt / <root>_latest.pt.")
  group.add_argument("--lora_adapter",
                     type=str,
                     help="PEFT adapter directory (save_pretrained layout).")
  group.add_argument("--device", type=str, help="Device: cuda or cpu.")
  add_shared_arg(group, ARG_CACHE_DIR)
  add_shared_arg(group, ARG_TRUST_REMOTE_CODE)
  add_shared_arg(group, ARG_ATTN_IMPLEMENTATION)

  group = parser.add_argument_group("data")
  add_shared_arg(group, ARG_DATASET)
  add_shared_arg(group, ARG_SPLIT, default="test")
  group.add_argument("--text_column",
                     type=str,
                     default="text",
                     help="Column used for perplexity (default: text).")
  group.add_argument("--prompt_column",
                     type=str,
                     default="prompt",
                     help="Column holding the prompt (default: prompt).")
  group.add_argument("--reference_column",
                     type=str,
                     default="reference",
                     help="Column holding the reference (default: reference).")
  add_shared_arg(group, ARG_MAX_ITEMS)
  add_shared_arg(group,
                 ARG_MAX_SEQ_LEN,
                 help="Perplexity window length (default: 512).")
  add_shared_arg(group, ARG_BATCH_SIZE, default=8)

  group = parser.add_argument_group("metrics")
  group.add_argument("--metrics",
                     nargs="+",
                     choices=_METRICS,
                     default=list(_METRICS[:3]),
                     help="Metrics to compute (default: ppl exact_match "
                     "token_f1).")
  group.add_argument("--max_new_tokens",
                     type=int,
                     default=64,
                     help="Generation length for quality metrics (default: 64).")
  group.add_argument("--do_sample",
                     action="store_true",
                     help="Sample generations instead of greedy decoding.")
  group.add_argument("--temperature",
                     type=float,
                     default=1.0,
                     help="Sampling temperature (default: 1.0).")
  group.add_argument("--top_k",
                     type=int,
                     default=0,
                     help="Keep only the top-K logits (0 disables).")
  group.add_argument("--top_p",
                     type=float,
                     default=1.0,
                     help="Nucleus sampling threshold (1.0 disables).")
  add_shared_arg(group, ARG_SEED, help="Sampling seed.")
  group.add_argument("--no_chat_template",
                     action="store_true",
                     help="Do not apply the tokenizer chat template.")
  group.add_argument("--pass_at_k",
                     type=int,
                     default=0,
                     help="k for pass@k (0 disables).")
  group.add_argument("--scorer",
                     type=str,
                     help="Scorer script (.py path/URL) defining "
                     "score(prompt, completion, reference) -> float, "
                     "required by --pass_at_k.")

  group = parser.add_argument_group("output")
  group.add_argument("--output",
                     type=str,
                     help="JSON report path; stdout when omitted.")

  add_logging_args(parser)


def compute_perplexity(model, tokenizer, texts, device, max_seq_len=512, batch_size=8):
  """Return perplexity over *texts* (teacher-forced, no generation).

    Each text is tokenized, truncated to *max_seq_len*, and the shifted
    next-token cross-entropy is accumulated with ``reduction="sum"`` over
    the non-ignored targets -- the same objective ``causal_lm`` uses.
    """
  model.eval()
  # Imported lazily: text_collator pulls in the pipelines package, and this
  # module is imported at CLI start-up before pipelines finish initializing.
  from genml_kit.training.text_collator import IGNORE_INDEX

  device = torch.device(device)
  total_loss = 0.0
  total_tokens = 0
  with torch.no_grad():
    for start in range(0, len(texts), batch_size):
      chunk = texts[start:start + batch_size]
      encoded = [tokenizer.encode(t, add_special_tokens=True) for t in chunk]
      encoded = [ids[:max_seq_len] for ids in encoded]
      encoded = [ids for ids in encoded if len(ids) >= 2]
      if not encoded:
        continue
      width = max(len(ids) for ids in encoded)
      pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
      input_ids = []
      attention_mask = []
      for ids in encoded:
        pad = width - len(ids)
        input_ids.append(list(ids) + [pad_id] * pad)
        attention_mask.append([1] * len(ids) + [0] * pad)
      input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
      attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
      logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
      shift_logits = logits[:, :-1, :].contiguous()
      shift_labels = input_ids[:, 1:].contiguous()
      # Positions predicted from a pad token carry no signal.
      shift_labels = shift_labels.masked_fill(attention_mask[:, 1:] == 0, IGNORE_INDEX)
      n_tokens = int((shift_labels != IGNORE_INDEX).sum().item())
      if n_tokens == 0:
        continue
      loss_sum = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                 shift_labels.view(-1),
                                 ignore_index=IGNORE_INDEX,
                                 reduction="sum")
      total_loss += loss_sum.item()
      total_tokens += n_tokens
  mean_loss = total_loss / max(total_tokens, 1)
  return float(torch.exp(torch.tensor(mean_loss)))


def exact_match(prediction, reference):
  """Return 1.0 when the stripped strings match, else 0.0."""
  return 1.0 if prediction.strip() == reference.strip() else 0.0


def token_f1(prediction, reference):
  """SQuAD-style token-overlap F1 between *prediction* and *reference*.

    Tokens are whitespace-split; the score is the harmonic mean of precision
    and recall over the token multiset intersection.  Two empty strings score
    1.0; an empty prediction against a non-empty reference scores 0.0.
    """
  pred_tokens = prediction.split()
  ref_tokens = reference.split()
  if not pred_tokens and not ref_tokens:
    return 1.0
  if not pred_tokens or not ref_tokens:
    return 0.0
  from collections import Counter

  common = Counter(pred_tokens) & Counter(ref_tokens)
  overlap = sum(common.values())
  if overlap == 0:
    return 0.0
  precision = overlap / len(pred_tokens)
  recall = overlap / len(ref_tokens)
  return 2 * precision * recall / (precision + recall)


def load_scorer(spec):
  """Load a ``score(prompt, completion, reference) -> float`` callable.

    *spec* is a ``.py`` path or HTTP(S) URL resolved via
    :func:`genml_kit.utils.script.load_extern`.
    """
  if not is_script_spec(spec):
    fatal(f"--scorer must be a .py path or URL, got {spec!r}", ValueError)
  return load_extern(spec, "score")


def _load_eval_data(args):
  """Load the eval split as a dict of column -> list."""
  from genml_kit.datasets.text_proxy import load_raw_split

  dataset = load_raw_split(args.dataset,
                           args.split,
                           cache_dir=args.cache_dir,
                           token=None)
  if args.max_items:
    dataset = dataset.select(range(min(args.max_items, len(dataset))))
  return dataset


def _generation_config(args):
  """Build the :class:`SamplingConfig` used for quality metrics."""
  return SamplingConfig(
      max_new_tokens=args.max_new_tokens,
      do_sample=args.do_sample,
      temperature=args.temperature,
      top_k=args.top_k,
      top_p=args.top_p,
      seed=args.seed,
  )


def run_eval(args):
  """Execute the ``eval`` subcommand."""
  setup_logging(args.log_level, args.log_targets)
  if args.pass_at_k and not args.scorer:
    fatal("--pass_at_k requires --scorer.", ValueError)
  device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
  logging.info("Using device: %s", device)

  if not args.dataset:
    fatal("No eval data: pass --dataset.", ValueError)
  dataset = _load_eval_data(args)
  columns = dataset.column_names

  model, tokenizer = load_lm_for_inference(
      model_name=args.model,
      tokenizer_name=args.tokenizer,
      checkpoint=args.checkpoint,
      lora_adapter=args.lora_adapter,
      device=device,
      cache_dir=args.cache_dir,
      trust_remote_code=args.trust_remote_code,
      attn_implementation=args.attn_implementation,
  )

  report = {"model": args.model, "split": args.split, "n_items": len(dataset)}

  if "ppl" in args.metrics:
    if args.text_column not in columns:
      fatal(
          f"--metrics ppl needs column {args.text_column!r} "
          f"(available: {columns}).", ValueError)
    report["ppl"] = compute_perplexity(model,
                                       tokenizer,
                                       list(dataset[args.text_column]),
                                       device,
                                       max_seq_len=args.max_seq_len,
                                       batch_size=args.batch_size)

  quality = [m for m in args.metrics if m in ("exact_match", "token_f1", "pass_at_k")]
  if quality:
    if args.prompt_column not in columns or args.reference_column not in columns:
      fatal(
          f"Quality metrics need columns {args.prompt_column!r} and "
          f"{args.reference_column!r} (available: {columns}).", ValueError)
    prompts = list(dataset[args.prompt_column])
    references = list(dataset[args.reference_column])
    report.update(_quality_report(model, tokenizer, prompts, references, args, device))

  _write_report(report, args.output)


def _quality_report(model, tokenizer, prompts, references, args, device):
  """Compute exact-match / token-F1 / pass@k over prompt-reference pairs."""
  report = {}
  config = _generation_config(args)
  completions = generate_from_prompts(model,
                                      tokenizer,
                                      prompts,
                                      config=config,
                                      device=device,
                                      batch_size=args.batch_size,
                                      chat_template=not args.no_chat_template)
  if "exact_match" in args.metrics:
    scores = [exact_match(c, r) for c, r in zip(completions, references)]
    report["exact_match"] = sum(scores) / max(len(scores), 1)
  if "token_f1" in args.metrics:
    scores = [token_f1(c, r) for c, r in zip(completions, references)]
    report["token_f1"] = sum(scores) / max(len(scores), 1)
  if "pass_at_k" in args.metrics:
    report["pass_at_k"] = _pass_at_k(model, tokenizer, prompts, references, args,
                                     device)
    report["pass_at_k_k"] = args.pass_at_k
  return report


def _pass_at_k(model, tokenizer, prompts, references, args, device):
  """Pass rate over ``k`` sampled generations per item (pluggable scorer).

    Returns the fraction of items for which *any* of the ``k`` samples scored
    non-zero.  Sampling is forced on so the ``k`` draws differ.
    """
  scorer = load_scorer(args.scorer)
  config = _generation_config(args)
  config.do_sample = True
  k = args.pass_at_k
  passed = 0
  for prompt, reference in zip(prompts, references):
    batch = [prompt] * k
    completions = generate_from_prompts(model,
                                        tokenizer,
                                        batch,
                                        config=config,
                                        device=device,
                                        batch_size=k,
                                        chat_template=not args.no_chat_template)
    if any(scorer(prompt, c, reference) > 0 for c in completions):
      passed += 1
  return passed / max(len(prompts), 1)


def _write_report(report, output):
  """Write the JSON *report* to *output* (stdout when ``None``)."""
  text = json.dumps(report, indent=2, sort_keys=True)
  if output:
    parent = os.path.dirname(os.path.abspath(output))
    os.makedirs(parent, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
      f.write(text + "\n")
    logging.info("Wrote eval report to %s", output)
    return
  print(text)
