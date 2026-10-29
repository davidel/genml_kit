"""``genml-kit-llm generate``: prompt(s) in, completions out (JSONL).

Prompts come from a text file (one per line), a JSONL file, or a HuggingFace
dataset column.  The model is loaded through the same registry the trainer
uses (``training/llm_model.py``), so a HuggingFace repo id or a registered
custom model both work.  Output is a JSONL stream of
``{"prompt": ..., "completion": ...}`` records.

This module exposes :func:`add_generate_args` (wired into the ``generate``
subparser by :mod:`genml_kit.cli.llm`) and :func:`run_generate`.
"""

import json
import logging
import os

import torch

from genml_kit.training.llm_model import load_lm_for_inference
from genml_kit.utils.args import add_logging_args
from genml_kit.utils.generation import SamplingConfig, generate_from_prompts
from genml_kit.utils.logging import fatal, setup_logging
from genml_kit.utils.shared_args import (ARG_ATTN_IMPLEMENTATION, ARG_BATCH_SIZE,
                                         ARG_CACHE_DIR, ARG_DATASET, ARG_MAX_ITEMS,
                                         ARG_MODEL, ARG_SEED, ARG_SPLIT, ARG_TOKENIZER,
                                         ARG_TRUST_REMOTE_CODE, add_shared_arg)


def add_generate_args(parser):
  """Add the ``generate`` subcommand arguments to *parser*."""
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

  group = parser.add_argument_group("prompts")
  group.add_argument("--prompts",
                     type=str,
                     help="Text file (one prompt per line) or JSONL file.")
  add_shared_arg(group, ARG_DATASET)
  add_shared_arg(group, ARG_SPLIT, default="test")
  group.add_argument("--prompt_column",
                     type=str,
                     default="prompt",
                     help="Column holding the prompt text (default: prompt).")
  group.add_argument("--text_column",
                     type=str,
                     help="Plain-text column, when reading a text dataset.")
  add_shared_arg(group, ARG_MAX_ITEMS)

  group = parser.add_argument_group("sampling")
  group.add_argument("--max_new_tokens",
                     type=int,
                     default=64,
                     help="Number of tokens to generate (default: 64).")
  group.add_argument("--do_sample",
                     action="store_true",
                     help="Sample instead of greedy decoding.")
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
  group.add_argument("--repetition_penalty",
                     type=float,
                     default=1.0,
                     help="Repetition penalty (1.0 disables).")
  group.add_argument("--num_beams",
                     type=int,
                     default=1,
                     help="Beam count for the portable backend (1 = off).")
  group.add_argument("--num_return_sequences",
                     type=int,
                     default=1,
                     help="Completions to produce per prompt.")
  add_shared_arg(group, ARG_SEED, help="Sampling seed.")
  group.add_argument("--no_chat_template",
                     action="store_true",
                     help="Do not apply the tokenizer chat template.")

  group = parser.add_argument_group("output")
  add_shared_arg(group, ARG_BATCH_SIZE, default=8)
  group.add_argument("--output",
                     type=str,
                     help="JSONL output path; stdout when omitted.")

  add_logging_args(parser)


def _read_prompts(args):
  """Collect the prompt strings from the configured source."""
  if args.prompts and args.dataset:
    fatal("Provide either --prompts or --dataset, not both.", ValueError)
  if args.prompts:
    return _read_prompts_file(args.prompts, args.prompt_column)
  if args.dataset:
    return _read_prompts_dataset(args)
  fatal("No prompt source: pass --prompts or --dataset.", ValueError)


def _read_prompts_file(path, prompt_column):
  """Read prompts from a plain-text (one per line) or JSONL file."""
  if not os.path.isfile(path):
    fatal(f"Prompt file not found: {path!r}", FileNotFoundError)
  prompts = []
  with open(path, encoding="utf-8") as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      if line.startswith("{"):
        record = json.loads(line)
        if prompt_column not in record:
          fatal(f"JSONL record missing {prompt_column!r} column: {record!r}",
                ValueError)
        prompts.append(record[prompt_column])
      else:
        prompts.append(line)
  return prompts


def _read_prompts_dataset(args):
  """Read prompts from a HuggingFace dataset column."""
  from genml_kit.datasets.text_proxy import load_raw_split

  dataset = load_raw_split(args.dataset,
                           args.split,
                           cache_dir=args.cache_dir,
                           token=None)
  column = args.text_column or args.prompt_column
  if column not in dataset.column_names:
    fatal(f"Column {column!r} not in dataset "
          f"(available: {dataset.column_names}).", ValueError)
  prompts = list(dataset[column])
  if args.max_items:
    prompts = prompts[:args.max_items]
  return prompts


def _sampling_config(args):
  """Build a :class:`SamplingConfig` from the parsed arguments."""
  return SamplingConfig(
      max_new_tokens=args.max_new_tokens,
      do_sample=args.do_sample,
      temperature=args.temperature,
      top_k=args.top_k,
      top_p=args.top_p,
      repetition_penalty=args.repetition_penalty,
      num_beams=args.num_beams,
      num_return_sequences=args.num_return_sequences,
      seed=args.seed,
  )


def run_generate(args):
  """Execute the ``generate`` subcommand."""
  setup_logging(args.log_level, args.log_targets)
  device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
  logging.info("Using device: %s", device)

  prompts = _read_prompts(args)
  if not prompts:
    fatal("No prompts to generate from.", ValueError)
  logging.info("Generating for %d prompt(s).", len(prompts))

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

  config = _sampling_config(args)
  completions = generate_from_prompts(model,
                                      tokenizer,
                                      prompts,
                                      config=config,
                                      device=device,
                                      batch_size=args.batch_size,
                                      chat_template=not args.no_chat_template)

  records = [{"prompt": p, "completion": c} for p, c in zip(prompts, completions)]
  _write_records(records, args.output)


def _write_records(records, output):
  """Write JSONL *records* to *output* (stdout when ``None``)."""
  if output:
    parent = os.path.dirname(os.path.abspath(output))
    os.makedirs(parent, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
      for record in records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    logging.info("Wrote %d record(s) to %s", len(records), output)
    return
  for record in records:
    print(json.dumps(record, ensure_ascii=False))
