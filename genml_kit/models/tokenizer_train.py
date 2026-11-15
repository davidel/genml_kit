"""Offline BPE tokenizer *training*: corpus -> HuggingFace-format artifact.

The counterpart to :mod:`genml_kit.models.tokenizer`, which only *loads*
tokenizers.  Building one is a one-shot offline job, deliberately kept apart
from model training so that the corpus subset used for the vocabulary is an
explicit, reproducible, reviewable artifact rather than a side effect of a
training run.

The emitted directory is a plain HuggingFace tokenizer directory, so it is
consumed with no new loader code::

    genml-kit-tokenizer --dataset roneneldan/TinyStories --text_column text \
      --fraction 0.1 --seed 42 --vocab_size 4096 --out hf/tok/ts_bpe_4096_s0.1

    genml-kit-train --tokenizer hf/tok/ts_bpe_4096_s0.1 ...

Subsetting a corpus is expressed as a *fraction* of the rows, never a byte or
token budget: a tokenizer cares about which text it sees, and row selection is
the only notion that works identically for a random-access dataset and for a
streaming one.
"""

import datetime
import itertools
import json
import logging
import os
import random

from genml_kit.datasets.base import is_random_access
from genml_kit.datasets.text_proxy import load_raw_split
from genml_kit.utils.logging import fatal

# Order is load-bearing: ``BpeTrainer`` assigns ids in the order given, so this
# list pins ``<pad>=0, <eos>=1, <bos>=2, <unk>=3``.  Reordering silently shifts
# every token id and invalidates checkpoints trained against older artifacts.
_SPECIAL_TOKENS = ("<pad>", "<eos>", "<bos>", "<unk>")
_UNK = "<unk>"
_PAD = "<pad>"
_BOS = "<bos>"
_EOS = "<eos>"

_ARTIFACT_TOKENIZER = "tokenizer.json"
_ARTIFACT_CONFIG = "tokenizer_config.json"
_ARTIFACT_META = "tokenizer_meta.json"

DEFAULT_VOCAB_SIZE = 4096


def _package_version():
  """Return the installed ``genml_kit`` version, or ``'unknown'``."""
  try:
    from importlib.metadata import version
    return version("genml_kit")
  except Exception:  # pragma: no cover - metadata absent in odd installs
    return "unknown"


def _check_fraction(fraction):
  """Validate a corpus fraction, allowing ``None`` for "use everything"."""
  if fraction is None:
    return None
  try:
    fraction = float(fraction)
  except (TypeError, ValueError) as e:
    fatal(f"--fraction must be a float, got {fraction!r}", ValueError, cause=e)
  if not 0.0 < fraction <= 1.0:
    fatal(f"--fraction must be in (0.0, 1.0], got {fraction!r}", ValueError)
  return fraction


def _sample_indices(num_rows, fraction, seed):
  """Pick ``fraction`` of ``num_rows`` indices, reproducibly and *exactly*.

  Used for random-access datasets, where the row count is known: the corpus is
  then exactly ``round(num_rows * fraction)`` rows drawn without replacement,
  and the result is independent of iteration order or on-disk sharding.  That
  exactness is the reason to prefer this path whenever it is available.

  Args:
    num_rows: Number of rows available.
    fraction: Fraction to keep, in ``(0.0, 1.0]``.
    seed: RNG seed.

  Returns:
    A sorted list of distinct row indices.
  """
  keep = int(round(num_rows * fraction))
  keep = max(0, min(keep, num_rows))
  rng = random.Random(seed)
  return sorted(rng.sample(range(num_rows), keep))


def _keep_bernoulli(rows, fraction, seed):
  """Yield each row with probability *fraction*, reproducibly.

  The streaming counterpart of :func:`_sample_indices`, for sources whose
  length is unknown or that can only be read once.  A per-row Bernoulli keep
  needs no row count, but it only promises an *expected* ``fraction`` of the
  rows -- the realised count is reported in the artifact metadata.

  Note that this still consumes the whole source: without a known row count it
  cannot stop early, so it bounds what is *trained on*, not what is *read*.
  Use ``max_rows`` to bound the read itself.

  Args:
    rows: Iterable of rows.
    fraction: Fraction to keep, in ``(0.0, 1.0]``.
    seed: RNG seed.

  Yields:
    The kept rows.
  """
  rng = random.Random(seed)
  for row in rows:
    if rng.random() <= fraction:
      yield row


def _take(rows, max_rows):
  """Truncate *rows* to the first *max_rows* items (no-op when ``None``)."""
  if max_rows is None:
    return rows
  return itertools.islice(rows, max_rows)


def _select_rows(dataset, fraction, seed, max_rows=None, sampler=None):
  """Return the rows to train on, honouring fraction, cap and custom sampler.

  Dispatch is on :func:`is_random_access`, the existing capability check: an
  indexable source gets the exact-count path, anything else gets Bernoulli.

  Args:
    dataset: Map-style or streaming dataset of dict rows.
    fraction: Fraction to keep, or ``None`` for every row.
    seed: RNG seed.
    max_rows: Optional cap on rows *considered*, applied before *fraction*.
    sampler: Optional ``fn(dataset, fraction=..., seed=...)`` returning an
      iterable of rows, overriding the built-in strategy.

  Returns:
    A tuple ``(rows, described_mode)`` where *rows* is the iterable to consume.
  """
  if sampler is not None:
    logging.info("Using custom sampler %r.",
                 sampler.__name__ if hasattr(sampler, "__name__") else sampler)
    return sampler(dataset, fraction=fraction, seed=seed), "custom"

  rows = _take(iter(dataset), max_rows)

  if fraction is None:
    return rows, "all"
  if fraction >= 1.0:
    return rows, "all"

  if is_random_access(dataset):
    num_rows = min(len(dataset), max_rows) if max_rows is not None else len(dataset)
    indices = _sample_indices(num_rows, fraction, seed)
    logging.info("Random-access corpus: taking %d of %d rows (%.4g, seed=%d).",
                 len(indices), num_rows, fraction, seed)
    return (dataset[i] for i in indices), "index"

  logging.info(
      "Streaming corpus: keeping each row with probability %.4g (seed=%d); "
      "the whole source is still read.", fraction, seed)
  return _keep_bernoulli(rows, fraction, seed), "bernoulli"


def _iter_texts(rows, text_column):
  """Yield the *text_column* string from each row, skipping empty entries.

  Args:
    rows: Iterable of dict rows.
    text_column: Column holding the raw text.

  Yields:
    Non-empty text strings.
  """
  scanned = 0
  kept = 0
  for row in rows:
    scanned += 1
    text = row.get(text_column)
    if text:
      kept += 1
      yield text
  logging.info("Scanned %d rows, kept %d non-empty texts.", scanned, kept)


def _train_bpe(texts, vocab_size):
  """Train a byte-level BPE tokenizer with a fixed special-token layout.

  Mirrors the historic in-experiment recipe exactly (``ByteLevel`` without a
  prefix space, no ``min_frequency`` override) so vocabularies stay comparable
  with artifacts produced before this CLI existed.

  Args:
    texts: Iterable of training strings.
    vocab_size: Target vocabulary size.

  Returns:
    A trained ``tokenizers.Tokenizer``.
  """
  from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

  tok = Tokenizer(models.BPE(unk_token=_UNK))
  tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
  tok.decoder = decoders.ByteLevel()
  trainer = trainers.BpeTrainer(
      vocab_size=vocab_size,
      special_tokens=list(_SPECIAL_TOKENS),
      initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
  )
  tok.train_from_iterator(texts, trainer=trainer)
  logging.info("Trained BPE: vocab_size=%d (requested %d).", tok.get_vocab_size(),
               vocab_size)
  return tok


def _write_artifact(tok, out_dir, meta):
  """Write the tokenizer plus its config and provenance into *out_dir*.

  Args:
    tok: The trained tokenizer.
    out_dir: Destination directory (created if absent).
    meta: Provenance dict written to ``tokenizer_meta.json``.
  """
  os.makedirs(out_dir, exist_ok=True)
  tok.save(os.path.join(out_dir, _ARTIFACT_TOKENIZER))

  config = {
      "tokenizer_class": "PreTrainedTokenizerFast",
      "unk_token": _UNK,
      "pad_token": _PAD,
      "bos_token": _BOS,
      "eos_token": _EOS,
  }
  with open(os.path.join(out_dir, _ARTIFACT_CONFIG), "w") as f:
    json.dump(config, f, indent=2)
    f.write("\n")

  with open(os.path.join(out_dir, _ARTIFACT_META), "w") as f:
    json.dump(meta, f, indent=2)
    f.write("\n")


def train_tokenizer(dataset,
                    *,
                    text_column,
                    out_dir,
                    vocab_size=DEFAULT_VOCAB_SIZE,
                    fraction=None,
                    seed=0,
                    split="train",
                    dataset_name=None,
                    dataset_arg=None,
                    cache_dir=None,
                    streaming=False,
                    max_rows=None,
                    sampler=None,
                    force=False):
  """Train a BPE tokenizer on a dataset column and write a HF-format artifact.

  Args:
    dataset: HuggingFace dataset name or on-disk ``save_to_disk`` directory.
    text_column: Column holding the raw training text.
    out_dir: Destination directory for the artifact.
    vocab_size: Target vocabulary size.
    fraction: Fraction of rows to train on, in ``(0.0, 1.0]``; ``None`` uses
      every row.
    seed: RNG seed, so the same corpus and fraction give the same vocabulary.
    split: Dataset split to train on.
    dataset_name: Optional dataset configuration name (``--dataset_arg name=``).
    dataset_arg: Extra kwargs forwarded verbatim to ``datasets.load_dataset``.
    cache_dir: HuggingFace cache directory, normally shared with the training
      pipeline so the corpus is not downloaded twice.
    streaming: Load the corpus as a stream instead of materialising it.
    max_rows: Optional cap on rows considered, applied before *fraction*.
    sampler: Optional ``fn(dataset, fraction=..., seed=...)`` returning rows.
    force: Rebuild even when *out_dir* already holds a tokenizer.

  Returns:
    The path to the written artifact directory.
  """
  artifact = os.path.join(out_dir, _ARTIFACT_TOKENIZER)
  # Check before touching the dataset: re-iterating a web-scale stream is a
  # real download, so a skipped rebuild must cost nothing.
  if os.path.exists(artifact) and not force:
    logging.info("%s already exists; skipping build (use --force to rebuild).",
                 artifact)
    return out_dir

  fraction = _check_fraction(fraction)
  if vocab_size <= len(_SPECIAL_TOKENS):
    fatal(f"--vocab_size must exceed {len(_SPECIAL_TOKENS)}, got {vocab_size}",
          ValueError)
  if max_rows is not None and max_rows <= 0:
    fatal(f"--max_rows must be positive, got {max_rows}", ValueError)

  # ``load_raw_split`` takes the HF subset/config through *dataset_arg*, so a
  # standalone *dataset_name* is folded in rather than dropped.
  load_args = dict(dataset_arg or {})
  if dataset_name is not None and "name" not in load_args:
    load_args["name"] = dataset_name

  logging.info("Loading dataset %r (split=%r, streaming=%s).", dataset, split,
               streaming)
  ds = load_raw_split(dataset,
                      split,
                      cache_dir=cache_dir,
                      dataset_arg=load_args,
                      streaming=streaming)
  if text_column not in ds.column_names:
    fatal(
        f"Text column {text_column!r} not in dataset {dataset!r}. "
        f"Available: {sorted(ds.column_names)}.", ValueError)

  rows, mode = _select_rows(ds, fraction, seed, max_rows=max_rows, sampler=sampler)
  texts = list(_iter_texts(rows, text_column))
  if not texts:
    fatal(
        f"No training text selected from {dataset!r}; refusing to write "
        "an empty tokenizer.", ValueError)
  logging.info("Training on %d texts.", len(texts))

  tok = _train_bpe(texts, vocab_size)
  meta = {
      "dataset": dataset,
      "dataset_name": dataset_name,
      "split": split,
      "text_column": text_column,
      "fraction": fraction,
      "seed": seed,
      "strategy": mode,
      "streaming": bool(streaming),
      "max_rows": max_rows,
      "vocab_size": tok.get_vocab_size(),
      "vocab_size_requested": vocab_size,
      "special_tokens": list(_SPECIAL_TOKENS),
      "rows_trained": len(texts),
      "genml_kit_version": _package_version(),
      "created": datetime.datetime.now().astimezone().isoformat(),
  }
  _write_artifact(tok, out_dir, meta)
  logging.info("Wrote tokenizer artifact to %s.", out_dir)
  return out_dir
