"""Text pipeline: HF corpora / instruction data for causal-LM training.

The text analogue of :mod:`genml_kit.pipelines.images`.  It owns the
loaders, the token-collation contract (``data={"input_ids",
"attention_mask"}``, ``meta={"labels": ...}``) and device transfer; it never
builds models or computes losses (the ``causal_lm`` method's job).

The tokenizer is resolved by the *method* during ``prepare_transforms``
(lifecycle position 1) and read here via ``method.tokenizer`` -- exactly how
``ImagesPipeline`` reads ``method.train_transforms``.
"""

import logging

import torch
from torch.utils.data import DataLoader

from genml_kit.datasets.text_proxy import load_raw_split
from genml_kit.pipelines.base import DataPipeline
from genml_kit.pipelines.registry import PIPELINES
from genml_kit.utils.logging import fatal
from genml_kit.utils.seed import seed_worker

#: Items sampled to estimate the token-per-character ratio of a corpus.  Small
#: enough to be free next to an epoch of training, large enough to average out
#: per-item tokenizer variance.
_TOKENS_PER_UNIT_PROBES = 64


def _estimate_tokens_per_unit(dataset, tokenizer):
  """Estimate tokens per character for *dataset*, for length-proxy batching.

    The packing sampler sizes batches from the cheap character-length proxy
    (``text_length_proxy``), so it needs the corpus' tokens-per-character ratio
    to convert that proxy to an approximate token count.  The ratio is measured
    on a small sample; it only steers how many documents fill a batch, never the
    hard window cap, so an imperfect estimate cannot break the memory bound.
    """
  n = len(dataset)
  if n == 0:
    return 1.0
  step = max(1, n // _TOKENS_PER_UNIT_PROBES)
  chars = 0
  tokens = 0
  for i in range(0, n, step):
    item = dataset[i]
    if "text" in item:
      text = item["text"] or ""
    else:
      text = (item.get("prompt") or "") + (item.get("response") or "")
    if not text:
      continue
    chars += len(text)
    tokens += len(tokenizer.encode(text, add_special_tokens=False)) or 1
  if chars == 0:
    return 1.0
  return tokens / chars


@PIPELINES.register
class TextPipeline(DataPipeline):
  """Text data pipeline: a HF corpus or instruction set (train/val split)."""

  NAME = "text"
  THROUGHPUT_UNIT = "sample"

  def __init__(self, **kwargs):
    super().__init__(**kwargs)
    self.train_dataset = None
    self.val_dataset = None
    self.train_collator = None
    self.val_collator = None
    self.data_generator = None
    self._built = False

  @classmethod
  def add_args(cls, parser):
    # ``--dataset``, ``--split``, ``--val_split``, ``--cache_dir``,
    # ``--max_seq_len``, ``--mask_prompt``, ``--pad_to_multiple_of`` and
    # ``--max_items`` are globally shared (genml_kit.utils.shared_args) and
    # registered once by the harness/entry point, so they are not re-declared
    # here.  The text pipeline consumes them: ``--dataset`` is the HF text
    # corpus, ``--split`` the training split.
    group = parser.add_argument_group("text pipeline")
    group.add_argument("--text_column",
                       type=str,
                       help="Column holding a raw LM sequence (plain LM mode).")
    group.add_argument("--instruction_column",
                       type=str,
                       help="Column holding the instruction/prompt "
                       "(instruction mode, with --response_column).")
    group.add_argument("--response_column",
                       type=str,
                       help="Column holding the response (instruction mode).")
    group.add_argument("--packing",
                       nargs="?",
                       const="wrapped",
                       default=None,
                       choices=["wrapped", "bfd"],
                       help="Pack sequences into fixed max_seq_len windows "
                       "(removes padding waste on short documents).  Bare "
                       "--packing = 'wrapped' (concatenate and slice); "
                       "'bfd' = best-fit-decreasing bin packing (fewer "
                       "windows, but reorders documents).")
    group.add_argument("--length_bucket",
                       action="store_true",
                       help="Batch items of similar length (length-bucketing "
                       "sampler) so padded batches waste little compute "
                       "without packing.  Mutually exclusive with --packing.")

  def build_loader(self, args, mode="train", *, needs_labels=None, **kwargs):
    """Build (and cache) the loader for *mode* (train/val).

        ``method`` supplies the tokenizer via ``method.tokenizer`` (resolved
        in ``prepare_transforms``).
        """
    method = kwargs.get("method")
    if not self._built:
      self._build(args, method=method)
    return self.val_loader if mode == "val" else self.train_loader

  def build_val_loader(self, args, **kwargs):
    return self.build_loader(args, mode="val", **kwargs)

  def _build_bucketed_train_loader(self, args):
    """Train loader using the length-bucketing batch sampler.

        Groups similar-length items into the same batch so the collator pads
        little, without packing and *without* any change to the attention
        mask (it stays exactly document-exact).
        """
    from genml_kit.datasets.length_bucket_sampler import (LengthBucketBatchSampler,
                                                          text_length_proxy)
    # One length proxy per item drives the bucketing; computed once here (not
    # per epoch) since it is a property of the data, not of the run.
    lengths = text_length_proxy(self.train_dataset)
    sampler = LengthBucketBatchSampler(lengths,
                                       batch_size=args.batch_size,
                                       drop_last=False,
                                       seed=args.seed)
    # ``batch_sampler`` fully determines the batches, so DataLoader forbids
    # passing ``batch_size`` / ``shuffle`` alongside it.
    return DataLoader(
        self.train_dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        pin_memory=False,
        collate_fn=self.train_collator,
    )

  def _build_packed_train_loader(self, args):
    """Train loader for ``wrapped`` packing, bounded by a token budget.

        ``--batch_size`` counts documents on every other path, but ``wrapped``
        packing emits ``ceil(total_tokens / max_seq_len)`` rows -- unbounded in
        the document count and a VRAM trap on long-document corpora.  Batch by a
        fixed token budget instead (:class:`PackingBatchSampler`) so a batch
        fills at most ``--batch_size`` windows, while a document longer than one
        window is still streamed in (its tail lands in a later window).
        """
    from genml_kit.datasets.length_bucket_sampler import text_length_proxy
    from genml_kit.datasets.packing_sampler import PackingBatchSampler
    lengths = text_length_proxy(self.train_dataset)
    sampler = PackingBatchSampler(lengths,
                                  batch_size=args.batch_size,
                                  max_seq_len=self._max_seq_len,
                                  tokens_per_unit=self._tokens_per_unit,
                                  seed=args.seed)
    # ``batch_sampler`` fully determines the batches, so DataLoader forbids
    # passing ``batch_size`` / ``shuffle`` alongside it.
    return DataLoader(
        self.train_dataset,
        batch_sampler=sampler,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        pin_memory=False,
        collate_fn=self.train_collator,
    )

  def _make_collator(self, tokenizer, batch_size):
    # Imported lazily: text_collator imports pipelines.contracts, which
    # triggers the pipelines package __init__ (and therefore this module's
    # registration) -- a module-scope import here would close that cycle.
    from genml_kit.training.text_collator import TextCollator
    windowing = self._sequence.windowing
    packing = windowing in ("wrapped", "bfd")
    # Only ``wrapped`` can emit more rows than documents, so only it needs the
    # hard window cap; ``bfd`` truncates per document already and the rest are
    # one row per document.
    max_windows = batch_size if windowing == "wrapped" else None
    return TextCollator(tokenizer,
                        max_seq_len=self._max_seq_len,
                        mask_prompt=self._mask_prompt,
                        packing=packing,
                        packing_strategy=windowing if packing else "wrapped",
                        boundary_loss=self._sequence.boundary_loss,
                        pad_to_multiple_of=self._pad_to_multiple_of,
                        max_windows=max_windows)

  def _build(self, args, method=None):
    if not args.dataset:
      fatal("The text pipeline requires --dataset.", ValueError)
    tokenizer = getattr(method, "tokenizer", None) if method is not None else None
    if tokenizer is None:
      fatal(
          "The text pipeline needs a tokenizer; the method's "
          "prepare_transforms must set method.tokenizer before the loaders "
          "are built.", ValueError)

    # Compile ``--sequence_mode`` and the windowing flags into one validated
    # object: the resolver owns the "packing and --length_bucket are mutually
    # exclusive" guard and the mode's boundary/framing policy, so no callee
    # reads the raw flags again.
    from genml_kit.training.sequence_mode import resolve_sequence_mode
    self._sequence = resolve_sequence_mode(args.sequence_mode, args.packing,
                                           args.length_bucket)
    self._max_seq_len = args.max_seq_len
    self._mask_prompt = args.mask_prompt
    self._pad_to_multiple_of = args.pad_to_multiple_of

    self.train_dataset, self.val_dataset = self._load_splits(args)
    self._tokens_per_unit = _estimate_tokens_per_unit(self.train_dataset, tokenizer)
    self.train_collator = self._make_collator(tokenizer, args.batch_size)
    self.val_collator = self._make_collator(tokenizer, args.batch_size)

    self.data_generator = torch.Generator()
    if args.seed is not None:
      self.data_generator.manual_seed(args.seed)

    windowing = self._sequence.windowing
    if windowing == "bucket":
      self.train_loader = self._build_bucketed_train_loader(args)
    elif windowing == "wrapped":
      self.train_loader = self._build_packed_train_loader(args)
    else:
      self.train_loader = DataLoader(
          self.train_dataset,
          batch_size=args.batch_size,
          shuffle=True,
          num_workers=args.num_workers,
          worker_init_fn=seed_worker,
          generator=self.data_generator,
          pin_memory=False,
          drop_last=windowing in ("wrapped", "bfd"),
          collate_fn=self.train_collator,
      )
    if self.val_dataset is not None:
      self.val_loader = DataLoader(
          self.val_dataset,
          batch_size=args.batch_size,
          shuffle=False,
          num_workers=args.num_workers,
          worker_init_fn=seed_worker,
          pin_memory=False,
          collate_fn=self.val_collator,
      )
    self._built = True
    logging.info("Text pipeline ready: %d train / %s val items.",
                 len(self.train_dataset),
                 len(self.val_dataset) if self.val_dataset is not None else "no")

  def _load_splits(self, args):
    """Load the train proxy and, when available, the val proxy.

        The available validation source is tried in order: the ``--val_split``
        held-out fraction of the training split, then a ``validation`` split on
        the hub.  Nothing is downloaded twice.
        """
    text_column = args.text_column
    prompt_column = args.instruction_column
    response_column = args.response_column

    def _wrap(hf, cap=None):
      from genml_kit.datasets.capped import maybe_cap
      from genml_kit.datasets.text_proxy import TextDatasetProxy
      return TextDatasetProxy(maybe_cap(hf, cap),
                              text_column=text_column,
                              prompt_column=prompt_column,
                              response_column=response_column)

    val_split = args.val_split
    if val_split:
      # Held-out fraction of the training split -- one load, split in memory.
      # ``--max_items`` caps the dataset *before* carving the val fraction, so
      # a budgeted run still yields the requested number of training rows plus a
      # real held-out set (previously the cap was silently ignored here).
      full = _wrap(load_raw_split(args.dataset,
                                  args.split,
                                  cache_dir=args.cache_dir,
                                  dataset_arg=args.dataset_arg),
                   cap=args.max_items).dataset
      n = len(full)
      n_val = max(1, int(n * val_split))
      if n_val >= n:
        fatal(f"--val_split={val_split} leaves no training rows "
              f"(dataset has {n}).", ValueError)
      return _wrap(full.select(range(n - n_val))), _wrap(
          full.select(range(n - n_val, n)))

    train = _wrap(load_raw_split(args.dataset,
                                 args.split,
                                 cache_dir=args.cache_dir,
                                 dataset_arg=args.dataset_arg),
                  cap=args.max_items)
    try:
      val_raw = load_raw_split(args.dataset,
                               "validation",
                               cache_dir=args.cache_dir,
                               dataset_arg=args.dataset_arg)
      val = _wrap(val_raw, cap=args.max_items)
    except (ValueError, KeyError, FileNotFoundError) as exc:
      logging.info("No validation split for %s (%s); val_loader disabled.",
                   args.dataset, exc)
      val = None
    return train, val

  def to_device(self, blob, device):
    """Move ``input_ids`` / ``attention_mask`` and ``meta['labels']``."""
    data = {k: v.to(device, non_blocking=True) for k, v in blob.data.items()}
    meta = dict(blob.meta)
    labels = meta.get("labels")
    if labels is not None:
      meta["labels"] = labels.to(device, non_blocking=True)
    return blob._replace(data=data, meta=meta)
