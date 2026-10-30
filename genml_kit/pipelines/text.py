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
                       action="store_true",
                       help="Pack sequences into fixed max_seq_len windows "
                       "(removes padding waste on short documents).")

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

  def _make_collator(self, tokenizer):
    # Imported lazily: text_collator imports pipelines.contracts, which
    # triggers the pipelines package __init__ (and therefore this module's
    # registration) -- a module-scope import here would close that cycle.
    from genml_kit.training.text_collator import TextCollator
    return TextCollator(tokenizer,
                        max_seq_len=self._max_seq_len,
                        mask_prompt=self._mask_prompt,
                        packing=self._packing,
                        pad_to_multiple_of=self._pad_to_multiple_of)

  def _build(self, args, method=None):
    if not args.dataset:
      fatal("The text pipeline requires --dataset.", ValueError)
    tokenizer = getattr(method, "tokenizer", None) if method is not None else None
    if tokenizer is None:
      fatal(
          "The text pipeline needs a tokenizer; the method's "
          "prepare_transforms must set method.tokenizer before the loaders "
          "are built.", ValueError)

    self._max_seq_len = args.max_seq_len
    self._mask_prompt = args.mask_prompt
    self._packing = args.packing
    self._pad_to_multiple_of = args.pad_to_multiple_of

    self.train_dataset, self.val_dataset = self._load_splits(args)
    self.train_collator = self._make_collator(tokenizer)
    self.val_collator = self._make_collator(tokenizer)

    self.data_generator = torch.Generator()
    if args.seed is not None:
      self.data_generator.manual_seed(args.seed)

    self.train_loader = DataLoader(
        self.train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        generator=self.data_generator,
        pin_memory=False,
        drop_last=self._packing,
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
      full = _wrap(load_raw_split(args.dataset, args.split,
                                  cache_dir=args.cache_dir)).dataset
      n = len(full)
      n_val = max(1, int(n * val_split))
      if n_val >= n:
        fatal(f"--val_split={val_split} leaves no training rows "
              f"(dataset has {n}).", ValueError)
      return _wrap(full.select(range(n - n_val))), _wrap(
          full.select(range(n - n_val, n)))

    train = _wrap(load_raw_split(args.dataset, args.split, cache_dir=args.cache_dir),
                  cap=args.max_items)
    try:
      val_raw = load_raw_split(args.dataset, "validation", cache_dir=args.cache_dir)
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
