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
from genml_kit.pipelines.contracts import DataBlob
from genml_kit.pipelines.registry import PIPELINES
from genml_kit.utils.logging import fatal
from genml_kit.utils.seed import seed_worker


@PIPELINES.register
class TextPipeline(DataPipeline):
  """Text data pipeline: a HF corpus or instruction set (train/val split)."""

  NAME = "text"

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
    # ``--dataset``, ``--split``, ``--val_split`` and ``--cache_dir`` are owned
    # by the always-registered ``images`` pipeline and must not be re-declared
    # (argparse option strings are global to the single parser).  The text
    # pipeline consumes them: ``--dataset`` is the HF text corpus, ``--split``
    # the training split.
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
    group.add_argument("--max_seq_len",
                       type=int,
                       default=512,
                       help="Token sequence length (truncation / packing "
                       "window).")
    group.add_argument("--mask_prompt",
                       action="store_true",
                       help="Mask prompt tokens out of the loss "
                       "(instruction mode).")
    group.add_argument("--packing",
                       action="store_true",
                       help="Pack sequences into fixed max_seq_len windows "
                       "(removes padding waste on short documents).")
    group.add_argument("--pad_to_multiple_of",
                       type=int,
                       help="Round the padded length up to this multiple "
                       "(e.g. 8 for tensor-core friendliness).")
    group.add_argument("--max_items",
                       type=int,
                       help="Cap the number of loaded items (smoke tests).")

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
    if not getattr(args, "dataset", None):
      fatal("The text pipeline requires --dataset.", ValueError)
    tokenizer = getattr(method, "tokenizer", None) if method is not None else None
    if tokenizer is None:
      fatal(
          "The text pipeline needs a tokenizer; the method's "
          "prepare_transforms must set method.tokenizer before the loaders "
          "are built.", ValueError)

    self._max_seq_len = getattr(args, "max_seq_len", 512)
    self._mask_prompt = getattr(args, "mask_prompt", False)
    self._packing = getattr(args, "packing", False)
    self._pad_to_multiple_of = getattr(args, "pad_to_multiple_of", None)

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
    text_column = getattr(args, "text_column", None)
    prompt_column = getattr(args, "instruction_column", None)
    response_column = getattr(args, "response_column", None)

    def _wrap(hf, cap=None):
      from genml_kit.datasets.text_proxy import TextDatasetProxy
      return TextDatasetProxy(hf,
                              text_column=text_column,
                              prompt_column=prompt_column,
                              response_column=response_column,
                              max_items=cap)

    val_split = getattr(args, "val_split", None)
    if val_split:
      # Held-out fraction of the training split -- one load, split in memory.
      full = _wrap(
          load_raw_split(args.dataset,
                         getattr(args, "split", "train"),
                         cache_dir=getattr(args, "cache_dir", None))).dataset
      n = len(full)
      n_val = max(1, int(n * val_split))
      if n_val >= n:
        fatal(f"--val_split={val_split} leaves no training rows "
              f"(dataset has {n}).", ValueError)
      return _wrap(full.select(range(n - n_val))), _wrap(
          full.select(range(n - n_val, n)))

    train = _wrap(load_raw_split(args.dataset,
                                 getattr(args, "split", "train"),
                                 cache_dir=getattr(args, "cache_dir", None)),
                  cap=getattr(args, "max_items", None))
    try:
      val_raw = load_raw_split(args.dataset,
                               "validation",
                               cache_dir=getattr(args, "cache_dir", None))
      val = _wrap(val_raw, cap=getattr(args, "max_items", None))
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
    return DataBlob(data, meta)
