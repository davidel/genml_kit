"""Preference pipeline: chosen/rejected pairs for alignment training.

The alignment analogue of :mod:`genml_kit.pipelines.text`.  It owns the
loaders, the preference-collation contract (four token tensors in ``data``
plus ``chosen_labels`` / ``rejected_labels`` in ``meta``) and device
transfer; it never builds models or computes losses (the ``dpo`` /
``reward_model`` methods' job).

The tokenizer is resolved by the *method* during ``prepare_transforms`` and
read here via ``method.tokenizer`` -- exactly how ``TextPipeline`` reads it.
"""

import logging

import torch
from torch.utils.data import DataLoader

from genml_kit.datasets.preference_proxy import load_raw_split
from genml_kit.pipelines.base import DataPipeline
from genml_kit.pipelines.contracts import DataBlob
from genml_kit.pipelines.registry import PIPELINES
from genml_kit.utils.logging import fatal
from genml_kit.utils.seed import seed_worker


@PIPELINES.register
class PreferencePipeline(DataPipeline):
  """Preference data pipeline: chosen/rejected pairs (train/val split)."""

  NAME = "preference"

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
    # Only flags no other owner registers.  ``--dataset`` / ``--split`` /
    # ``--val_split`` / ``--cache_dir`` belong to the ``images`` pipeline and
    # ``--max_seq_len`` / ``--mask_prompt`` / ``--pad_to_multiple_of`` /
    # ``--max_items`` to the ``text`` pipeline -- argparse option strings are
    # global to the single shared parser, so re-declaring them here would
    # raise ``ArgumentError``.  They are read via ``getattr`` instead.
    group = parser.add_argument_group("preference pipeline")
    group.add_argument("--chosen_column",
                       type=str,
                       default="chosen",
                       help="Column holding the preferred response.")
    group.add_argument("--rejected_column",
                       type=str,
                       default="rejected",
                       help="Column holding the dispreferred response.")
    group.add_argument("--prompt_column",
                       type=str,
                       help="Optional shared prompt column (masked out of "
                       "the loss when --mask_prompt is set).")

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
    # Imported lazily: preference_collator imports pipelines.contracts, which
    # triggers the pipelines package __init__ (and therefore this module's
    # registration) -- a module-scope import here would close that cycle.
    from genml_kit.training.preference_collator import PreferenceCollator
    return PreferenceCollator(tokenizer,
                              max_seq_len=self._max_seq_len,
                              mask_prompt=self._mask_prompt,
                              pad_to_multiple_of=self._pad_to_multiple_of)

  def _build(self, args, method=None):
    if not getattr(args, "dataset", None):
      fatal("The preference pipeline requires --dataset.", ValueError)
    tokenizer = getattr(method, "tokenizer", None) if method is not None else None
    if tokenizer is None:
      fatal(
          "The preference pipeline needs a tokenizer; the method's "
          "prepare_transforms must set method.tokenizer before the loaders "
          "are built.", ValueError)

    self._max_seq_len = getattr(args, "max_seq_len", 512)
    self._mask_prompt = getattr(args, "mask_prompt", False)
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
    logging.info("Preference pipeline ready: %d train / %s val items.",
                 len(self.train_dataset),
                 len(self.val_dataset) if self.val_dataset is not None else "no")

  def _load_splits(self, args):
    """Load the train proxy and, when available, the val proxy.

        The available validation source is tried in order: the ``--val_split``
        held-out fraction of the training split, then a ``validation`` split on
        the hub.  Nothing is downloaded twice.
        """
    chosen_column = getattr(args, "chosen_column", "chosen")
    rejected_column = getattr(args, "rejected_column", "rejected")
    prompt_column = getattr(args, "prompt_column", None)

    def _wrap(hf, cap=None):
      from genml_kit.datasets.preference_proxy import PreferenceDatasetProxy
      return PreferenceDatasetProxy(hf,
                                    chosen_column=chosen_column,
                                    rejected_column=rejected_column,
                                    prompt_column=prompt_column,
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
    """Move the four token tensors and both label tensors to *device*."""
    data = {k: v.to(device, non_blocking=True) for k, v in blob.data.items()}
    meta = dict(blob.meta)
    for key in ("chosen_labels", "rejected_labels"):
      labels = meta.get(key)
      if labels is not None:
        meta[key] = labels.to(device, non_blocking=True)
    return DataBlob(data, meta)
