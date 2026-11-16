"""Dataset capability contract: iterate first, index only if you can.

``genml_kit`` used to assume every training corpus was *random-access*
(``__len__`` + ``__getitem__``).  The contract here is the inverse: the core
iterates, and random access is merely one *capability* a source may also have.

A source is classified by what it can actually do, via duck typing -- there is
no required base class and no inheritance refactor:

* **iterable** -- defines ``__iter__``; the only thing the core needs.
* **random-access** -- defines ``__len__`` *and* ``__getitem__``, so it can be
  indexed and is therefore also iterable in index order.

:class:`IterableMixin` gives any random-access source the default ``__iter__``
for free, so existing map-style datasets satisfy the iterable contract without
change.  :func:`num_samples` is a *best-effort estimate* used purely for human
facing progress output -- **no code may break when it is absent or wrong**.
"""


class IterableMixin:
  """Default ``__iter__`` for random-access datasets (yield in index order).

    Any object exposing ``__len__`` + ``__getitem__`` becomes iterable for free,
    so the core never needs to know which kind of source it holds:

    >>> class Numbers(IterableMixin):
    ...   def __len__(self):
    ...     return 3
    ...   def __getitem__(self, i):
    ...     return i * 10
    >>> list(Numbers())
    [0, 10, 20]
    """

  def __iter__(self):
    for i in range(len(self)):
      yield self[i]


def is_random_access(obj):
  """Return True if *obj* can be indexed and sized.

    Both halves are required: a ``__getitem__`` without ``__len__`` cannot be
    enumerated, and a ``__len__`` without ``__getitem__`` (e.g. a stream that
    declares a count) cannot be indexed.

    A proxy that *defines* ``__len__`` but raises ``TypeError`` to signal
    "streaming, no length" is **not** random-access.  Presence of the method is
    therefore not enough: it must also answer.

    Args:
      obj: Any dataset-like object.

    Returns:
      bool: True when *obj* can actually be sized and indexed.
    """
  if isinstance(obj, type):
    # A class exposes ``__getitem__`` through the descriptor protocol.
    return False
  if not (hasattr(obj, "__len__") and hasattr(obj, "__getitem__")):
    return False
  try:
    len(obj)
  except TypeError:
    return False
  return True


def is_iterable(obj):
  """Return True if *obj* can be iterated with ``for ... in ...``.

    This is the broadest of three overlapping capability checks, and the only
    duck-typed one:

    * :func:`is_iterable` (here) -- any object with ``__iter__``; random access
      implies it, but so does a bare stream.
    * :func:`genml_kit.datasets.iterable.is_streaming_source` -- ``isinstance``
      against *torch*'s ``IterableDataset`` (a ``DataLoader`` uses this to pick
      its fetch path).
    * :func:`genml_kit.datasets.hf_proxy.is_hf_stream` -- ``isinstance`` against
      HuggingFace ``datasets.IterableDataset``, which is *not* a torch
      ``IterableDataset``.

    Prefer :func:`is_random_access` when the question is "can I index it?": a
    stream that merely defines ``__iter__`` is not random-access.

    Args:
      obj: Any dataset-like object.

    Returns:
      bool: True when ``__iter__`` is present.
    """
  return hasattr(obj, "__iter__")


def is_shuffleable(obj):
  """Return True when shuffling by index is meaningful for *obj*.

    Index shuffling needs a materialisable index space, i.e. random access.  A
    one-pass stream cannot be shuffled this way; it is shuffled at the batch
    level instead (see the round-robin buffer) or not at all.

    Args:
      obj: Any dataset-like object.

    Returns:
      bool: True when *obj* is random-access.
    """
  return is_random_access(obj)


def num_samples(obj):
  """Best-effort **sample** count for *obj*, or None when it cannot be known.

    Purely informational: this feeds progress output only.  Never use it to
    size a loop, and never let its absence or inaccuracy break a run.

    The returned value always counts *samples* (items), never batches.  Callers
    that need a batch count (the trainer's progress denominator) obtain it from
    :func:`genml_kit.datasets.iterable.safe_len` first and only fall back to
    this estimate, converting samples to batches with the loader's batch size.

    Resolution order:

    1. a loader-like ``obj`` that is not itself indexable (no ``__getitem__``)
       but exposes a ``.dataset`` -- recurse into that dataset, so a size-less
       ``DataLoader`` around a stream still surfaces the stream's estimate;
    2. ``len(obj)`` when the source can be sized (exact, random access);
    3. an ``obj.num_samples()`` method when the source computes its own estimate
       (streams and wrappers that forward or derive a count);
    4. ``None`` -- unknown, which callers must tolerate.

    Args:
      obj: Any dataset-like object (a dataset, a wrapper, or a loader).

    Returns:
      int | None: The estimated sample count, or None when unknown.
    """
  # A loader (``DataLoader`` / its sharded wrapper) has no ``__getitem__`` and
  # carries no samples of its own; recurse into the dataset it wraps so
  # ``num_samples`` speaks in samples for a loader too.  An indexable object
  # (a dataset, or a ``CappedDataset`` whose cap only its own ``len`` reflects)
  # is skipped here so its exact count wins instead.
  if not hasattr(obj, "__getitem__"):
    inner = getattr(obj, "dataset", None)
    if inner is not None and inner is not obj:
      return num_samples(inner)
  try:
    return len(obj)
  except TypeError:
    pass
  method = getattr(obj, "num_samples", None)
  if callable(method):
    return method()
  return None
