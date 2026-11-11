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
  """Best-effort sample count for *obj*, or None when it cannot be known.

    Purely informational: this feeds progress output only.  Never use it to
    size a loop, and never let its absence or inaccuracy break a run.

    Resolution order:

    1. ``len(obj)`` when the source can be sized (exact, random access);
    2. an ``nsamples`` attribute when the source declares an estimate (e.g. an
       ``nsamples`` dataset-arg for a stream);
    3. ``None`` -- unknown, which callers must tolerate.

    Args:
      obj: Any dataset-like object.

    Returns:
      int | None: The estimated sample count, or None when unknown.
    """
  try:
    return len(obj)
  except TypeError:
    return getattr(obj, "nsamples", None)
