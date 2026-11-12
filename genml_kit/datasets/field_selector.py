"""Field selection over dict-returning datasets.

:class:`FieldSelectorDataset` and :class:`IterableFieldSelectorDataset` pick and
rename fields from a source whose items are ``dict``, keeping only the requested
ones.  They differ in what they wrap, not in what they select:

* :class:`FieldSelectorDataset` wraps a random-access dataset (``__getitem__`` +
  ``__len__``) and forwards both.
* :class:`IterableFieldSelectorDataset` wraps a one-pass stream.  It subclasses
  :class:`~genml_kit.datasets.iterable.StreamingTransform`, which is what makes
  ``isinstance(wrapper, torch.utils.data.IterableDataset)`` true -- the only
  thing ``DataLoader`` consults when choosing its fetch path.  A wrapper that
  merely *defines* ``__iter__`` is not enough: torch decides the dataset kind by
  ``isinstance`` at construction, and our own ``is_iterable`` helper is never
  called by it.  As a transform it forwards the source's worker split instead of
  splitting a second time.

Both fail loudly on a missing source field (``KeyError``) rather than yielding a
silently partial item.
"""

from genml_kit.datasets.iterable import StreamingTransform
from genml_kit.utils.logging import fatal


def _validate_fields(fields):
  """Reject an empty mapping: a selector that selects nothing is a bug."""
  if not fields:
    fatal("a field selector requires a non-empty 'fields' mapping.", ValueError)
  return dict(fields)


class _FieldSelectionMixin:
  """Select-and-rename core shared by the map-style and streaming selectors.

  Holds the source, the validated ``{SRC_NAME: DST_NAME}`` mapping, and the
  per-item pick.  The two concrete classes differ only in how they read the
  source (``__getitem__`` vs ``__iter__``), not in what they select.
  """

  def __init__(self, source, fields):
    self._source = source
    self._fields = _validate_fields(fields)

  @property
  def fields(self):
    """The ``{SRC_NAME: DST_NAME}`` selection mapping."""
    return dict(self._fields)

  def _pick(self, item):
    return {dst: item[src] for src, dst in self._fields.items()}


class FieldSelectorDataset(_FieldSelectionMixin):
  """Select and rename fields of a random-access dict-returning dataset.

  Parameters
  ----------
  source : Dataset
      A dataset whose items are ``dict`` and which supports ``__len__`` and
      ``__getitem__``.
  fields : dict[str, str]
      Mapping ``{SRC_NAME: DST_NAME}``.  For every item, the value at
      key *SRC_NAME* is picked from the source item and stored under
      *DST_NAME* in the returned dict.  All other fields are dropped.  A source
      item missing *SRC_NAME* raises ``KeyError``: a requested field that is
      absent is a misconfiguration, and failing loudly beats feeding a silently
      wrong column to the model.

  Example
  -------
  >>> ds = FieldSelectorDataset(my_dataset, fields={"img": "image"})
  >>> ds[0]  # {"image": <value of my_dataset[0]["img"]>}
  """

  def __len__(self):
    return len(self._source)

  def __getitem__(self, idx):
    return self._pick(self._source[idx])

  def __iter__(self):
    for item in self._source:
      yield self._pick(item)


class IterableFieldSelectorDataset(_FieldSelectionMixin, StreamingTransform):
  """Select and rename fields of a one-pass streaming dataset.

  Same selection contract as :class:`FieldSelectorDataset`, for a source that
  can only be iterated.  It is a :class:`StreamingTransform`, so:

  * ``DataLoader`` sees an ``IterableDataset`` and takes its iterable fetch path;
  * it forwards the source's worker split instead of applying its own, so the
    split happens exactly once at the leaf (not ``1 / num_workers**2``);
  * the length is unknown, so ``len()`` raises ``TypeError`` (which
    ``num_samples`` and ``is_random_access`` handle).

  ``set_epoch`` is forwarded to the source when it has one, so a source that
  reshuffles per epoch stays consistent.

  Parameters
  ----------
  source : IterableDataset
      A dataset that can be iterated and yields ``dict`` items.
  fields : dict[str, str]
      Mapping ``{SRC_NAME: DST_NAME}``, as in :class:`FieldSelectorDataset`.
  """

  #: A field selector only picks and renames; it never introduces labels, so it
  #: reports what an ensemble summary needs from an unlabeled member.
  has_labels = False
  #: No schema to count after selection.
  num_labels = 0
  label_names = ()

  def __init__(self, source, fields):
    super().__init__(source, fields)
    #: Human-readable name, forwarded so ensemble summaries label the member.
    self.name = getattr(source, "name", type(source).__name__)

  def source_iter(self):
    for item in self._source:
      yield self._pick(item)
