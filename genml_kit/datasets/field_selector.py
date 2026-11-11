"""Field selection over dict-returning datasets.

:class:`FieldSelectorDataset` and :class:`IterableFieldSelectorDataset` pick and
rename fields from a source whose items are ``dict``, keeping only the requested
ones.  They differ in what they wrap, not in what they select:

* :class:`FieldSelectorDataset` wraps a random-access dataset (``__getitem__`` +
  ``__len__``) and forwards both.
* :class:`IterableFieldSelectorDataset` wraps a one-pass stream.  It subclasses
  :class:`~genml_kit.datasets.iterable.StreamingDataset`, which is what makes
  ``isinstance(wrapper, torch.utils.data.IterableDataset)`` true -- the only
  thing ``DataLoader`` consults when choosing its fetch path.  A wrapper that
  merely *defines* ``__iter__`` is not enough: torch decides the dataset kind by
  ``isinstance`` at construction, and our own ``is_iterable`` helper is never
  called by it.

Both fail loudly on a missing source field (``KeyError``) rather than yielding a
silently partial item.
"""

from genml_kit.datasets.iterable import StreamingDataset
from genml_kit.utils.logging import fatal


def _validate_fields(fields):
  """Reject an empty mapping: a selector that selects nothing is a bug."""
  if not fields:
    fatal("a field selector requires a non-empty 'fields' mapping.", ValueError)
  return dict(fields)


class FieldSelectorDataset:
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

  def __init__(self, source, fields):
    self._source = source
    self._fields = _validate_fields(fields)

  @property
  def fields(self):
    """The ``{SRC_NAME: DST_NAME}`` selection mapping."""
    return dict(self._fields)

  def _pick(self, item):
    return {dst: item[src] for src, dst in self._fields.items()}

  def __len__(self):
    return len(self._source)

  def __getitem__(self, idx):
    return self._pick(self._source[idx])

  def __iter__(self):
    for item in self._source:
      yield self._pick(item)


class IterableFieldSelectorDataset(StreamingDataset):
  """Select and rename fields of a one-pass streaming dataset.

  Same selection contract as :class:`FieldSelectorDataset`, for a source that
  can only be iterated.  It is a :class:`StreamingDataset`, so:

  * ``DataLoader`` sees an ``IterableDataset`` and takes its iterable fetch path;
  * the strided worker split is inherited, applied *after* field selection, so
    workers stay balanced;
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
    self._source = source
    self._fields = _validate_fields(fields)
    #: Human-readable name, forwarded so ensemble summaries label the member.
    self.name = getattr(source, "name", type(source).__name__)

  @property
  def fields(self):
    """The ``{SRC_NAME: DST_NAME}`` selection mapping."""
    return dict(self._fields)

  def _pick(self, item):
    return {dst: item[src] for src, dst in self._fields.items()}

  def source_iter(self):
    for item in self._source:
      yield self._pick(item)

  def set_epoch(self, epoch):
    super().set_epoch(epoch)
    # Forward so a source that reshuffles per epoch stays consistent.
    if hasattr(self._source, "set_epoch"):
      self._source.set_epoch(epoch)
