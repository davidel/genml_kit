"""FieldSectorDataset — select and rename fields from a dict-returning dataset."""

from genml_kit.utils.logging import fatal


class FieldSectorDataset:
  """Wrap a dict-returning dataset and keep only the selected fields.

  Parameters
  ----------
  source : Dataset
      A dataset whose ``__getitem__`` returns a ``dict``.
  fields : dict[str, str]
      Mapping ``{SRC_NAME: DST_NAME}``.  For every item, the value at
      key *SRC_NAME* is picked from the source item and stored under
      *DST_NAME* in the returned dict.  All other fields are dropped.

  Example
  -------
  >>> ds = FieldSectorDataset(my_dataset, fields={"img": "image"})
  >>> ds[0]  # {"image": <value of my_dataset[0]["img"]>}
  """

  def __init__(self, source, fields):
    if not fields:
      fatal("FieldSectorDataset requires a non-empty 'fields' mapping.", ValueError)
    self._source = source
    self._fields = dict(fields)

  def __len__(self):
    return len(self._source)

  @property
  def fields(self):
    """The ``{SRC_NAME: DST_NAME}`` selection mapping."""
    return dict(self._fields)

  def __getitem__(self, idx):
    item = self._source[idx]
    return {dst: item[src] for src, dst in self._fields.items()}
