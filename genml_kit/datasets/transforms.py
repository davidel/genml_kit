"""Dict-dataset transform utilities shared across tasks.

Promoted from ``pretrain/cli.py::_TransformWrapper`` (plan §12.6, item 1)
and extended for multi-field (pair) datasets: the VO pipeline applies one
augmentation to both frames of a pair through the same wrapper the
pre-training harness uses for its single image field.
"""

from torch import nn

from genml_kit.utils.logging import fatal


class DictFieldTransform(nn.Module):
  """Apply a transform to selected fields of a dict-returning dataset.

  Parameters
  ----------
  dataset : Dataset
      A dataset whose ``__getitem__`` returns a ``dict``.
  transform : callable
      Applied to every field listed in *fields* that is present in the
      item; absent fields are passed through untouched.
  fields : tuple[str, ...]
      The item keys to transform.

  Notes
  -----
  The single-field case used by the pre-training harness is
  ``DictFieldTransform(ds, t, fields=("image",))``; a VO pair dataset
  uses ``fields=("A", "B")`` so both frames see identical augmentation.
  """

  def __init__(self, dataset, transform, fields=("image",)):
    super().__init__()
    if not fields:
      fatal("DictFieldTransform requires a non-empty 'fields' tuple.", ValueError)
    self._ds = dataset
    self._t = transform
    self._fields = tuple(fields)

  def __len__(self):
    return len(self._ds)

  @property
  def fields(self):
    """The transformed field names."""
    return self._fields

  def forward(self, idx):
    # Copy: never mutate the source row.
    row = dict(self._ds[idx])
    for f in self._fields:
      if f in row:
        row[f] = self._t(row[f])
    return row

  # For backwards compatibility with Dataset interface
  def __getitem__(self, idx):
    return self.forward(idx)
