"""Streaming source over webdataset tar shards.

A webdataset "shard" is a tar archive whose members are grouped into samples:
each member's name is ``<sample_key>.<extension>``, so ``s0001.png`` and
``s0001.cls`` are two fields of one sample.  A shard list such as
``train-{0000..0009}.tar`` is expanded by webdataset's brace expansion.

This class yields **whole samples**: one ``{member_name: decoded_value}`` dict
per sample, with every member decoded by webdataset according to its extension
(``png``/``jpg`` -> image, ``txt`` -> str, ``cls`` -> int, ``json`` -> dict,
``npy`` -> ndarray, ``pth`` -> Tensor, anything else -> bytes).  Selecting or
renaming fields is a separate concern, handled by ``FieldSelectorDataset`` on top::

    raw = WebDatasetSource("data/train-{0000..0009}.tar")
    ds = IterableFieldSelectorDataset(raw, fields={"png": "image", "cls": "label"})

Nothing here is image-specific: a text corpus with a single ``txt`` member, an
instruction-tuning set of ``prompt``/``response`` members, and a feature dump of
``npy`` members all work the same way.

What this class owns
--------------------
Webdataset's defaults conflict with the rest of the toolkit, and a user cannot
reach past them from the outside:

1. **No shard- or item-level splitting.**  ``wds.WebDataset`` installs a
   ``nodesplitter`` and a ``workersplitter`` by default.  The node splitter
   *raises* whenever a process group has more than one rank, telling the user to
   fix a pipeline this code owns; the worker splitter slices items
   (``islice(src, worker, None, num_workers)``) where the rest of the toolkit
   splits them strided (``i % num_workers == worker_id``).  Both are disabled
   here, so the whole stream reaches :class:`StreamingDataset`, whose ``__iter__``
   does the strided split, and the rank split is done at batch level by
   ``utils.distributed._ShardedBatchLoader``.  One split, in one place.

2. **No shuffling.**  webdataset's ``shuffle`` stage sits *before* batching and
   seeds itself from PID and wall-clock time unless told otherwise, so it is
   neither reproducible nor rank-consistent.  Shuffling for streams is done by
   the round-robin buffer in ``_ShardedBatchLoader``, at batch granularity,
   from one seeded permutation shared by every rank.

3. **One epoch is one pass to exhaustion.**  ``wds`` has ``with_epoch(n)``, but
   that truncates at ``n`` and, when ``n`` is not a multiple of the batch size,
   restarts the stream and *repeats* samples to reach the count.  The toolkit's
   epoch is exhaustion of the stream, so no length is imposed here.

Images arrive as ``image_format`` (RGB by default) because ``decode`` takes a
single spec applied to every image member: the source mode is not preserved, and
the spec is what fixes it.  Any other member is passed through in whatever type
webdataset decoded it to.
"""

from PIL import Image

from genml_kit.datasets.iterable import StreamingDataset
from genml_kit.utils.logging import fatal

#: Sample fields webdataset adds itself; kept, but never treated as an image.
_META_PREFIX = "__"

#: PIL modes accepted as an ``image_format``.  A short list on purpose: these are
#: the modes a training pipeline has a use for, and a typo should fail at
#: construction rather than on the first sample.
_IMAGE_FORMATS = ("RGB", "RGBA", "L", "LA", "P", "CMYK")


def _require_webdataset():
  """Import ``webdataset``, or fail with the extra that provides it."""
  try:
    import webdataset as wds
  except ImportError:
    fatal(
        "Streaming tar shards require the 'webdataset' package. "
        "Install it with: pip install genml_kit[webdataset]", ImportError)
  return wds


def _convert_image(value, image_format):
  """Convert one decoded image member to a PIL image of *image_format*."""
  if value.mode == image_format:
    return value
  return value.convert(image_format)


class WebDatasetSource(StreamingDataset):
  """Iterate a webdataset shard list as dictionaries of decoded members.

    Each yielded item is the whole sample: ``{member_name: decoded_value}`` for
    every member, including webdataset's own ``__key__``, ``__url__`` and
    ``__local_path__`` entries, which identify the sample and are useful when
    debugging a corpus.  Feed the result through ``FieldSelectorDataset`` to pick
    and rename columns.

    Images are converted to *image_format*; every other member is passed through
    in the type webdataset decoded it to (``str`` for text, ``int`` for ``cls``,
    ``dict`` for JSON, ``ndarray`` for ``npy``, ``Tensor`` for ``pth``).

    The shard list is re-expanded and the tars re-opened on every iteration, so
    the source is re-iterable: epoch *N+1* starts from the top of the shard list
    exactly as epoch *N* did.

    Args:
      urls: A shard path or a brace-expanded pattern
        (``shards-{0000..0009}.tar``), or an explicit list of them.
      image_format: PIL mode every image member is converted to.  Only affects
        members webdataset decodes as images.
      nsamples: Optional estimate of the sample count, for progress output only;
        pass it when the pipeline should render ``[i/~N]`` rather than ``[i/?]``.
      transform: Optional callable applied to every image member before it is
        yielded.
    """

  def __init__(self,
               urls,
               *,
               image_format="RGB",
               nsamples=None,
               transform=None):
    if image_format not in _IMAGE_FORMATS:
      fatal(f"Unsupported image_format {image_format!r}; "
            f"expected one of {', '.join(_IMAGE_FORMATS)}.", ValueError)
    self.urls = urls
    self.image_format = image_format
    self.transform = transform
    if nsamples is not None:
      self.nsamples = nsamples

  def _pipeline(self):
    """Build the webdataset pipeline for one pass over the shards."""
    wds = _require_webdataset()
    pipeline = wds.WebDataset(
        self.urls,
        # Both splitters off: this class and _ShardedBatchLoader do the
        # splitting, at item and batch granularity respectively.
        nodesplitter=None,
        workersplitter=None,
        # Shard order is stable; per-epoch reordering would be a second,
        # unseeded shuffle competing with the round-robin buffer.
        shardshuffle=False,
        empty_check=False)
    # The decode spec is fixed at "pil": it yields a PIL image for every image
    # member, and ``image_format`` then decides the mode.  webdataset's own spec
    # strings cannot express "PIL, mode chosen later", so converting after the
    # fact is the only way to keep one knob.
    return pipeline.decode("pil")

  def source_iter(self):
    """Yield one ``{member_name: decoded_value}`` dict per sample."""
    for sample in self._pipeline():
      out = {}
      for name, value in sample.items():
        if name.startswith(_META_PREFIX):
          # webdataset's own bookkeeping; kept as-is, never an image.
          out[name] = value
          continue
        # Only members webdataset *decoded as images* are images.  Keying on the
        # Python type instead would corrupt every array-valued member: a ``npy``
        # feature dump decodes to an ndarray and must be passed through, not
        # turned into a picture.
        if isinstance(value, Image.Image):
          value = _convert_image(value, self.image_format)
          if self.transform is not None:
            value = self.transform(value)
        out[name] = value
      yield out
