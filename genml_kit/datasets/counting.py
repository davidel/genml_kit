"""Exact sample-count estimation for streaming (iterable) datasets.

A streaming source has no index space, so ``len(dataset)`` cannot answer how
many samples it holds.  :func:`genml_kit.datasets.base.num_samples` therefore
falls back to a *declared* estimate (progress output only, never load-bearing).
This module recovers an **exact** count from cheap metadata -- without reading
the payloads -- for the two families the toolkit streams from:

* :func:`hf_num_samples` -- the HuggingFace split size declared on the dataset
  card (``load_dataset_builder(...).info.splits[split].num_examples``), or the
  row count in each split's ``parquet``/``arrow`` footer.  Both read only
  metadata, and both go through the shared HuggingFace cache.
* :func:`webdataset_num_samples` -- the ``num_samples`` / ``shard_lengths`` a
  Hub webdataset repo ships in its ``_info.json`` (or a ``wids`` descriptor's
  per-shard ``nsamples``).

The result feeds the informational ``num_samples=`` argument of
:class:`~genml_kit.datasets.hf_proxy.HFDatasetProxy` /
:class:`~genml_kit.datasets.webdataset_source.WebDatasetSource`.  When no exact
source exists the functions return ``None`` -- the honest "unknown" -- rather
than a byte-ratio guess: extrapolating a count by sampling the stream is
deliberately out of scope here.

Every remote read honours *cache_dir* so it is populated once and reused by the
later training scan.
"""

import glob
import json
import logging
import os
import re

import datasets

#: Extensions whose row count can be read straight from a file footer.
_FOOTER_EXTS = (".parquet", ".arrow")

#: JSON descriptors a webdataset shard repo may ship with sample counts.
_WDS_DESCRIPTORS = ("_info.json", "dataset.json", "wids.json")


def hf_num_samples(dataset, name=None, split="train", cache_dir=None, **load_kwargs):
  """Return the exact sample count for a HuggingFace split, or ``None``.

  Resolution order, cheapest first:

  1. the split size declared on the dataset card (a dataset *builder* is
     materialised -- no Arrow data is built or downloaded);
  2. the summed row count of the split's ``parquet``/``arrow`` files.

  Step 2 reuses the file list ``datasets`` itself resolved for the split
  (``builder.config.data_files``), so ``train`` cannot collide with
  ``validation`` and ``hf://`` URLs carry their pinned revision.  Only when no
  builder is available does it fall back to a local ``*.parquet``/``*.arrow``
  glob filtered by the split name.

  Args:
    dataset: A HuggingFace repo id (``"user/name"``), or a local path to a
      ``save_to_disk`` directory or a directory of parquet/arrow files.
    name: Optional dataset configuration/subset name.
    split: Split to count (default ``"train"``).
    cache_dir: HuggingFace cache directory.  Forwarded to
      ``load_dataset_builder`` and any ``hf_hub_download`` fallback, so the
      download is cached and a later ``load_dataset`` reuses it.
    **load_kwargs: Extra keyword arguments forwarded to
      ``datasets.load_dataset_builder`` (e.g. ``revision=``, ``token=``).

  Returns:
    int | None: The exact row count, or ``None`` when it cannot be determined
    exactly (unknown split, gated repo, non-footer format, ...).
  """
  builder = _load_builder(dataset, name, cache_dir, load_kwargs)

  if builder is not None:
    declared = _declared_split_size(builder, split)
    if declared is not None:
      return declared

  files = _resolved_split_files(builder, split)
  if not files:
    files = _local_split_files(dataset, split)

  total = _sum_file_rows(files, cache_dir)
  # ``_sum_file_rows`` returns None when any file is uncountable, so an
  # incomplete sum is never reported as an exact count.
  return total


def webdataset_num_samples(shards=None, repo_id=None, split=None, cache_dir=None):
  """Return the exact sample count for a webdataset shard list, or ``None``.

  A webdataset has no ``num_examples`` channel of its own, but a Hub shard repo
  usually ships one of two JSON descriptors that do:

  * ``_info.json`` -- ``{"splits": {split: {"num_samples" | "shard_lengths"}}}``
    (the convention ``datasets`` writes for ``webdataset`` repos);
  * a ``wids`` descriptor -- ``{"shardlist": [{"nsamples": N}, ...]}``.

  Neither the ``.idx`` offset files nor the ``_info.json`` ``filenames`` list is
  used: ``.idx`` indexes *files* (not the multi-file samples that webdataset
  groups by key), and a filename list counts shards, not samples.

  Args:
    shards: A local shard path/pattern (``shards-{0000..0009}.tar``) or a list
      of them.  The directory of the first is searched for a descriptor.
    repo_id: A HuggingFace webdataset repo id.  When given, its ``_info.json``
      (then ``dataset.json`` / ``wids.json``) is fetched through the cache.
    split: Split to count.  Required only when a descriptor declares more than
      one split.
    cache_dir: HuggingFace cache directory, forwarded to ``hf_hub_download``.

  Returns:
    int | None: The exact sample count, or ``None`` when no descriptor is
    found or it carries no usable count.
  """
  if repo_id is not None:
    return _remote_webdataset_total(repo_id, split, cache_dir)

  for path in _descriptor_candidates(shards):
    total = _webdataset_total_from_file(path, split)
    if total is not None:
      return total
  return None


def _load_builder(dataset, name, cache_dir, load_kwargs):
  """Materialise a ``DatasetBuilder``, or return ``None`` on any failure."""
  try:
    return datasets.load_dataset_builder(dataset,
                                         name=name,
                                         cache_dir=cache_dir,
                                         **load_kwargs)
  except Exception as e:
    # A missing/gated/script-only dataset still deserves a cheap Tier-2 attempt
    # from the local/remote file footers, so this is logged, not raised.
    logging.debug("load_dataset_builder(%r) failed: %s", dataset, e)
    return None


def _declared_split_size(builder, split):
  """Return the split's declared ``num_examples``, or ``None`` when absent."""
  splits = getattr(builder.info, "splits", None)
  if not splits or split not in splits:
    return None
  num_examples = splits[split].num_examples
  if num_examples:
    return int(num_examples)
  return None


def _resolved_split_files(builder, split):
  """Return the file paths ``datasets`` resolved for *split*, or ``[]``.

  ``builder.config.data_files`` maps a split to its files (local paths or
  ``hf://`` URLs, revision-pinned).  Keys may be ``NamedSplit`` or ``str``.
  """
  if builder is None:
    return []
  data_files = getattr(getattr(builder, "config", None), "data_files", None)
  if not data_files:
    return []
  for key, value in data_files.items():
    if str(key) == split:
      return [str(path) for path in value]
  return []


def _local_split_files(dataset, split):
  """Glob a local directory for *split*'s footer files, or return ``[]``.

  Only used when no builder resolved the file list.  A file counts as part of
  *split* when the split name appears as a whole token in its name (so ``val``
  does not match ``validation``); when nothing matches, nothing is returned
  rather than silently summing every split.
  """
  path = os.path.expanduser(str(dataset))
  if os.path.isfile(path):
    return [path]
  if not os.path.isdir(path):
    return []
  found = []
  for ext in _FOOTER_EXTS:
    for hit in sorted(glob.glob(os.path.join(path, "**", "*" + ext), recursive=True)):
      if _matches_split(hit, split):
        found.append(hit)
  return found


def _matches_split(path, split):
  """Return True when *split* is a whole token in *path*'s file name."""
  stem = os.path.basename(path)
  pattern = r"(^|[^A-Za-z])" + re.escape(split) + r"([^A-Za-z]|$)"
  return re.search(pattern, stem) is not None


def _sum_file_rows(files, cache_dir):
  """Sum the row counts of *files*, or return ``None`` if any is uncountable."""
  if not files:
    return None
  total = 0
  for path in files:
    rows = _file_rows(path, cache_dir)
    if rows is None:
      return None
    total += rows
  return total


def _file_rows(path, cache_dir):
  """Return the row count of one ``parquet``/``arrow`` file, or ``None``."""
  if path.startswith("hf://"):
    return _remote_file_rows(path, cache_dir)
  if os.path.isfile(path):
    return _local_file_rows(path)
  return None


def _local_file_rows(path):
  """Read a local footer file's row count, or return ``None``."""
  try:
    with open(path, "rb") as handle:
      return _footer_rows(handle, path)
  except Exception as e:
    logging.debug("cannot read footer of %r: %s", path, e)
    return None


def _remote_file_rows(url, cache_dir):
  """Read a Hub file's row count: footer range-read, then cached download.

  The range read pulls only the footer over HTTP (through HuggingFace's
  client); when it is unavailable the file is downloaded with
  ``hf_hub_download`` so it lands in *cache_dir* for the later scan to reuse.
  """
  ext = os.path.splitext(url)[1].lower()
  if ext not in _FOOTER_EXTS:
    return None

  try:
    from huggingface_hub import HfFileSystem
  except ImportError:
    HfFileSystem = None
  if HfFileSystem is not None:
    try:
      with HfFileSystem().open(url, "rb") as handle:
        return _footer_rows(handle, url)
    except Exception as e:
      logging.debug("footer range-read of %r failed: %s", url, e)

  return _downloaded_file_rows(url, cache_dir)


def _downloaded_file_rows(url, cache_dir):
  """Download a Hub footered file into the cache, then read its rows."""
  parsed = _parse_hf_url(url)
  if parsed is None:
    return None
  repo_id, filename = parsed
  try:
    from huggingface_hub import hf_hub_download
  except ImportError:
    return None
  try:
    local = hf_hub_download(repo_id=repo_id,
                            filename=filename,
                            repo_type="dataset",
                            cache_dir=cache_dir)
  except Exception as e:
    logging.debug("hf_hub_download(%r, %r) failed: %s", repo_id, filename, e)
    return None
  return _local_file_rows(local)


def _footer_rows(handle, path):
  """Count rows in an open ``parquet`` or ``arrow`` file object, or ``None``."""
  ext = os.path.splitext(path)[1].lower()
  if ext == ".parquet":
    return _parquet_rows(handle)
  if ext == ".arrow":
    return _arrow_rows(handle)
  return None


def _parquet_rows(handle):
  """Row count from a parquet footer."""
  import pyarrow.parquet as pq

  return int(pq.ParquetFile(handle).metadata.num_rows)


def _arrow_rows(handle):
  """Row count from an Arrow IPC file *or* stream.

  ``datasets.save_to_disk`` writes the IPC **stream** layout, while other
  producers write the **file** layout; the two need different readers.  Each
  reader is retried from the start because the failed attempt may have consumed
  the header bytes.
  """
  import pyarrow as pa

  try:
    reader = pa.ipc.open_file(handle)
    return int(
        sum(reader.get_batch(i).num_rows for i in range(reader.num_record_batches)))
  except Exception:
    handle.seek(0)
    reader = pa.ipc.open_stream(handle)
    return int(sum(batch.num_rows for batch in reader))


def _parse_hf_url(url):
  """Split ``hf://datasets/<repo>@<rev>/<path>`` into ``(repo_id, path)``."""
  rest = url[len("hf://"):]
  repo_part, sep, rev_file = rest.partition("@")
  segments = repo_part.split("/")
  if len(segments) < 3 or segments[0] != "datasets":
    return None
  repo_id = "/".join(segments[1:3])
  if sep:
    _, _, filename = rev_file.partition("/")
  else:
    filename = "/".join(segments[3:])
  if not filename:
    return None
  return repo_id, filename


def _descriptor_candidates(shards):
  """Yield local JSON descriptor paths to probe for *shards*."""
  candidates = []
  for directory in _shard_dirs(shards):
    for descriptor in _WDS_DESCRIPTORS:
      path = os.path.join(directory, descriptor)
      if os.path.isfile(path):
        candidates.append(path)
  return candidates


def _shard_dirs(shards):
  """Return the distinct local directories that hold *shards*."""
  if shards is None:
    return []
  entries = shards if isinstance(shards, (list, tuple)) else [shards]
  dirs = []
  for entry in entries:
    if not isinstance(entry, str):
      continue
    directory = os.path.dirname(entry) or "."
    if directory not in dirs:
      dirs.append(directory)
  return dirs


def _remote_webdataset_total(repo_id, split, cache_dir):
  """Fetch a Hub webdataset descriptor and count its samples, or ``None``."""
  try:
    from huggingface_hub import hf_hub_download
  except ImportError:
    return None
  for descriptor in _WDS_DESCRIPTORS:
    try:
      local = hf_hub_download(repo_id=repo_id,
                              filename=descriptor,
                              repo_type="dataset",
                              cache_dir=cache_dir)
    except Exception:
      continue
    total = _webdataset_total_from_file(local, split)
    if total is not None:
      return total
  return None


def _webdataset_total_from_file(path, split):
  """Parse a webdataset descriptor *path* and count samples, or ``None``."""
  try:
    with open(path) as handle:
      data = json.load(handle)
  except Exception as e:
    logging.debug("cannot parse webdataset descriptor %r: %s", path, e)
    return None
  return _webdataset_total(data, split)


def _webdataset_total(data, split):
  """Extract a sample total from a parsed webdataset descriptor, or ``None``."""
  if not isinstance(data, dict):
    return None
  if "splits" in data:
    return _webdataset_splits_total(data["splits"], split)
  if "shardlist" in data:
    return _shardlist_total(data["shardlist"])
  return None


def _webdataset_splits_total(splits, split):
  """Sum a ``datasets``-style ``splits`` descriptor for *split*, or ``None``."""
  if not isinstance(splits, dict) or not splits:
    return None
  if split is not None:
    info = splits.get(split)
  elif len(splits) == 1:
    info = next(iter(splits.values()))
  else:
    logging.debug("webdataset descriptor has %d splits; pass split= to pick "
                  "one.", len(splits))
    return None
  if not isinstance(info, dict):
    return None
  declared = info.get("num_samples")
  if isinstance(declared, int) and declared > 0:
    return declared
  lengths = info.get("shard_lengths")
  if (isinstance(lengths, list) and lengths and
      all(isinstance(length, int) for length in lengths)):
    return int(sum(lengths))
  return None


def _shardlist_total(shardlist):
  """Sum a ``wids`` ``shardlist``'s ``nsamples`` field, or ``None``."""
  if not isinstance(shardlist, list) or not shardlist:
    return None
  total = 0
  for entry in shardlist:
    if not isinstance(entry, dict):
      return None
    count = entry.get("nsamples")
    if not isinstance(count, int):
      return None
    total += count
  return total
