"""Sample-count estimation for streaming (iterable) datasets.

A streaming source has no index space, so ``len(dataset)`` cannot answer how
many samples it holds.  :func:`genml_kit.datasets.base.num_samples` therefore
falls back to a *declared* estimate (progress output only, never load-bearing).
This module recovers a count from cheap metadata -- without reading the payloads
-- for the two families the toolkit streams from:

* :func:`hf_num_samples` -- the split size declared on the dataset card
  (``load_dataset_builder(...).info.splits[split].num_examples``), the size the
  HuggingFace datasets server computed (``/size``), or the row count in each
  split's ``parquet``/``arrow`` footer.  All read metadata only, and all go
  through the shared HuggingFace cache.
* :func:`webdataset_num_samples` -- the ``num_samples`` / ``shard_lengths`` a
  Hub webdataset repo ships in its ``_info.json`` (or a ``wids`` descriptor's
  per-shard ``nsamples``), a DALI ``.idx`` index, or -- as the last resort -- a
  tar header walk that counts samples without decoding them.

The result feeds the informational ``num_samples=`` argument of
:class:`~genml_kit.datasets.hf_proxy.HFDatasetProxy` /
:class:`~genml_kit.datasets.webdataset_source.WebDatasetSource`.  Two kinds of
number can come back:

* **exact** -- a declared size, a footer row count, a tar walk, an ``.idx``
  header, or a datasets-server count that is *not* a partial scan;
* **estimated** -- the datasets server's ``estimated_num_rows`` for a dataset it
  only scanned partially.  This is used only as a labelled fallback (an INFO log
  says so) and never replaces an exact source.

Anything the module cannot establish confidently is ``None`` -- the honest
"unknown" -- rather than a locally computed byte-ratio guess.

Every remote read honours *cache_dir* so it is populated once and reused by the
later training scan.
"""

import glob
import json
import logging
import os
import re
import urllib.parse
import urllib.request

import datasets

#: Extensions whose row count can be read straight from a file footer.
_FOOTER_EXTS = (".parquet", ".arrow")

#: JSON descriptors a webdataset shard repo may ship with sample counts.
_WDS_DESCRIPTORS = ("_info.json", "dataset.json", "wids.json")

#: Base URL of the HuggingFace datasets server (override for tests/endpoints).
_DATASETS_SERVER_URL = os.environ.get("HF_DATASETS_SERVER_URL",
                                      "https://datasets-server.huggingface.co")

#: tar stores metadata in 512-byte blocks; file data is padded to a block.
_TAR_BLOCK = 512

#: webdataset's own bookkeeping members (``__key__``, ``__url__``, ...).
_TAR_META_RE = re.compile(r"__[^/]*__($|/)")

#: Split ``name.ext`` into (key, ext) exactly as webdataset's ``base_plus_ext``.
_TAR_BASE_EXT_RE = re.compile(r"^((?:.*/|)[^.]+)[.]([^/]*)$")

#: Sentinel marking "no sample seen yet" in the tar walk.
_NO_PREFIX = object()


def hf_num_samples(dataset,
                   name=None,
                   split="train",
                   cache_dir=None,
                   token=None,
                   **load_kwargs):
  """Return the sample count for a HuggingFace split, or ``None``.

  Resolution order, cheapest first:

  1. the split size declared on the dataset card (a dataset *builder* is
     materialised -- no Arrow data is built or downloaded);
  2. the HuggingFace datasets server's ``/size`` for the repo -- exact when the
     server reports a full scan (``partial`` false), else its
     ``estimated_num_rows`` as a *labelled* fallback estimate;
  3. the summed row count of the split's ``parquet``/``arrow`` files.

  Step 3 reuses the file list ``datasets`` itself resolved for the split
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
    token: Optional HuggingFace token for the datasets-server call (needed for
      gated/private repos).
    **load_kwargs: Extra keyword arguments forwarded to
      ``datasets.load_dataset_builder`` (e.g. ``revision=``).

  Returns:
    int | None: The sample count, or ``None`` when it cannot be established.
    A datasets-server *estimate* is returned only as a fallback and is logged.
  """
  builder = _load_builder(dataset, name, cache_dir, token, load_kwargs)

  if builder is not None:
    declared = _declared_split_size(builder, split)
    if declared is not None:
      return declared

  if _looks_like_repo_id(dataset):
    served = _server_size(dataset, name, split, token)
    if served is not None:
      return served

  files = _resolved_split_files(builder, split)
  if not files:
    files = _local_split_files(dataset, split)

  # ``_sum_file_rows`` returns None when any file is uncountable, so an
  # incomplete sum is never reported as a count.
  return _sum_file_rows(files, cache_dir)


def webdataset_num_samples(shards=None,
                           repo_id=None,
                           split=None,
                           cache_dir=None,
                           token=None):
  """Return the sample count for a webdataset shard list, or ``None``.

  A webdataset shard list has no schema of its own, so the count is recovered
  from, in order of cost:

  1. a JSON descriptor next to the shards (or at the repo root) --
     ``_info.json`` / ``dataset.json`` / ``wids.json`` -- carrying
     ``num_samples`` / ``shard_lengths`` / per-shard ``nsamples``;
  2. a per-shard DALI ``.idx`` index whose header records the sample count;
  3. the datasets server's ``/size`` for a repo, accepted only when it is a
     full scan (a webdataset repo almost never is, so this is usually a no-op);
  4. a tar **header walk**: read the 512-byte headers, skip the data blocks, and
     count samples the way webdataset groups them (a new sample whenever the
     key prefix changes from the previous member).  Exact, and decoding-free.

  A ``.idx`` in webdataset's own ``wids`` format is *not* read: it is a pickle
  of per-*file* offsets, so it neither yields a sample count nor can be loaded
  safely.  Only the DALI text index is used.

  Args:
    shards: A local shard path/pattern (``shards-{0000..0009}.tar``) or a list
      of them.  Descriptors are looked for beside the first shard.
    repo_id: A HuggingFace webdataset repo id.  When given, descriptors are
      fetched through the cache.
    split: Split to count.  Needed only when a descriptor declares several.
    cache_dir: HuggingFace cache directory, forwarded to ``hf_hub_download``.
    token: Optional HuggingFace token for the datasets-server call.

  Returns:
    int | None: The sample count, or ``None`` when it cannot be established
    (no descriptor, unreadable index, and a tar that cannot be walked).
  """
  if repo_id is not None:
    return _remote_webdataset_total(repo_id, split, cache_dir, token)

  for path in _descriptor_candidates(shards):
    total = _webdataset_total_from_file(path, split)
    if total is not None:
      return total
  return _local_tar_total(shards)


def _load_builder(dataset, name, cache_dir, token, load_kwargs):
  """Materialise a ``DatasetBuilder``, or return ``None`` on any failure."""
  kwargs = dict(load_kwargs)
  if token is not None and "token" not in kwargs:
    kwargs["token"] = token
  try:
    return datasets.load_dataset_builder(dataset,
                                         name=name,
                                         cache_dir=cache_dir,
                                         **kwargs)
  except Exception as e:
    # A missing/gated/script-only dataset still deserves a cheap footer attempt
    # from the local/remote files, so this is logged, not raised.
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


def _looks_like_repo_id(dataset):
  """Return True when *dataset* is a Hub repo id rather than a local path."""
  return isinstance(dataset, str) and "/" in dataset and not os.path.exists(
      os.path.expanduser(dataset))


def _server_size(dataset, config, split, token, exact_only=False):
  """Ask the datasets server for *split*'s size, or ``None``.

  The server reports ``{"size": {...}, "partial": bool}``; ``num_rows`` is only
  trustworthy when ``partial`` is false (a partially-scanned webdataset repo
  reports the wrong number).  When ``exact_only`` is false a partial scan's
  ``estimated_num_rows`` is returned as a *labelled* fallback.
  """
  payload = _http_get_json(_datasets_server_url(dataset, config, split), token)
  if not isinstance(payload, dict):
    return None
  body = payload.get("size")
  if not isinstance(body, dict):
    return None
  entry = _server_split_entry(body.get("splits"), config, split)
  if entry is None:
    return None

  rows = entry.get("num_rows")
  if payload.get("partial") is False and isinstance(rows, int) and rows > 0:
    return rows

  if exact_only:
    return None
  estimate = entry.get("estimated_num_rows")
  if isinstance(estimate, int) and estimate > 0:
    logging.info(
        "Using HuggingFace's *estimated* row count for %s (split %s): %d "
        "(the datasets server only scanned it partially).", dataset, split, estimate)
    return estimate
  return None


def _server_split_entry(splits, config, split):
  """Return the ``splits`` entry matching *config*/*split*, or ``None``."""
  if not isinstance(splits, list):
    return None
  for entry in splits:
    if not isinstance(entry, dict) or entry.get("split") != split:
      continue
    if config is not None and entry.get("config") != config:
      continue
    return entry
  return None


def _datasets_server_url(dataset, config, split):
  """Build the ``/size`` URL for *dataset*/*config*/*split*."""
  query = {"dataset": dataset}
  if config is not None:
    query["config"] = config
  if split is not None:
    query["split"] = split
  return f"{_DATASETS_SERVER_URL}/size?{urllib.parse.urlencode(query)}"


def _http_get_json(url, token):
  """GET *url* and parse JSON, or return ``None`` on any failure."""
  request = urllib.request.Request(url)
  request.add_header("Accept", "application/json")
  if token:
    request.add_header("Authorization", f"Bearer {token}")
  try:
    with urllib.request.urlopen(request, timeout=30) as response:
      return json.loads(response.read().decode("utf-8"))
  except Exception as e:
    logging.debug("datasets-server GET %r failed: %s", url, e)
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


def _remote_webdataset_total(repo_id, split, cache_dir, token):
  """Fetch a Hub webdataset count, or ``None``.

  Descriptors first; then an *exact* datasets-server count (a webdataset repo is
  almost always a partial scan, so this normally declines); then a tar header
  walk over the repo's shards.
  """
  for descriptor in _WDS_DESCRIPTORS:
    local = _fetch_repo_file(repo_id, descriptor, cache_dir)
    if local is None:
      continue
    total = _webdataset_total_from_file(local, split)
    if total is not None:
      return total

  served = _server_size(repo_id, None, split, token, exact_only=True)
  if served is not None:
    return served

  return _remote_tar_total(repo_id, cache_dir)


def _remote_tar_total(repo_id, cache_dir):
  """Walk every ``.tar`` shard of *repo_id*, or ``None`` if any is uncountable."""
  names = _repo_files(repo_id)
  shards = sorted(name for name in names if name.endswith(".tar"))
  if not shards:
    return None
  # Probe an ``.idx`` only when the repo actually ships one, so a shard list
  # without indexes does not cost a failing request per shard.
  idx_names = {name for name in names if name.endswith(".idx")}
  total = 0
  for name in shards:
    count = _remote_shard_count(repo_id, name, name + ".idx" in idx_names, cache_dir)
    if count is None:
      return None
    total += count
  return total


def _remote_shard_count(repo_id, name, has_idx, cache_dir):
  """Count one remote shard: its DALI ``.idx`` if present, else a header walk."""
  if has_idx:
    from_idx = _dali_idx_count(_fetch_repo_file(repo_id, name + ".idx", cache_dir))
    if from_idx is not None:
      return from_idx
  try:
    from huggingface_hub import HfFileSystem
  except ImportError:
    return None
  url = f"hf://datasets/{repo_id}/{name}"
  try:
    with HfFileSystem().open(url, "rb") as handle:
      return _count_tar_stream(handle)
  except Exception as e:
    logging.debug("tar walk of %r failed: %s", url, e)
    return None


def _repo_files(repo_id):
  """List the file names of a Hub dataset repo, or ``[]``."""
  try:
    from huggingface_hub import HfApi
  except ImportError:
    return []
  try:
    return HfApi().list_repo_files(repo_id, repo_type="dataset")
  except Exception as e:
    logging.debug("list_repo_files(%r) failed: %s", repo_id, e)
    return []


def _fetch_repo_file(repo_id, filename, cache_dir):
  """Return a cached local path for a repo file, or ``None`` when absent."""
  try:
    from huggingface_hub import hf_hub_download
  except ImportError:
    return None
  try:
    return hf_hub_download(repo_id=repo_id,
                           filename=filename,
                           repo_type="dataset",
                           cache_dir=cache_dir)
  except Exception:
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


def _local_tar_total(shards):
  """Count local shards by their ``.idx`` or a header walk, or ``None``."""
  paths = _expand_shards(shards)
  if not paths:
    return None
  total = 0
  for path in paths:
    if _is_remote(path):
      return None
    from_idx = _dali_idx_count(path + ".idx")
    count = from_idx if from_idx is not None else _local_tar_count(path)
    if count is None:
      return None
    total += count
  return total


def _expand_shards(shards):
  """Expand a shard brace pattern/list into concrete paths, or ``[]``.

  Uses webdataset's own expansion (``::``-separated patterns, env vars, brace
  ranges), falling back to the raw value when webdataset is not installed.
  """
  if shards is None:
    return []
  try:
    from webdataset.shardlists import expand_urls
  except ImportError:
    return list(shards) if isinstance(shards, (list, tuple)) else [shards]
  try:
    return list(expand_urls(shards))
  except Exception as e:
    logging.debug("shard expansion of %r failed: %s", shards, e)
    return []


def _is_remote(path):
  """Return True when *path* is a URL rather than a local file."""
  scheme = urllib.parse.urlparse(str(path)).scheme
  return scheme in ("http", "https", "s3", "gs", "hf", "ftp")


def _local_tar_count(path):
  """Header-walk one local tar, or return ``None``."""
  try:
    with open(path, "rb") as handle:
      return _count_tar_stream(handle)
  except Exception as e:
    logging.debug("cannot walk tar %r: %s", path, e)
    return None


def _count_tar_stream(handle):
  """Count samples in an open tar stream by walking its 512-byte headers.

  A sample is one run of members sharing a key prefix, matching webdataset's
  ``group_by_keys`` (a new sample begins whenever the prefix differs from the
  *previous* member's).  Meta members and extension-less names are skipped.
  Data blocks are seeked over, never read.  Returns ``None`` unless the archive
  ends with a proper end-of-archive marker, so a truncated tar is never reported
  as a (smaller) count.
  """
  try:
    handle.seek(0)
  except Exception:
    return None
  count = 0
  prev = _NO_PREFIX
  position = 0
  while True:
    handle.seek(position)
    header = handle.read(_TAR_BLOCK)
    if len(header) < _TAR_BLOCK:
      # Ran off the end without the two zero blocks a valid tar ends with: the
      # file is truncated, so any count would be wrong.
      return None
    if header == b"\0" * _TAR_BLOCK:
      # End-of-archive marker: a second zero block must follow for a valid tar.
      tail = handle.read(_TAR_BLOCK)
      return count if tail == b"\0" * _TAR_BLOCK else None
    size = _tar_member_size(header)
    if size is None:
      return None
    prefix = _tar_sample_prefix(_tar_member_name(header))
    if prefix is not None and prefix != prev:
      count += 1
      prev = prefix
    position += _TAR_BLOCK + _tar_data_blocks(size)


def _tar_member_name(header):
  """Decode a tar header's ``name`` field (offset 0, 100 bytes)."""
  return header[:100].split(b"\0", 1)[0].decode("utf-8", "replace")


def _tar_member_size(header):
  """Decode a tar header's octal ``size`` field, or ``None`` when malformed."""
  field = header[124:136].split(b"\0", 1)[0].strip()
  if not field:
    # An empty size field is a valid (zero-length) member header.
    return 0
  try:
    return int(field, 8)
  except ValueError:
    return None


def _tar_data_blocks(size):
  """Round a member's byte size up to whole 512-byte tar blocks."""
  return (size + _TAR_BLOCK - 1) // _TAR_BLOCK * _TAR_BLOCK


def _tar_sample_prefix(name):
  """Return a member's sample key (``name`` without extension), or ``None``.

  Mirrors webdataset: ``__meta__``-style members and names without a ``.ext``
  are not part of any sample.
  """
  if not name or _TAR_META_RE.search(name):
    return None
  match = _TAR_BASE_EXT_RE.match(name)
  if not match:
    return None
  return match.group(1)


def _dali_idx_count(path):
  """Return the sample count from a DALI ``.idx`` header, or ``None``.

  The DALI index starts with ``#v1.2 <num_samples>``.  A webdataset ``wids``
  ``.index`` is a *pickle* of per-file offsets, so it is deliberately not read
  (neither a sample count nor safe to load).
  """
  if path is None:
    return None
  try:
    with open(path, errors="replace") as handle:
      first = handle.readline().strip()
  except Exception as e:
    logging.debug("cannot read index %r: %s", path, e)
    return None
  if not first.startswith("#v"):
    return None
  parts = first.split()
  if len(parts) < 2:
    return None
  try:
    count = int(parts[1])
  except ValueError:
    return None
  return count if count >= 0 else None
