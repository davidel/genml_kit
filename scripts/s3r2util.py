#!/usr/bin/env python3
"""CloudFlare R2 CLI utility (S3-compatible).

Provides ls/cp/rm/mv/cat and other commands for R2 buckets via the
boto3 S3 API.

Output conventions:
  - stdout carries data only: object bytes for `cat`, the URL for
    `presign`.
  - stderr carries everything else (status/progress/log lines, tqdm
    bars, errors), so piped data is never polluted.

NOTE: This script could in theory be simplified by setting the standard
AWS environment variables (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
AWS_REGION) from the CloudFlare R2 ones and forwarding directly to the
`aws s3` CLI (e.g. `aws s3 ls --endpoint-url <r2-endpoint>`). R2
implements the S3 API, so the AWS CLI works as a drop-in client. That
approach would remove the boto3 dependency and most of this script.
However, keeping our own wrapper lets us tailor the interface (e.g.
trailing-slash destination semantics, r2:// path prefixes) without
depending on the `aws` CLI being installed.
"""
import argparse
import datetime as dt
import fnmatch
import os
import re
import sys
import time

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from tqdm import tqdm


def fatal(msg, code=1):
  print(f"Error: {msg}", file=sys.stderr)
  sys.exit(code)


def getenv(name, vtype=str, default=None):
  """Fetch an environment variable and convert it to `vtype`.

  Blank or unset variables yield `default`. Fatal with a stderr
  message when the raw value cannot be converted, so misconfiguration
  fails fast instead of mid-run.
  """
  raw = (os.getenv(name) or "").strip()
  if not raw:
    return default
  try:
    return vtype(raw)
  except ValueError:
    fatal(f"Environment variable {name} must be of type "
          f"{vtype.__name__}, got: {raw!r}")


# Chunk size for streaming object bytes through `cat`. Keeps memory
# flat for multi-GB objects; never read a whole body into RAM.
DEFAULT_STREAM_CHUNK_SIZE = 1024 * 1024
STREAM_CHUNK_SIZE = getenv("R2_STREAM_CHUNK_SIZE",
                           vtype=int,
                           default=DEFAULT_STREAM_CHUNK_SIZE)


def log(msg, quiet=False):
  """Print a status message to stderr unless --quiet is active.

  Always targets stderr (never stdout) so that stdout stays reserved
  for data output (`cat`, `presign`).
  """
  if not quiet:
    print(msg, file=sys.stderr)


def create_s3_client():
  account_id = (os.getenv("CLOUDFLARE_ACCOUNT_ID") or "").strip()
  access_key_id = (os.getenv("R2_ACCESS_KEY_ID") or "").strip()
  secret_access_key = (os.getenv("R2_SECRET_ACCESS_KEY") or "").strip()

  if not all([account_id, access_key_id, secret_access_key]):
    fatal("Missing credentials. Please set CLOUDFLARE_ACCOUNT_ID, "
          "R2_ACCESS_KEY_ID, and R2_SECRET_ACCESS_KEY in your environment.")

  return boto3.client(
      service_name="s3",
      endpoint_url=f"https://{account_id}.r2.cloudflarestorage.com",
      aws_access_key_id=access_key_id,
      aws_secret_access_key=secret_access_key,
      region_name="auto",
  )


def parse_r2_path(path):
  """Parses 'r2://bucket-name/object/key' strings.

  Returns:
    A tuple of (bucket_name, object_key), or (None, None) if not an R2 path.
  """
  if path.startswith("r2://"):
    cleaned = path[5:]
    if "/" in cleaned:
      bucket, key = cleaned.split("/", 1)
      return bucket, key
    return cleaned, ""
  return None, None


def has_wildcard(path):
  """Check if path contains shell wildcard characters (* or ?)."""
  return "*" in path or "?" in path


def get_static_prefix(pattern):
  """Return the longest prefix of *pattern* that contains no wildcard chars."""
  m = re.search(r"[*?]", pattern)
  return pattern[:m.start()] if m else pattern


def expand_wildcard(s3_client, r2_path):
  """Expand a wildcard R2 path via server-side prefix listing + client-side fnmatch.

  Returns a list of (bucket, key, size) tuples for all matching objects.
  Size comes from the list_objects_v2 response (no extra HEAD requests).
  """
  bucket, pattern = parse_r2_path(r2_path)
  if not bucket:
    fatal(f"Wildcard path must start with r2://: {r2_path}")

  prefix = get_static_prefix(pattern)

  # Try the prefix as-is first, then fall back to toggling leading '/'.
  # Some buckets store keys with a leading '/' (e.g. '/content/file.pt')
  # while parse_r2_path strips it (giving 'content/file.pt').
  prefixes_to_try = [prefix]
  if prefix.startswith("/"):
    prefixes_to_try.append(prefix[1:])
  elif prefix:
    prefixes_to_try.append("/" + prefix)

  matches = []
  try:
    paginator = s3_client.get_paginator("list_objects_v2")
    for try_prefix in prefixes_to_try:
      page_iter = paginator.paginate(Bucket=bucket, Prefix=try_prefix)
      for page in page_iter:
        for obj in page.get("Contents", []):
          key = obj["Key"].lstrip("/")
          if fnmatch.fnmatch(key, pattern):
            matches.append((bucket, key, obj["Size"]))
      if matches:
        break
  except (ClientError, BotoCoreError) as exc:
    fatal(f"Error listing objects: {exc}")

  if not matches:
    fatal(f"No objects matched pattern: {r2_path}")
  return matches


def compute_dest_key(src_key, src_prefix, dst_prefix):
  """Map a matched source key to a destination key.

  Args:
    src_key: the full object key of the matched source object.
    src_prefix: the static (non-wildcard) prefix of the source pattern.
    dst_prefix: the destination prefix (bucket/key/path).
  """
  relative = src_key[len(src_prefix):].lstrip("/")
  if not relative:
    return dst_prefix
  if not dst_prefix:
    # Bucket-root destination: never emit a leading '/' in the key.
    return relative
  # Ensure exactly one '/' between prefix and relative.
  return dst_prefix.rstrip("/") + "/" + relative


def format_size(num_bytes):
  """Format byte count as a human-readable string (KB, MB, GB, TB)."""
  for unit in ("B", "KB", "MB", "GB", "TB"):
    if abs(num_bytes) < 1024:
      return f"{num_bytes:.2f} {unit}"
    num_bytes /= 1024
  return f"{num_bytes:.2f} PB"


def format_date(d):
  """Format a datetime to local time string (YYYY-MM-DD HH:MM:SS tz)."""
  return d.astimezone().strftime("%Y-%m-%d %H:%M:%S %z")


def _cat_source(s3_client, source):
  """Return an iterator of byte chunks for one `cat` source.

  The caller owns draining (and thereby closing) the iterator.
  Raises FileNotFoundError for missing local files and ClientError /
  BotoCoreError for bad remote objects so the caller can report them.
  """
  bucket, key = parse_r2_path(source)
  if bucket is None:
    path = os.path.expanduser(source)
    if not os.path.isfile(path):
      raise FileNotFoundError(f"No such file: {source}")
    return _stream_local_file(path)
  if not key:
    fatal("'cat' requires an object key: r2://bucket/key "
          "(cannot cat a whole bucket or prefix)")
  if has_wildcard(key):
    return _stream_remote_matches(s3_client, expand_wildcard(s3_client, source))
  response = s3_client.get_object(Bucket=bucket, Key=key)
  return response["Body"].iter_chunks(STREAM_CHUNK_SIZE)


def _stream_remote_matches(s3_client, matches):
  """Yield byte chunks from each matched object, in listing order."""
  for bucket, key, _size in matches:
    response = s3_client.get_object(Bucket=bucket, Key=key)
    yield from response["Body"].iter_chunks(STREAM_CHUNK_SIZE)


def _read_chunks(fh):
  """Read a binary file object in STREAM_CHUNK_SIZE pieces."""
  return iter(lambda: fh.read(STREAM_CHUNK_SIZE), b"")


def _stream_local_file(path):
  """Yield byte chunks from a local file, closing it when consumed."""
  with open(path, "rb") as fh:
    yield from _read_chunks(fh)


def handle_cat(args, s3_client):
  """Stream R2 objects and/or local files to stdout.

  Always quiet: no status lines, no progress bar (`--quiet` is accepted
  by the global parser but irrelevant here). Streams in fixed-size
  chunks so multi-GB objects never load fully into memory, making it
  suitable for piping into e.g. `tar -xf -` without staging the file.
  """
  sources = args.sources
  out = sys.stdout.buffer
  for source in sources:
    try:
      chunks = _cat_source(s3_client, source)
      for chunk in chunks:
        out.write(chunk)
    except (ClientError, BotoCoreError) as e:
      fatal(f"cat failed for {source}: {e}")
    except FileNotFoundError as e:
      fatal(f"cat failed: {e}")
    except BrokenPipeError:
      _handle_broken_pipe()
  try:
    out.flush()
  except BrokenPipeError:
    _handle_broken_pipe()


def _handle_broken_pipe():
  """Exit cleanly when the stdout consumer goes away (e.g. `| head`)."""
  devnull = os.open(os.devnull, os.O_WRONLY)
  os.dup2(devnull, sys.stdout.fileno())
  os.close(devnull)
  sys.exit(0)


def handle_ls(args, s3_client):
  bucket, prefix = parse_r2_path(args.path)
  if not bucket:
    fatal("'ls' target must start with r2://")

  try:
    kwargs = {"Bucket": bucket}
    if prefix:
      kwargs["Prefix"] = prefix

    paginator = s3_client.get_paginator("list_objects_v2")
    found = False

    for page in paginator.paginate(**kwargs):
      if "Contents" in page:
        found = True
        for obj in page["Contents"]:
          local_time = obj["LastModified"].astimezone().strftime("%Y-%m-%d %H:%M:%S %z")
          log(f"{local_time}  {format_size(obj['Size']):>14s}  {obj['Key']}",
              quiet=args.quiet)

    if not found:
      log(f"No objects found in r2://{bucket}/{prefix}", quiet=args.quiet)

  except (ClientError, BotoCoreError) as e:
    fatal(f"ls failed: {e}")


def is_dir_destination(destination):
  """Check if the destination should be treated as a directory.

  Returns True if it ends with '/' or is an existing local directory.
  """
  return destination.endswith("/") or os.path.isdir(destination)


def join_dir_destination(destination, name):
  """Join a directory destination with a file name, Unix ``cp``-style.

  ``cp file dir/`` places the file's basename inside the directory:
  ``dir/name``.  ``mv`` behaves the same way.
  """
  if destination.endswith("/"):
    return destination + name
  return destination + "/" + name


def handle_cp(args, s3_client):
  sources = args.sources
  destination = args.destination
  is_dir = is_dir_destination(destination) or len(sources) > 1

  if len(sources) > 1 and not is_dir:
    fatal("Multiple sources require a directory destination (existing "
          "directory or trailing '/').")

  for source in sources:
    # Wildcard R2 source
    if has_wildcard(source):
      src_bucket, src_prefix = parse_r2_path(source)
      if not src_bucket:
        fatal(f"Wildcard source must start with r2://: {source}")
      matches = expand_wildcard(s3_client, source)
      dst_bucket, dst_prefix = parse_r2_path(destination)

      if not dst_bucket:
        # R2 wildcard -> local directory
        local_dir = destination
        if not os.path.isdir(local_dir):
          fatal(f"Destination directory does not exist: {local_dir}")

        total_size = sum(size for _, _, size in matches)
        log(f"Wildcard downloading {len(matches)} object(s) -> {local_dir}/",
            quiet=args.quiet)
        if args.progress:
          with tqdm(total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc="Download",
                    disable=args.quiet) as pbar:
            for _, obj_key, _ in matches:
              local_path = os.path.join(local_dir, os.path.basename(obj_key))
              s3_client.download_file(src_bucket,
                                      obj_key,
                                      local_path,
                                      Callback=pbar.update)
        else:
          for _, obj_key, _ in matches:
            local_path = os.path.join(local_dir, os.path.basename(obj_key))
            s3_client.download_file(src_bucket, obj_key, local_path)
        log(f"Downloaded {len(matches)} object(s).", quiet=args.quiet)

      else:
        # R2 wildcard -> R2 directory
        log(
            f"Wildcard copying {len(matches)} object(s) -> "
            f"r2://{dst_bucket}/{dst_prefix}",
            quiet=args.quiet)
        if args.progress:
          total_size = sum(size for _, _, size in matches)
          with tqdm(total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc="Copy",
                    disable=args.quiet) as pbar:
            for _, obj_key, _ in matches:
              dest_key = compute_dest_key(obj_key, src_prefix, dst_prefix)
              s3_client.copy({
                  "Bucket": src_bucket,
                  "Key": obj_key
              },
                             dst_bucket,
                             dest_key,
                             Callback=pbar.update)
        else:
          for _, obj_key, _ in matches:
            dest_key = compute_dest_key(obj_key, src_prefix, dst_prefix)
            s3_client.copy(
                {
                    "Bucket": src_bucket,
                    "Key": obj_key
                },
                dst_bucket,
                dest_key,
            )
        log(f"Copied {len(matches)} object(s).", quiet=args.quiet)
      continue

    # Single source
    src_bucket, src_key = parse_r2_path(source)

    if is_dir:
      # Unix cp semantics: a directory destination receives the source's
      # basename (cp aaaa/bbbb dest/ -> dest/bbbb, same for r2 keys).
      if src_bucket:
        dest_path = join_dir_destination(destination, os.path.basename(src_key))
      else:
        dest_path = join_dir_destination(destination, os.path.basename(source))
    else:
      dest_path = destination

    dst_bucket, dst_key = parse_r2_path(dest_path)

    try:
      if not src_bucket and dst_bucket:
        log(f"Uploading {source} -> r2://{dst_bucket}/{dst_key}", quiet=args.quiet)
        if args.progress:
          file_size = os.path.getsize(source)
          with tqdm(total=file_size,
                    unit="B",
                    unit_scale=True,
                    desc="Upload",
                    disable=args.quiet) as pbar:
            s3_client.upload_file(source, dst_bucket, dst_key, Callback=pbar.update)
        else:
          s3_client.upload_file(source, dst_bucket, dst_key)

      elif src_bucket and not dst_bucket:
        log(f"Downloading r2://{src_bucket}/{src_key} -> {dest_path}", quiet=args.quiet)
        local_dir = os.path.dirname(dest_path)
        if local_dir:
          os.makedirs(local_dir, exist_ok=True)
        if args.progress:
          head = s3_client.head_object(Bucket=src_bucket, Key=src_key)
          total = head["ContentLength"]
          with tqdm(total=total,
                    unit="B",
                    unit_scale=True,
                    desc="Download",
                    disable=args.quiet) as pbar:
            s3_client.download_file(src_bucket,
                                    src_key,
                                    dest_path,
                                    Callback=pbar.update)
        else:
          s3_client.download_file(src_bucket, src_key, dest_path)
        log("Download complete.", quiet=args.quiet)

      elif src_bucket and dst_bucket:
        log(f"Copying r2://{src_bucket}/{src_key} -> "
            f"r2://{dst_bucket}/{dst_key}",
            quiet=args.quiet)
        copy_source = {"Bucket": src_bucket, "Key": src_key}
        if args.progress:
          head = s3_client.head_object(Bucket=src_bucket, Key=src_key)
          total = head["ContentLength"]
          with tqdm(total=total,
                    unit="B",
                    unit_scale=True,
                    desc="Copy",
                    disable=args.quiet) as pbar:
            s3_client.copy(copy_source, dst_bucket, dst_key, Callback=pbar.update)
        else:
          s3_client.copy(copy_source, dst_bucket, dst_key)
        log("Remote copy complete.", quiet=args.quiet)

      else:
        fatal("Both source and destination cannot be local files.")

    except (ClientError, BotoCoreError) as e:
      fatal(f"cp failed: {e}")


def handle_rm(args, s3_client):
  bucket, key = parse_r2_path(args.path)
  if not bucket or not key:
    fatal("Target must be a specific object starting with r2://bucket/key")

  if args.recursive:
    # List all objects under the prefix and batch-delete them.
    try:
      paginator = s3_client.get_paginator("list_objects_v2")
      total = 0
      deleted = 0
      batch = []
      for page in paginator.paginate(Bucket=bucket, Prefix=key):
        for obj in page.get("Contents", []):
          total += 1
          if args.dry_run:
            log(f"  would delete: {obj['Key']}", quiet=args.quiet)
          else:
            batch.append(obj["Key"])
            if len(batch) >= 1000:
              deleted += batch_delete(s3_client, bucket, batch)
              batch = []

      if args.dry_run:
        log(f"Would delete {total} object(s).", quiet=args.quiet)
        return

      # Flush remaining batch.
      if batch:
        deleted += batch_delete(s3_client, bucket, batch)

      if total == 0:
        log(f"No objects found under r2://{bucket}/{key}", quiet=args.quiet)
      else:
        log(f"Deleted {deleted}/{total} object(s).", quiet=args.quiet)
    except (ClientError, BotoCoreError) as e:
      fatal(f"rm failed: {e}")
  else:
    try:
      log(f"Deleting r2://{bucket}/{key}", quiet=args.quiet)
      s3_client.delete_object(Bucket=bucket, Key=key)
      log("Delete complete.", quiet=args.quiet)
    except (ClientError, BotoCoreError) as e:
      fatal(f"rm failed: {e}")


def handle_du(args, s3_client):
  """Report total size of objects matching an R2 prefix or wildcard pattern."""
  path = args.path
  bucket, prefix = parse_r2_path(path)
  if not bucket:
    fatal("'du' target must start with r2://")

  # If no filter given, list the entire bucket.
  if not prefix:
    prefix = ""

  # Determine if the user passed a wildcard.
  if has_wildcard(path):
    matches = expand_wildcard(s3_client, path)
    total = sum(size for _, _, size in matches)
    log(f"{format_size(total)}\t{len(matches)} object(s)\t{path}", quiet=args.quiet)
  else:
    # Plain prefix listing — accumulate sizes as we paginate.
    try:
      paginator = s3_client.get_paginator("list_objects_v2")
      total = 0
      count = 0
      for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
          total += obj["Size"]
          count += 1
      log(f"{format_size(total)}\t{count} object(s)\tr2://{bucket}/{prefix}",
          quiet=args.quiet)
    except (ClientError, BotoCoreError) as e:
      fatal(f"du failed: {e}")


def batch_delete(s3_client, bucket, keys):
  """Delete a list of keys in batches of 1000, checking for errors."""
  deleted = 0
  errors = []
  for i in range(0, len(keys), 1000):
    batch = keys[i:i + 1000]
    resp = s3_client.delete_objects(
        Bucket=bucket,
        Delete={
            "Objects": batch,
            "Quiet": True
        },
    )
    deleted += len(batch) - len(resp.get("Errors", []))
    for err in resp.get("Errors", []):
      errors.append(f"  {err['Key']}: {err['Code']} - {err.get('Message', '')}")
  if errors:
    print(f"WARNING: {len(errors)} deletion error(s):", file=sys.stderr)
    for e in errors:
      print(e, file=sys.stderr)
  return deleted


def local_file_info(path):
  """Return (size, mtime_epoch) for a local file."""
  stat = os.stat(path)
  return stat.st_size, stat.st_mtime


def r2_object_index(s3_client, bucket, prefix):
  """Build {key: (size, last_modified_epoch)} for all objects under prefix."""
  index = {}
  paginator = s3_client.get_paginator("list_objects_v2")
  for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
    for obj in page.get("Contents", []):
      index[obj["Key"]] = (obj["Size"], obj["LastModified"].timestamp())
  return index


def should_upload(local_size, local_mtime, remote_size, remote_mtime):
  """Decide whether a local file should be uploaded based on size/mtime."""
  if remote_size is None:
    # Object doesn't exist remotely.
    return True
  return (local_size != remote_size) or (local_mtime > remote_mtime)


def should_download(local_size, local_mtime, remote_size, remote_mtime):
  """Decide whether a remote object should be downloaded to local."""
  if local_size is None:
    # Local file doesn't exist.
    return True
  return (local_size != remote_size) or (remote_mtime > local_mtime)


def handle_sync(args, s3_client):
  """Synchronize files between local directory and R2 prefix."""
  src = args.source
  dst = args.destination

  # Determine direction from the r2:// prefixes.
  src_bucket, src_prefix = parse_r2_path(src)
  dst_bucket, dst_prefix = parse_r2_path(dst)

  if src_bucket and dst_bucket:
    fatal("sync between two R2 paths is not supported. Use cp instead.")

  if not src_bucket and not dst_bucket:
    fatal("sync between two local paths is not supported. Use cp instead.")

  if src_bucket:
    # R2 -> local direction.
    sync_r2_to_local(s3_client, src_bucket, src_prefix, dst, args)
  else:
    # Local -> R2 direction.
    sync_local_to_r2(s3_client, src, dst_bucket, dst_prefix, args)


def sync_local_to_r2(s3_client, local_dir, r2_bucket, r2_prefix, args):
  """Upload changed local files to R2."""
  if not os.path.isdir(local_dir):
    fatal(f"Source directory does not exist: {local_dir}")

  # Ensure r2_prefix ends with '/' for proper key construction.
  if r2_prefix and not r2_prefix.endswith("/"):
    r2_prefix += "/"

  # Build remote index.
  remote = r2_object_index(s3_client, r2_bucket, r2_prefix)

  # Collect local files.
  to_upload = []
  local_rel_keys = set()
  for root, _, files in os.walk(local_dir):
    for fname in files:
      local_path = os.path.join(root, fname)
      rel_path = os.path.relpath(local_path, local_dir)
      # R2 key uses forward slashes, no leading '/'.
      r2_key = (r2_prefix + rel_path).replace("\\", "/")
      local_rel_keys.add(r2_key)

      local_size, local_mtime = local_file_info(local_path)
      remote_size, remote_mtime = remote.get(r2_key, (None, None))

      # Apply exclude filter.
      if args.exclude and fnmatch.fnmatch(rel_path, args.exclude):
        continue

      if should_upload(local_size, local_mtime, remote_size, remote_mtime):
        to_upload.append((local_path, r2_key, local_size))

  # Handle --delete: remove remote objects not present locally.
  to_delete = []
  if args.delete:
    for r2_key in remote:
      if r2_key not in local_rel_keys:
        # Don't delete prefix markers or things we excluded.
        if args.exclude and fnmatch.fnmatch(os.path.relpath(r2_key, r2_prefix),
                                            args.exclude):
          continue
        to_delete.append(r2_key)

  if args.dry_run:
    for local_path, r2_key, size in to_upload:
      log(f"  upload: {local_path} -> r2://{r2_bucket}/{r2_key}  ({format_size(size)})",
          quiet=args.quiet)
    for r2_key in to_delete:
      log(f"  delete: r2://{r2_bucket}/{r2_key}", quiet=args.quiet)
    log(
        f"Would upload {len(to_upload)}, delete {len(to_delete)}, "
        f"skip {len(remote) - len(to_upload) - len(to_delete)}.",
        quiet=args.quiet)
    return

  # Upload.
  total_size = sum(size for _, _, size in to_upload)
  log(
      f"Syncing {len(to_upload)} file(s) ({format_size(total_size)}) -> "
      f"r2://{r2_bucket}/{r2_prefix}",
      quiet=args.quiet)
  if args.progress:
    with tqdm(total=total_size,
              unit="B",
              unit_scale=True,
              desc="Upload",
              disable=args.quiet) as pbar:
      for local_path, r2_key, _size in to_upload:
        s3_client.upload_file(local_path, r2_bucket, r2_key, Callback=pbar.update)
  else:
    for local_path, r2_key, _size in to_upload:
      s3_client.upload_file(local_path, r2_bucket, r2_key)

  # Delete.
  if to_delete:
    log(f"Deleting {len(to_delete)} object(s) ...", quiet=args.quiet)
    batch_delete(s3_client, r2_bucket, to_delete)

  log(
      f"Sync complete: {len(to_upload)} uploaded, {len(to_delete)} deleted, "
      f"{len(remote) - len(to_upload)} unchanged.",
      quiet=args.quiet)


def sync_r2_to_local(s3_client, r2_bucket, r2_prefix, local_dir, args):
  """Download changed R2 objects to local directory."""
  if not local_dir:
    fatal("Local destination must be a directory.")

  if r2_prefix and not r2_prefix.endswith("/"):
    r2_prefix += "/"

  # Build remote index.
  remote = r2_object_index(s3_client, r2_bucket, r2_prefix)

  # Build local index (keys relative to r2_prefix).
  # {rel_path: (size, mtime)}.
  local_files = {}
  if os.path.isdir(local_dir):
    for root, _, files in os.walk(local_dir):
      for fname in files:
        local_path = os.path.join(root, fname)
        rel_path = os.path.relpath(local_path, local_dir)
        local_files[rel_path] = local_file_info(local_path)

  to_download = []
  for r2_key, (remote_size, remote_mtime) in remote.items():
    # Strip prefix.
    rel_path = r2_key[len(r2_prefix):]
    if not rel_path:
      # Skip the prefix itself if it's an object.
      continue

    # Apply exclude filter.
    if args.exclude and fnmatch.fnmatch(rel_path, args.exclude):
      continue

    local_size, local_mtime = local_files.get(rel_path, (None, None))
    if should_download(local_size, local_mtime, remote_size, remote_mtime):
      to_download.append((r2_key, rel_path, remote_size))

  # Handle --delete: remove local files not present on R2.
  to_delete = []
  if args.delete:
    remote_rel_keys = set()
    for r2_key in remote:
      rel_path = r2_key[len(r2_prefix):]
      if rel_path:
        remote_rel_keys.add(rel_path)
    for rel_path in local_files:
      if rel_path not in remote_rel_keys:
        if args.exclude and fnmatch.fnmatch(rel_path, args.exclude):
          continue
        to_delete.append(rel_path)

  if args.dry_run:
    for r2_key, rel_path, size in to_download:
      log(
          f"  download: r2://{r2_bucket}/{r2_key} -> "
          f"{os.path.join(local_dir, rel_path)}  ({format_size(size)})",
          quiet=args.quiet)
    for rel_path in to_delete:
      log(f"  delete: {os.path.join(local_dir, rel_path)}", quiet=args.quiet)
    log(
        f"Would download {len(to_download)}, delete {len(to_delete)}, "
        f"skip {len(remote) - len(to_download)}.",
        quiet=args.quiet)
    return

  # Download.
  total_size = sum(size for _, _, size in to_download)
  log(
      f"Syncing {len(to_download)} file(s) ({format_size(total_size)}) <- "
      f"r2://{r2_bucket}/{r2_prefix}",
      quiet=args.quiet)
  if args.progress:
    with tqdm(total=total_size,
              unit="B",
              unit_scale=True,
              desc="Download",
              disable=args.quiet) as pbar:
      for r2_key, rel_path, _size in to_download:
        local_path = os.path.join(local_dir, rel_path)
        local_dir_path = os.path.dirname(local_path)
        if local_dir_path:
          os.makedirs(local_dir_path, exist_ok=True)
        s3_client.download_file(r2_bucket, r2_key, local_path, Callback=pbar.update)
        # Sync local mtime to match remote so next sync doesn't re-download.
        remote_mtime = remote[r2_key][1]
        os.utime(local_path, (remote_mtime, remote_mtime))
  else:
    for r2_key, rel_path, _size in to_download:
      local_path = os.path.join(local_dir, rel_path)
      local_dir_path = os.path.dirname(local_path)
      if local_dir_path:
        os.makedirs(local_dir_path, exist_ok=True)
      s3_client.download_file(r2_bucket, r2_key, local_path)
      # Sync local mtime to match remote so next sync doesn't re-download.
      remote_mtime = remote[r2_key][1]
      os.utime(local_path, (remote_mtime, remote_mtime))

  # Delete.
  for rel_path in to_delete:
    local_path = os.path.join(local_dir, rel_path)
    log(f"Deleting {local_path}", quiet=args.quiet)
    os.remove(local_path)

  log(
      f"Sync complete: {len(to_download)} downloaded, {len(to_delete)} deleted, "
      f"{len(remote) - len(to_download)} unchanged.",
      quiet=args.quiet)


def handle_presign(args, s3_client):
  """Generate a time-limited pre-signed download URL for an R2 object."""
  bucket, key = parse_r2_path(args.path)
  if not bucket or not key:
    fatal("Target must be a specific object starting with r2://bucket/key")

  try:
    url = s3_client.generate_presigned_url(
        "get_object",
        Params={
            "Bucket": bucket,
            "Key": key
        },
        ExpiresIn=args.expires,
    )
    print(url)
  except (ClientError, BotoCoreError) as e:
    fatal(f"presign failed: {e}")


def handle_mb(args, s3_client):
  """Create a new R2 bucket."""
  bucket_name = args.bucket
  try:
    # R2 requires LocationConstraint when creating buckets.
    s3_client.create_bucket(
        Bucket=bucket_name,
        CreateBucketConfiguration={"LocationConstraint": "auto"},
    )
    log(f"Bucket created: r2://{bucket_name}", quiet=args.quiet)
  except (ClientError, BotoCoreError) as e:
    if hasattr(e, 'response') and e.response.get(
        "Error", {}).get("Code") == "BucketAlreadyExists":
      fatal(f"Bucket already exists: {bucket_name}")
    fatal(f"mb failed: {e}")


def handle_rb(args, s3_client):
  """Remove an R2 bucket. Must be empty unless --force is used."""
  bucket_name = args.bucket
  try:
    # Check if bucket has objects.
    resp = s3_client.list_objects_v2(Bucket=bucket_name, MaxKeys=1)
    has_objects = resp.get("KeyCount", 0) > 0

    if has_objects and not args.force:
      fatal("Bucket is not empty. Use --force to delete all objects first.")

    if has_objects and args.force:
      # Delete all objects recursively (streaming, no OOM).
      paginator = s3_client.get_paginator("list_objects_v2")
      total = 0
      deleted = 0
      batch = []
      for page in paginator.paginate(Bucket=bucket_name):
        for obj in page.get("Contents", []):
          total += 1
          batch.append(obj["Key"])
          if len(batch) >= 1000:
            deleted += batch_delete(s3_client, bucket_name, batch)
            batch = []
      if batch:
        deleted += batch_delete(s3_client, bucket_name, batch)
      log(f"Deleted {deleted}/{total} object(s) from {bucket_name} ...",
          quiet=args.quiet)

    s3_client.delete_bucket(Bucket=bucket_name)
    log(f"Bucket deleted: r2://{bucket_name}", quiet=args.quiet)
  except (ClientError, BotoCoreError) as e:
    if hasattr(e, 'response') and e.response.get("Error",
                                                 {}).get("Code") == "NoSuchBucket":
      fatal(f"Bucket does not exist: {bucket_name}")
    fatal(f"rb failed: {e}")


def parse_relative_date(s):
  """Parse a relative date string like '7d', '24h', '30m' into seconds."""
  m = re.match(r"^(\d+)([dhms])$", s)
  if not m:
    fatal(f"Invalid relative date: {s}. Use e.g. '7d', '24h', '30m'.")
  val, unit = int(m.group(1)), m.group(2)
  return val * {"d": 86400, "h": 3600, "m": 60, "s": 1}[unit]


def parse_date_filter(s):
  """Parse a date filter: ISO 8601 or relative ('7d', '24h', '-7d').

  Returns epoch seconds. Relative dates are computed from now.
  The leading '-' is optional for relative dates.
  """
  # Try relative date first (e.g. '7d', '-7d', '24h').
  stripped = s.lstrip("-")
  m = re.match(r"^(\d+)([dhms])$", stripped)
  if m:
    return time.time() - parse_relative_date(stripped)

  # Try ISO 8601 date.
  try:
    return dt.datetime.fromisoformat(s).timestamp()
  except ValueError:
    fatal(f"Invalid date format: {s}. Use ISO 8601 (e.g. 2025-01-15) "
          "or relative (e.g. 7d, -24h).")


def handle_find(args, s3_client):
  """Filter R2 objects by size, date, or name pattern under a prefix."""
  bucket, prefix = parse_r2_path(args.path)
  if not bucket:
    fatal("'find' target must start with r2://")
  if not prefix:
    prefix = ""

  # Parse date filters into epoch seconds.
  newer_than = parse_date_filter(args.newer_than) if args.newer_than else None
  older_than = parse_date_filter(args.older_than) if args.older_than else None

  ext_filter = f".{args.ext}" if args.ext else None

  try:
    paginator = s3_client.get_paginator("list_objects_v2")
    matches = 0
    total_size = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
      for obj in page.get("Contents", []):
        key = obj["Key"]
        size = obj["Size"]
        last_modified = obj["LastModified"].timestamp()

        # Apply filters.
        if args.min_size is not None and size < args.min_size:
          continue
        if args.max_size is not None and size > args.max_size:
          continue
        if newer_than is not None and last_modified < newer_than:
          continue
        if older_than is not None and last_modified > older_than:
          continue
        if ext_filter and not key.endswith(ext_filter):
          continue
        if args.name and not fnmatch.fnmatch(os.path.basename(key), args.name):
          continue

        log(
            f"{format_date(obj['LastModified'])}\t{format_size(size)}\t"
            f"r2://{bucket}/{key}",
            quiet=args.quiet)
        matches += 1
        total_size += size
  except (ClientError, BotoCoreError) as e:
    fatal(f"find failed: {e}")


def handle_mv(args, s3_client):
  """Renames R2 objects (copy + delete). All paths must be r2://."""
  sources = args.sources
  destination = args.destination
  is_dir = destination.endswith("/") or len(sources) > 1

  dst_bucket, _ = parse_r2_path(destination)
  if not dst_bucket:
    fatal("Destination must be an R2 path (r2://bucket/key)")

  if len(sources) > 1 and not is_dir:
    fatal("Multiple sources require a directory destination "
          "(ending with '/').")

  for source in sources:
    src_bucket, _ = parse_r2_path(source)
    if not src_bucket:
      fatal(f"Source must be an R2 path: {source}")
    # Extract the key portion, stripping leading '/' to avoid
    # creating double-slash R2 URLs.
    src_key = source.split("/", 3)[3].lstrip("/")

    # Wildcard source
    if has_wildcard(source):
      # Already normalized.
      src_prefix = src_key
      matches = expand_wildcard(s3_client, source)
      _, dst_prefix = parse_r2_path(destination)

      log(f"Wildcard moving {len(matches)} object(s):", quiet=args.quiet)
      try:
        total_size = sum(size for _, _, size in matches)
        if args.progress:
          with tqdm(total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc="Move",
                    disable=args.quiet) as pbar:
            for _, obj_key, _ in matches:
              dest_key = compute_dest_key(obj_key, src_prefix, dst_prefix)
              s3_client.copy({
                  "Bucket": src_bucket,
                  "Key": obj_key
              },
                             dst_bucket,
                             dest_key,
                             Callback=pbar.update)
              s3_client.delete_object(Bucket=src_bucket, Key=obj_key)
        else:
          for _, obj_key, _ in matches:
            dest_key = compute_dest_key(obj_key, src_prefix, dst_prefix)
            s3_client.copy(
                {
                    "Bucket": src_bucket,
                    "Key": obj_key
                },
                dst_bucket,
                dest_key,
            )
            s3_client.delete_object(Bucket=src_bucket, Key=obj_key)
        log(f"Moved {len(matches)} object(s).", quiet=args.quiet)
      except (ClientError, BotoCoreError) as e:
        fatal(f"mv failed: {e}")
      continue

    # Single source
    _, dst_prefix = parse_r2_path(destination)
    src_basename = os.path.basename(src_key)
    dest_key = dst_prefix + src_basename if is_dir else dst_prefix

    log(f"Moving r2://{src_bucket}/{src_key} -> r2://{dst_bucket}/{dest_key}",
        quiet=args.quiet)
    try:
      s3_client.copy(
          {
              "Bucket": src_bucket,
              "Key": src_key
          },
          dst_bucket,
          dest_key,
      )
      s3_client.delete_object(Bucket=src_bucket, Key=src_key)
      log("Move complete.", quiet=args.quiet)
    except (ClientError, BotoCoreError) as e:
      fatal(f"mv failed: {e}")


def main():
  parser = argparse.ArgumentParser(
      description="Cloudflare R2 CLI tool mimicking AWS S3 commands.",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter,
  )
  parser.add_argument("-q",
                      "--quiet",
                      action="store_true",
                      help="Suppress all output except errors")
  subparsers = parser.add_subparsers(dest="command",
                                     required=True,
                                     help="Sub-command to run")

  parser_ls = subparsers.add_parser(
      "ls", help="List objects in a bucket prefix (Format: r2://bucket-name/prefix)")
  parser_ls.add_argument("path", help="Target bucket path starting with r2://")
  parser_ls.set_defaults(func=handle_ls)

  parser_cat = subparsers.add_parser(
      "cat", help="Stream R2 objects and/or local files to stdout")
  parser_cat.add_argument(
      "sources",
      nargs="+",
      help="Files to concatenate: local paths and/or r2://bucket/key. "
      "Wildcards (*, ?) are supported for R2 sources.")
  parser_cat.set_defaults(func=handle_cat)

  parser_cp = subparsers.add_parser("cp", help="Copy files locally or remotely")
  parser_cp.add_argument("sources",
                         nargs="+",
                         help="One or more source paths (local or r2://bucket/key). "
                         "Wildcards (*, ?) are supported for R2 sources.")
  parser_cp.add_argument("destination",
                         help="Destination path (local or r2://bucket/key). "
                         "When multiple sources are given, the destination is "
                         "treated as a directory.")
  parser_cp.add_argument("--progress",
                         action="store_true",
                         help="Show a progress bar during transfer")
  parser_cp.set_defaults(func=handle_cp)

  parser_rm = subparsers.add_parser("rm", help="Remove an object or directory from R2")
  parser_rm.add_argument("path", help="Target object key path starting with r2://")
  parser_rm.add_argument("-r",
                         "--recursive",
                         action="store_true",
                         help="Delete all objects under the given prefix")
  parser_rm.add_argument("--dry-run",
                         action="store_true",
                         help="Show what would be deleted without deleting")
  parser_rm.set_defaults(func=handle_rm)

  parser_mv = subparsers.add_parser("mv", help="Rename R2 objects (copy + delete)")
  parser_mv.add_argument("sources",
                         nargs="+",
                         help="One or more R2 source paths (r2://bucket/key). "
                         "Wildcards (*, ?) are supported.")
  parser_mv.add_argument("destination",
                         help="Destination R2 path (r2://bucket/key). "
                         "When multiple sources are given, the destination is "
                         "treated as a directory prefix.")
  parser_mv.set_defaults(func=handle_mv)

  parser_du = subparsers.add_parser(
      "du", help="Report total size of objects under a prefix or matching a wildcard")
  parser_du.add_argument(
      "path",
      help="R2 prefix or wildcard pattern (e.g. r2://bucket/prefix/ or "
      "r2://bucket/prefix/*.pt)")
  parser_du.set_defaults(func=handle_du)

  parser_presign = subparsers.add_parser(
      "presign",
      help="Generate a time-limited pre-signed download URL for an R2 object")
  parser_presign.add_argument("path", help="R2 object path (r2://bucket/key)")
  parser_presign.add_argument("--expires",
                              type=int,
                              default=3600,
                              help="URL validity in seconds")
  parser_presign.set_defaults(func=handle_presign)

  parser_find = subparsers.add_parser(
      "find", help="Filter R2 objects by size, date, or name pattern under a prefix")
  parser_find.add_argument("path", help="R2 prefix (r2://bucket/prefix/)")
  parser_find.add_argument("--min-size", type=int, help="Minimum object size in bytes")
  parser_find.add_argument("--max-size", type=int, help="Maximum object size in bytes")
  parser_find.add_argument(
      "--newer-than",
      help="Only objects modified after this date (ISO 8601 or relative "
      "like -7d, -24h)")
  parser_find.add_argument(
      "--older-than",
      help="Only objects modified before this date (ISO 8601 or relative "
      "like -7d, -24h)")
  parser_find.add_argument("--name",
                           help="Glob pattern to match against the object "
                           "basename (e.g. '*.pt')")
  parser_find.add_argument("--ext", help="Shorthand for --name *.EXT")
  parser_find.set_defaults(func=handle_find)

  parser_sync = subparsers.add_parser(
      "sync", help="Synchronize files between a local directory and an R2 prefix")
  parser_sync.add_argument("source",
                           help="Source path: local directory or r2://bucket/prefix/")
  parser_sync.add_argument(
      "destination", help="Destination path: r2://bucket/prefix/ or local directory")
  parser_sync.add_argument(
      "--delete",
      action="store_true",
      help="Remove files at destination that don't exist at source")
  parser_sync.add_argument("--dry-run",
                           action="store_true",
                           help="Show what would be transferred without doing it")
  parser_sync.add_argument("--progress",
                           action="store_true",
                           help="Show progress bars during transfer")
  parser_sync.add_argument(
      "--exclude",
      help="Glob pattern for files to skip (matched against relative path)")
  parser_sync.set_defaults(func=handle_sync)

  parser_mb = subparsers.add_parser("mb", help="Create a new R2 bucket")
  parser_mb.add_argument("bucket", help="Bucket name (without r2:// prefix)")
  parser_mb.set_defaults(func=handle_mb)

  parser_rb = subparsers.add_parser("rb", help="Remove an R2 bucket")
  parser_rb.add_argument("bucket", help="Bucket name (without r2:// prefix)")
  parser_rb.add_argument("--force",
                         action="store_true",
                         help="Delete all objects before removing the bucket")
  parser_rb.set_defaults(func=handle_rb)

  args = parser.parse_args()
  s3_client = create_s3_client()
  args.func(args, s3_client)


if __name__ == "__main__":
  main()
