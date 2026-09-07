"""Unified cloud storage helpers for GCS, Cloudflare R2, and AWS S3."""

import contextlib
import logging
import os
import tempfile

import torch

from scdiag.logging_utils import fatal

# Umask captured once at import time so that files created through
# ``tempfile.mkstemp`` (which forces 0600) can be given the same mode a
# plain ``open()`` would have produced.
_UMASK = os.umask(0)
os.umask(_UMASK)


def parse_storage_uri(uri):
  """Parse a ``gs://``, ``r2://``, or ``s3://`` URI into ``(scheme, bucket,
  prefix)``.

    Args:
        uri: A storage URI.  Must start with ``gs://`` (Google Cloud
            Storage), ``r2://`` (Cloudflare R2), or ``s3://`` (AWS S3).

    Returns:
        A tuple ``(scheme, bucket, prefix)`` where *scheme* is ``"gs"``,
        ``"r2"``, or ``"s3"``.
  """
  if uri.startswith("gs://"):
    scheme = "gs"
    without_scheme = uri[len("gs://"):]
  elif uri.startswith("r2://"):
    scheme = "r2"
    without_scheme = uri[len("r2://"):]
  elif uri.startswith("s3://"):
    scheme = "s3"
    without_scheme = uri[len("s3://"):]
  else:
    fatal(f"URI must start with gs://, r2://, or s3://, got: {uri}", ValueError)

  parts = without_scheme.split("/", 1)
  bucket = parts[0]
  prefix = parts[1] if len(parts) > 1 else ""
  return scheme, bucket, prefix


def _upload_s3(bucket_name, local_path, prefix):
  """Upload *local_path* to an AWS S3 bucket under *prefix*.

    Credentials are taken from the standard AWS environment variables
    (``AWS_ACCESS_KEY_ID``, ``AWS_SECRET_ACCESS_KEY``, and optionally
    ``AWS_SESSION_TOKEN`` for temporary STS credentials).  When the key
    variables are unset, boto3's default credential chain is used
    instead (IAM instance role, ``~/.aws/credentials``, SSO cache).
  """
  import boto3

  access_key = os.environ.get("AWS_ACCESS_KEY_ID")
  secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
  client_kwargs = {}
  if access_key and secret_key:
    client_kwargs = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "aws_session_token": os.environ.get("AWS_SESSION_TOKEN"),
    }
  s3_client = boto3.client("s3", **client_kwargs)

  blob_name = os.path.basename(local_path)
  if prefix:
    blob_name = f"{prefix}/{blob_name}"
  s3_client.upload_file(local_path, bucket_name, blob_name)
  return f"s3://{bucket_name}/{blob_name}"


def _upload_gcs(bucket_name, local_path, prefix):
  """Upload *local_path* to a GCS bucket under *prefix*."""
  from google.cloud import storage

  client = storage.Client()
  blob_name = os.path.basename(local_path)
  if prefix:
    blob_name = f"{prefix}/{blob_name}"
  bucket = client.bucket(bucket_name)
  blob = bucket.blob(blob_name)
  blob.upload_from_filename(local_path)
  return f"gs://{bucket_name}/{blob_name}"


def _r2_client():
  """Build a boto3 client targeting the Cloudflare R2 S3 API.

  The endpoint is taken from ``R2_ENDPOINT_URL`` when set, and otherwise
  derived from ``CLOUDFLARE_ACCOUNT_ID``.

    Returns:
        A boto3 S3 client configured for R2.
  """
  import boto3

  endpoint_url = os.environ.get("R2_ENDPOINT_URL")
  if not endpoint_url:
    account_id = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    if account_id:
      endpoint_url = f"https://{account_id}.r2.cloudflarestorage.com"
    else:
      fatal("CLOUDFLARE_ACCOUNT_ID or R2_ENDPOINT_URL must be set for R2 "
            "access.", ValueError)
  return boto3.client(
      "s3",
      endpoint_url=endpoint_url,
      region_name="auto",
      aws_access_key_id=os.environ.get("R2_ACCESS_KEY_ID"),
      aws_secret_access_key=os.environ.get("R2_SECRET_ACCESS_KEY"),
  )


def _upload_r2(bucket_name, local_path, prefix):
  """Upload *local_path* to a Cloudflare R2 bucket under *prefix*."""
  s3_client = _r2_client()

  blob_name = os.path.basename(local_path)
  if prefix:
    blob_name = f"{prefix}/{blob_name}"
  s3_client.upload_file(local_path, bucket_name, blob_name)
  return f"r2://{bucket_name}/{blob_name}"


def storage_upload(bucket_name, local_path, prefix="", scheme="gs"):
  """Upload *local_path* to cloud storage.

    Args:
        bucket_name: Target bucket name.
        local_path: Local file to upload.
        prefix: Optional key prefix inside the bucket.
        scheme: ``"gs"`` for Google Cloud Storage, ``"r2"`` for
            Cloudflare R2, or ``"s3"`` for AWS S3.

    Returns:
        The full remote URI of the uploaded object.
  """
  if scheme == "gs":
    return _upload_gcs(bucket_name, local_path, prefix)
  elif scheme == "r2":
    return _upload_r2(bucket_name, local_path, prefix)
  elif scheme == "s3":
    return _upload_s3(bucket_name, local_path, prefix)
  else:
    fatal(f"Unsupported storage scheme: {scheme!r}", ValueError)


def _s3_head_exists(s3_client, bucket_name, blob_name):
  """Return ``True`` when *blob_name* exists in *bucket_name*.

    A missing object maps to ``False``; any other error (credentials,
    connectivity, permissions) propagates to the caller so it can be
    reported as a failed fetch instead of a silent miss.
    """
  try:
    s3_client.head_object(Bucket=bucket_name, Key=blob_name)
  except s3_client.exceptions.ClientError as error:
    code = str(error.response.get("Error", {}).get("Code", ""))
    if code in ("404", "NoSuchKey"):
      return False
    raise
  return True


def _download_s3(bucket_name, local_path, prefix):
  """Download an object from AWS S3 to *local_path*.

    The object key mirrors :func:`_upload_s3` exactly
    (``{prefix}/{basename}``), and credentials follow the same strategy.

    Returns:
        The remote URI of the downloaded object, or ``None`` when the
        object does not exist.
    """
  import boto3

  access_key = os.environ.get("AWS_ACCESS_KEY_ID")
  secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
  client_kwargs = {}
  if access_key and secret_key:
    client_kwargs = {
        "aws_access_key_id": access_key,
        "aws_secret_access_key": secret_key,
        "aws_session_token": os.environ.get("AWS_SESSION_TOKEN"),
    }
  s3_client = boto3.client("s3", **client_kwargs)

  blob_name = os.path.basename(local_path)
  if prefix:
    blob_name = f"{prefix}/{blob_name}"
  if not _s3_head_exists(s3_client, bucket_name, blob_name):
    return None
  s3_client.download_file(bucket_name, blob_name, local_path)
  return f"s3://{bucket_name}/{blob_name}"


def _download_gcs(bucket_name, local_path, prefix):
  """Download an object from GCS to *local_path*.

    Returns:
        The remote URI of the downloaded object, or ``None`` when the
        object does not exist.
    """
  from google.cloud import storage

  client = storage.Client()
  blob_name = os.path.basename(local_path)
  if prefix:
    blob_name = f"{prefix}/{blob_name}"
  bucket = client.bucket(bucket_name)
  blob = bucket.blob(blob_name)
  if not blob.exists():
    return None
  blob.download_to_filename(local_path)
  return f"gs://{bucket_name}/{blob_name}"


def _download_r2(bucket_name, local_path, prefix):
  """Download an object from Cloudflare R2 to *local_path*.

    Returns:
        The remote URI of the downloaded object, or ``None`` when the
        object does not exist.
    """
  s3_client = _r2_client()
  blob_name = os.path.basename(local_path)
  if prefix:
    blob_name = f"{prefix}/{blob_name}"
  if not _s3_head_exists(s3_client, bucket_name, blob_name):
    return None
  s3_client.download_file(bucket_name, blob_name, local_path)
  return f"r2://{bucket_name}/{blob_name}"


def storage_download(remote_uri, local_path):
  """Download a remote object into *local_path*.

    The remote object name mirrors :func:`storage_upload` exactly:
    ``{prefix}/{basename(local_path)}``.  A missing object is a normal,
    silent miss; connection or credential problems are logged as
    warnings because the download is a resume fallback and must never
    crash startup.

    Args:
        remote_uri: ``gs://``, ``r2://``, or ``s3://`` prefix URI (the
            ``--remote_checkpoint`` value).
        local_path: Destination file path; its basename selects the
            remote object.

    Returns:
        ``True`` when *local_path* was written from remote storage.
    """
  scheme, bucket, prefix = parse_storage_uri(remote_uri)
  try:
    if scheme == "gs":
      result = _download_gcs(bucket, local_path, prefix)
    elif scheme == "r2":
      result = _download_r2(bucket, local_path, prefix)
    elif scheme == "s3":
      result = _download_s3(bucket, local_path, prefix)
    else:
      fatal(f"Unsupported storage scheme: {scheme!r}", ValueError)
  except Exception as error:
    logging.warning(f"  Remote checkpoint fetch from {remote_uri} failed: {error}")
    return False
  if result is None:
    return False
  logging.info(f"  Fetched remote checkpoint: {result}")
  return True


def save_checkpoint(save_dict, path, remote_uri=None):
  """Save a checkpoint dict to disk and optionally sync to cloud storage.

    Safe to call concurrently from multiple processes: the write goes to
    a process-exclusive temp file that is atomically renamed into place,
    so *path* only ever contains a fully written checkpoint.

    Args:
        save_dict: Dictionary to pass to ``torch.save``.
        path: Local file path for the checkpoint.
        remote_uri: Optional ``gs://``, ``r2://``, or ``s3://`` URI to
            upload to.
    """
  dirname = os.path.dirname(path)
  if dirname:
    os.makedirs(dirname, exist_ok=True)
  # Write to a unique, process-exclusive temporary file first, then
  # atomically rename so a crash mid-write can never leave a truncated
  # checkpoint at *path* (which is also the resume source for
  # ``_latest.pt``).  ``mkstemp`` hands out a distinct name per call and
  # creates the file with ``O_EXCL``, so concurrent processes saving to
  # the same *path* cannot clobber each other's temp file, publish a
  # partially written one, or delete a file another writer is still
  # using.  The temp file must live in the destination directory so the
  # rename stays on a single filesystem.
  fd, tmp_path = tempfile.mkstemp(dir=dirname or ".",
                                  prefix=f".{os.path.basename(path)}.",
                                  suffix=".tmp")
  tmp_file = None
  try:
    # ``mkstemp`` creates the file 0600; restore the mode a plain
    # ``open()`` would have produced so the published checkpoint keeps
    # its usual permissions.
    os.chmod(tmp_path, 0o666 & ~_UMASK)
    tmp_file = os.fdopen(fd, "wb")
    with tmp_file:
      torch.save(save_dict, tmp_file)
      # Flush and fsync before the rename so that after a crash or power
      # loss the file visible at *path* is always fully written, never
      # truncated or empty.
      tmp_file.flush()
      os.fsync(tmp_file.fileno())
    os.replace(tmp_path, path)
  except BaseException:
    # Best-effort cleanup: close the descriptor if the write failed (or
    # if anything raised before ``fdopen`` took ownership of it), then
    # unlink the partial temp file.  The original exception propagates.
    with contextlib.suppress(OSError):
      if tmp_file is not None:
        tmp_file.close()
      else:
        os.close(fd)
    with contextlib.suppress(OSError):
      os.remove(tmp_path)
    raise
  # The local commit is durable at this point; the remote sync is
  # best-effort and must stay after the rename.
  if remote_uri:
    scheme, bucket, prefix = parse_storage_uri(remote_uri)
    remote = storage_upload(bucket, path, prefix, scheme=scheme)
    logging.info(f"  Synced to remote: {remote}")
  return path
