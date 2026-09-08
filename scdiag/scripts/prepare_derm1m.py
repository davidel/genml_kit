#!/usr/bin/env python
"""Prepare Derm1M dataset for pretraining by extracting images from zip archives.

The redlessone/Derm1M HuggingFace dataset contains CSV metadata with filenames
that reference images inside zip archives (IIYI.zip, ISIC.zip, etc.). This
script downloads and extracts those archives, then moves the images into a
flat ImageFolder directory compatible with genml-kit-pretrain.

Usage:
    python scripts/prepare_derm1m.py --output_dir ./derm1m_images
    python scripts/prepare_derm1m.py --output_dir ./derm1m_images --min_resolution 224

    Then use with pretraining:
    genml-kit-pretrain --datasets ./derm1m_images --image_size 448 ...
"""

import argparse
import logging
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

# All zip file extensions we look for in the Derm1M dataset repo.
_ZIP_EXT = ".zip"
_DEFAULT_REPO_ID = "redlessone/Derm1M"

BASE_URL = "https://huggingface.co/datasets/redlessone/Derm1M/resolve/main"


def fetch_zip_names(repo_id=_DEFAULT_REPO_ID, token=None):
  """Fetch the list of zip archive names from a HuggingFace dataset repo.

    Uses the HuggingFace API to dynamically discover zip files instead
    of maintaining a hardcoded list.
    """
  from huggingface_hub import HfApi

  api = HfApi()
  all_files = api.list_repo_files(repo_id=repo_id, repo_type="dataset", token=token)
  zips = sorted(f for f in all_files if f.endswith(_ZIP_EXT) and not f.startswith("."))
  if not zips:
    raise RuntimeError(f"No zip files found in {repo_id}. Check the repo name or "
                       "access token.")
  return zips


def download_file(url, dest_path, token=None):
  """Download a file using huggingface_hub for better performance."""
  from huggingface_hub import hf_hub_download

  # Extract repo_id and filename from URL
  # URL format: https://huggingface.co/datasets/redlessone/Derm1M/resolve/main/IIYI.zip
  parts = url.split("/datasets/")[-1].split("/resolve/")
  repo_id = parts[0]  # redlessone/Derm1M
  filename = parts[1].split("/", 1)[1] if "/" in parts[1] else parts[1]

  print(f"  Downloading {filename} ...")
  downloaded_path = hf_hub_download(
      repo_id=repo_id,
      filename=filename,
      repo_type="dataset",
      local_dir=os.path.dirname(dest_path),
      token=token,
  )

  # Move to expected location if different
  if Path(downloaded_path) != dest_path:
    shutil.move(downloaded_path, dest_path)

  print(f"    Done  ({dest_path.stat().st_size / 1024**3:.2f} GB)")


def collect_images_from_dir(src_dir):
  """Recursively find all image files under *src_dir*."""
  _EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
  images = []
  for path in sorted(src_dir.rglob("*")):
    if path.is_file() and path.suffix.lower() in _EXTS:
      images.append(path)
  return images


def main():
  parser = argparse.ArgumentParser(
      description="Prepare Derm1M dataset for pretraining",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=__doc__,
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Output directory for extracted images",
  )
  parser.add_argument(
      "--token",
      type=str,
      help="HuggingFace API token (or set HF_TOKEN env var)",
  )
  parser.add_argument(
      "--zip_dir",
      type=str,
      help="Directory to store zip archives. Existing zips are reused, "
      "missing ones are downloaded. If omitted, a temp directory is used.",
  )
  parser.add_argument(
      "--min_resolution",
      type=int,
      default=None,
      help="Skip images whose width or height is smaller than this value",
  )
  parser.add_argument(
      "--skip_download",
      action="store_true",
      help="Fail instead of downloading missing zips (use with --zip_dir "
      "to verify all archives are present before proceeding)",
  )
  args = parser.parse_args()

  token = args.token or os.environ.get("HF_TOKEN")
  output_path = Path(args.output_dir)

  # Check if output directory already has images
  image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
  if output_path.exists() and any(
      f.suffix.lower() in image_exts for f in output_path.rglob("*") if f.is_file()):
    print(
        f"Output directory {output_path} already contains images, skipping preparation."
    )
    print("Delete it and re-run to refresh, or use a different --output_dir.")
    return

  output_path.mkdir(parents=True, exist_ok=True)

  # Determine zip directory.
  if args.zip_dir:
    zip_dir = Path(args.zip_dir)
    zip_dir.mkdir(parents=True, exist_ok=True)
  else:
    zip_dir = Path(tempfile.mkdtemp(prefix="derm1m_"))

  # Fetch the list of zips dynamically from the HF repo
  zip_names = fetch_zip_names(token=token)
  print(f"Found {len(zip_names)} zip archives: {', '.join(zip_names)}")

  # Download zips
  for zip_name in zip_names:
    url = f"{BASE_URL}/{zip_name}"
    zip_path = zip_dir / zip_name

    if zip_path.exists():
      if args.skip_download:
        print(f"  {zip_name} exists, skipping "
              f"(use without --skip_download to re-download)")
        continue
      else:
        print(f"  {zip_name} exists, overwriting")

    download_file(url, zip_path, token)

  # Extract all zips
  for zip_name in zip_names:
    zip_path = zip_dir / zip_name
    print(f"  Extracting {zip_name} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
      zf.extractall(zip_dir)
    print("    Done")

  # Walk the extracted directories and copy images into the output folder.
  # Flatten everything into a single directory — ImageFolderDataset only
  # scans root_dir.iterdir(), not subdirectories.
  # All files are renamed to IMG_XXXXXXXX.ext (zero-padded sequential number)
  # to avoid unicode / special-character filenames from the zip archives.
  image_count = 0
  for root, _dirs, files in os.walk(zip_dir):
    root_path = Path(root)
    for fname in sorted(files):
      fpath = root_path / fname
      if fpath.suffix.lower() not in {
          ".jpg",
          ".jpeg",
          ".png",
          ".bmp",
          ".tiff",
          ".tif",
          ".webp",
          ".gif",
      }:
        continue
      # Optionally skip images smaller than --min_resolution.
      if args.min_resolution is not None:
        try:
          from PIL import Image
          with Image.open(fpath) as img:
            if img.width < args.min_resolution or img.height < args.min_resolution:
              continue
        except OSError as exc:
          logging.warning("Skipping unreadable image %s: %s", fpath, exc)
      image_count += 1
      ext = fpath.suffix.lower()
      dest = output_path / f"IMG_{image_count:08d}{ext}"
      shutil.move(str(fpath), str(dest))
      if image_count % 10000 == 0:
        print(f"  Moved {image_count:,} images ...")

  print(f"\nTotal images moved: {image_count:,}")

  # Clean up temp dir only if we created it (not user-specified).
  if not args.zip_dir:
    shutil.rmtree(zip_dir, ignore_errors=True)

  print("\n" + "=" * 60)
  print("DONE! Use the following command to pretrain:")
  print(f"  genml-kit-pretrain --datasets {args.output_dir} --image_size 448 ...")
  print("=" * 60)


if __name__ == "__main__":
  main()
