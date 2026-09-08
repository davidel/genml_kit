#!/usr/bin/env python
"""Prepare ISIC archive images for pretraining.

Downloads images from the ISIC (International Skin Imaging Collaboration)
archive using the isic-cli tool, and stores them in a target folder
compatible with genml-kit-pretrain.

Usage:
    python scripts/prepare_isic.py --output_dir ./isic_images

    # Download with a specific query (e.g., only melanoma cases)
    python scripts/prepare_isic.py --output_dir ./isic_melanoma \
        --search 'diagnosis:"melanoma"'

    # Filter out small images during download
    python scripts/prepare_isic.py --output_dir ./isic_images \
        --min_resolution 224

    # Download the SIIM-ISIC 2020 Challenge subset
    python scripts/prepare_isic.py --output_dir ./isic_2020 \
        --search "anonymous:true"

    # Download the entire archive (may take several days)
    python scripts/prepare_isic.py --output_dir ./isic_full

Requirements:
    pip install isic-cli

Note:
    The ISIC archive contains dermatoscopic images of skin lesions.
    A full download can be very large (hundreds of GB) and may take
    several days. It is recommended to start with a specific query
    to benchmark your ConvViT pipeline first.
"""

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path


def download_isic_archive(target_dir, search_query=None, verbose=False, max_retries=3):
  """Invoke the isic-cli tool via Python to download images and metadata.

  Args:
      target_dir: Relative or absolute path where images will be saved.
      search_query: Lucene query string to filter downloads.
                    Leave None to attempt a full archive download.
      verbose: If True, print detailed progress information.
      max_retries: Number of retry attempts for failed downloads.
  """
  target_path = Path(target_dir)
  target_path.mkdir(parents=True, exist_ok=True)

  # Check if the isic executable exists on the system path
  isic_executable = shutil.which("isic")
  if not isic_executable:
    print(
        "Error: 'isic-cli' is not installed or not found in system PATH.",
        file=sys.stderr,
    )
    print("Please run: pip install isic-cli", file=sys.stderr)
    raise RuntimeError("'isic' executable not found")

  # Build the isic command
  # 'isic image download <folder>' automatically pulls images and metadata CSVs
  cmd = [isic_executable, "image", "download"]

  if search_query:
    cmd.extend(["--search", search_query])

  cmd.append(str(target_path))

  if verbose:
    print(f"Running command: {' '.join(cmd)}")

  for attempt in range(1, max_retries + 1):
    try:
      if verbose:
        print(f"\nAttempt {attempt}/{max_retries}")

      # Run the command and stream output
      process = subprocess.run(
          cmd,
          check=True,
          capture_output=not verbose,
          text=True,
      )

      if verbose and process.stdout:
        print(process.stdout)

      print(f"Successfully downloaded images to {target_path}")
      return

    except subprocess.CalledProcessError as e:
      print(f"Attempt {attempt} failed with return code {e.returncode}")
      if e.stderr and not verbose:
        print(f"Error output: {e.stderr}")
      if attempt == max_retries:
        raise RuntimeError(f"Failed to download after {max_retries} attempts") from e
      print("Retrying in 5 seconds...")
      time.sleep(5)


def main():
  parser = argparse.ArgumentParser(
      description="Prepare ISIC archive images for pretraining",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=__doc__,
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Output directory for downloaded images",
  )
  parser.add_argument(
      "--search",
      type=str,
      default=None,
      help="Lucene query string to filter downloads "
      "(e.g., 'diagnosis:\"melanoma\"' or 'anonymous:true'). "
      "Leave empty to download the entire archive.",
  )
  parser.add_argument(
      "--verbose",
      action="store_true",
      help="Print detailed progress information",
  )
  parser.add_argument(
      "--min_resolution",
      type=int,
      default=None,
      help=
      "After download, remove images whose width or height is smaller than this value",
  )
  parser.add_argument(
      "--max_retries",
      type=int,
      default=3,
      help="Number of retry attempts for failed downloads (default: 3)",
  )
  args = parser.parse_args()

  # Check if output directory already has images
  image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp", ".gif"}
  output_path = Path(args.output_dir)
  if output_path.exists() and any(
      f.suffix.lower() in image_exts for f in output_path.rglob("*") if f.is_file()):
    print(f"Output directory {output_path} already contains images.")
    print("Delete it and re-run to refresh, or use a different --output_dir.")
    return

  print("=" * 60)
  print("ISIC Archive Downloader")
  print("=" * 60)
  print(f"Output directory: {output_path}")
  if args.search:
    print(f"Search query: {args.search}")
  else:
    print("Search query: None (downloading entire archive)")
  print(f"Max retries: {args.max_retries}")
  print("=" * 60)

  try:
    download_isic_archive(
        target_dir=args.output_dir,
        search_query=args.search,
        verbose=args.verbose,
        max_retries=args.max_retries,
    )

    # Post-download resolution filtering.
    if args.min_resolution is not None:
      from PIL import Image
      removed = 0
      kept = 0
      for fpath in sorted(output_path.rglob("*")):
        if not fpath.is_file():
          continue
        if fpath.suffix.lower() not in image_exts:
          continue
        try:
          with Image.open(fpath) as img:
            if (img.width < args.min_resolution or img.height < args.min_resolution):
              fpath.unlink()
              removed += 1
              continue
        except OSError as exc:
          logging.warning("Skipping unreadable image %s: %s", fpath, exc)
        kept += 1
      print(f"Resolution filter: kept {kept}, removed {removed} "
            f"(min_resolution={args.min_resolution})")

    print("\n" + "=" * 60)
    print("DONE! Use the following command to pretrain:")
    print(f"  genml-kit-pretrain --datasets {args.output_dir} --image_size 448 ...")
    print("=" * 60)

  except RuntimeError as e:
    print(f"\nError: {e}", file=sys.stderr)
    sys.exit(1)
  except KeyboardInterrupt:
    print("\nDownload interrupted by user.")
    sys.exit(1)


if __name__ == "__main__":
  main()
