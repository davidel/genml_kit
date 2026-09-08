#!/usr/bin/env python
"""Prepare HAM10000 dataset with lesion_id-grouped splits.

Downloads the marmal88/skin_cancer dataset from HuggingFace, and creates
train/val/test splits grouped by lesion_id to prevent data leakage
(same lesion never appears in multiple splits).

Output is an ImageFolder structure compatible with:
    --dataset imagefolder/OUTPUT_PATH

Usage:
    python scripts/prepare_ham10000.py --output_dir ./ham10000_grouped
    python scripts/prepare_ham10000.py --output_dir ./ham10000_grouped \
        --test_size 0.15 --val_size 0
    python scripts/prepare_ham10000.py --output_dir ./ham10000_grouped \
        --min_resolution 224
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

from datasets import load_dataset


def group_split_by_lesion_id(
    dataset,
    test_size=0.1,
    val_size=0.1,
    seed=42,
    label_col="dx",
):
  """Split dataset by lesion_id groups to prevent data leakage.

    Args:
        dataset: HuggingFace dataset with 'image', 'lesion_id', and label columns
        test_size: Fraction of lesion_ids for test set
        val_size: Fraction of lesion_ids for validation set
        seed: Random seed for reproducibility
        label_col: Name of the label column (default: 'dx')

    Returns:
        Dict with 'train', 'val', 'test' keys containing lists of (image, label) tuples
    """
  random.seed(seed)

  # Group dataset indices by lesion_id.
  # Use the column-level access to avoid loading full images into memory.
  lesion_ids = dataset["lesion_id"]
  lesion_to_indices = defaultdict(list)

  for idx, lesion_id in enumerate(lesion_ids):
    lesion_to_indices[lesion_id].append(idx)

  # Split lesion_ids into train/val/test
  all_lesion_ids = list(lesion_to_indices.keys())
  random.shuffle(all_lesion_ids)

  n_test = int(len(all_lesion_ids) * test_size)
  n_val = int(len(all_lesion_ids) * val_size)

  test_lesion_ids = set(all_lesion_ids[:n_test])
  val_lesion_ids = (set(all_lesion_ids[n_test:n_test + n_val]) if n_val > 0 else set())
  train_lesion_ids = set(all_lesion_ids[n_test + n_val:])

  # Collect indices for each split
  train_indices = []
  val_indices = []
  test_indices = []

  for lesion_id, indices in lesion_to_indices.items():
    if lesion_id in test_lesion_ids:
      test_indices.extend(indices)
    elif lesion_id in val_lesion_ids:
      val_indices.extend(indices)
    else:
      train_indices.extend(indices)

  # Extract examples for each split
  def get_examples(indices):
    examples = []
    for idx in indices:
      example = dataset[idx]
      examples.append((example["image"], example[label_col]))
    return examples

  train_examples = get_examples(train_indices)
  val_examples = get_examples(val_indices)
  test_examples = get_examples(test_indices)

  print("\nSplit by lesion_id:")
  if n_val > 0:
    print(f"  Lesion IDs: {len(train_lesion_ids)} train, "
          f"{len(val_lesion_ids)} val, {len(test_lesion_ids)} test")
    print(f"  Images: {len(train_examples)} train, "
          f"{len(val_examples)} val, {len(test_examples)} test")
  else:
    print(f"  Lesion IDs: {len(train_lesion_ids)} train, "
          f"{len(test_lesion_ids)} test (no val)")
    print(f"  Images: {len(train_examples)} train, "
          f"{len(test_examples)} test (no val)")

  return {
      "train": train_examples,
      "val": val_examples,
      "test": test_examples,
  }


def save_as_imagefolder(
    splits,
    output_dir,
    label_names=None,
    min_resolution=None,
):
  """Save splits as ImageFolder structure.

    Args:
        splits: Dict with 'train', 'val', 'test' keys holding (image, label)
            tuples.
        output_dir: Output directory path.
        label_names: Optional dict mapping label_id -> class_name.
            If None, uses numeric names.
        min_resolution: If set, skip images whose width or height is
            smaller than this value.
    """
  output_path = Path(output_dir)

  for split_name, examples in splits.items():
    # Skip empty splits (e.g., val when --val_size=0)
    if len(examples) == 0:
      print(f"\nSkipping {split_name} split (empty)")
      continue

    print(f"\nSaving {split_name} split with {len(examples)} images...")

    # Group by label
    label_to_images = defaultdict(list)
    for image, label in examples:
      label_to_images[label].append(image)

    # Save images
    for label, images in sorted(label_to_images.items()):
      class_name = label_names[label] if label_names else str(label)

      class_dir = output_path / split_name / class_name
      class_dir.mkdir(parents=True, exist_ok=True)

      saved = 0
      for image in images:
        if (min_resolution is not None and
            (image.width < min_resolution or image.height < min_resolution)):
          continue
        dst_path = class_dir / f"{saved:05d}.jpg"
        # Save PIL Image as JPEG
        image.save(dst_path, "JPEG", quality=95)
        saved += 1

      print(f"  {class_name}: {saved}/{len(images)} images")

  print(f"\nSaved ImageFolder dataset to {output_dir}")


def main():
  parser = argparse.ArgumentParser(
      description="Prepare HAM10000 dataset with lesion_id-grouped splits",
      formatter_class=argparse.RawDescriptionHelpFormatter,
      epilog=__doc__,
  )
  parser.add_argument(
      "--dataset",
      type=str,
      default="marmal88/skin_cancer",
      help="HuggingFace dataset name (default: %(default)s)",
  )
  parser.add_argument(
      "--output_dir",
      type=str,
      required=True,
      help="Output directory for ImageFolder dataset",
  )
  parser.add_argument(
      "--test_size",
      type=float,
      default=0.1,
      help="Fraction of lesion_ids for test set (default: %(default)s)",
  )
  parser.add_argument(
      "--val_size",
      type=float,
      default=0.1,
      help="Fraction of lesion_ids for validation set (default: %(default)s)",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=42,
      help="Random seed for reproducibility (default: %(default)s)",
  )
  parser.add_argument(
      "--min_resolution",
      type=int,
      default=None,
      help="Skip images whose width or height is smaller than this value",
  )
  parser.add_argument(
      "--cache_dir",
      type=str,
      help=
      "HuggingFace datasets cache directory (default: ~/.cache/huggingface/datasets)",
  )

  args = parser.parse_args()

  # Validate sizes
  if args.test_size + args.val_size >= 1.0:
    parser.error("test_size + val_size must be < 1.0")

  # Load dataset
  print(f"Loading dataset: {args.dataset}")
  raw = load_dataset(args.dataset, cache_dir=args.cache_dir)

  # Merge all splits into a single dataset for re-splitting by lesion_id
  if isinstance(raw, dict):
    # DatasetDict: concatenate all splits
    from datasets import concatenate_datasets

    splits = list(raw.keys())
    print(f"Dataset has {len(splits)} splits: {splits}")
    dataset = concatenate_datasets([raw[split] for split in splits])
    print("Merged into single dataset")
  else:
    # Single Dataset
    dataset = raw

  print(f"Dataset loaded: {len(dataset)} images")
  print(f"Columns: {list(dataset.features.keys())}")

  # Detect label column (HAM10000 uses 'dx', others may use 'label')
  label_col = "label" if "label" in dataset.features else "dx"
  print(f"Using '{label_col}' as label column")

  # Get label names
  label_names = None
  if hasattr(dataset.features[label_col], "names"):
    label_names = {i: name for i, name in enumerate(dataset.features[label_col].names)}
    print(f"Labels: {label_names}")

  # Check for lesion_id column
  if "lesion_id" not in dataset.features:
    print("Error: Dataset does not have 'lesion_id' column")
    print("Available columns:", list(dataset.features.keys()))
    return

  # Split by lesion_id
  splits = group_split_by_lesion_id(
      dataset,
      test_size=args.test_size,
      val_size=args.val_size,
      seed=args.seed,
      label_col=label_col,
  )

  # Save as ImageFolder
  save_as_imagefolder(splits,
                      args.output_dir,
                      label_names,
                      min_resolution=args.min_resolution)

  print("\n" + "=" * 60)
  print("DONE! Use the following command to train with grouped splits:")
  print(f"  python -m genml_kit.training.train --dataset imagefolder/{args.output_dir}")
  print("=" * 60)


if __name__ == "__main__":
  main()
