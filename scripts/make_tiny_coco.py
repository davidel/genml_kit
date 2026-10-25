#!/usr/bin/env python3
"""Write a small COCO-JSON detection corpus for CPU smoke runs.

The generated corpus follows the standard COCO object-detection schema
(``images``, ``annotations`` with ``[x, y, width, height]`` ``bbox``, and
``categories``) and can be fed straight to the ``detection`` pipeline:

  python scripts/make_tiny_coco.py tiny_coco -n 200

Each image is drawn from a small palette of class colours and carries a
configurable number of axis-aligned, non-overlapping-ish boxes, so a model
has a real signal to fit during a short CPU overfit run.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image


def build_parser():
  """Build the command-line parser."""
  parser = argparse.ArgumentParser(
      description="Generate a tiny COCO-JSON detection corpus.")
  parser.add_argument(
      "root",
      nargs="?",
      default="tiny_coco",
      help="Output directory (default: %(default)s).",
  )
  parser.add_argument(
      "-n",
      "--num_images",
      type=int,
      default=12,
      help="Number of images to generate (default: %(default)s).",
  )
  parser.add_argument(
      "--size",
      type=int,
      default=128,
      help="Image side length in pixels, square images (default: %(default)s).",
  )
  parser.add_argument(
      "--num_classes",
      type=int,
      default=2,
      help="Number of object categories (default: %(default)s).",
  )
  parser.add_argument(
      "--boxes_per_image",
      type=int,
      default=2,
      help="Boxes drawn on each image (default: %(default)s).",
  )
  parser.add_argument(
      "--seed",
      type=int,
      default=0,
      help="RNG seed for reproducibility (default: %(default)s).",
  )
  return parser


def _class_color(class_index, num_classes):
  """Return an RGB tuple that is distinct per class index."""
  # Spread hues around the circle so classes stay visually separable.
  hue = class_index / max(num_classes, 1)
  return (
      int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * hue))),
      int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (hue + 1 / 3)))),
      int(255 * (0.5 + 0.5 * np.cos(2 * np.pi * (hue + 2 / 3)))),
  )


def _draw_box(arr, rng, size, class_index, num_classes):
  """Paint one coloured rectangle and return its COCO bbox + category id."""
  # Keep the box inside the frame with an 8 px margin on every side.
  x = int(rng.integers(8, max(size // 2, 9)))
  y = int(rng.integers(8, max(size // 2, 9)))
  w = int(rng.integers(24, max(size // 2, 25)))
  h = int(rng.integers(24, max(size // 2, 25)))
  x2, y2 = min(x + w, size), min(y + h, size)
  arr[y:y2, x:x2] = _class_color(class_index, num_classes)
  return (x, y, x2 - x, y2 - y, class_index + 1)


def main(argv=None):
  """Generate the corpus described by the command line."""
  args = build_parser().parse_args(argv)

  root = Path(args.root)
  (root / "images").mkdir(parents=True, exist_ok=True)

  rng = np.random.default_rng(args.seed)
  images, anns = [], []
  ann_id = 1
  for i in range(args.num_images):
    arr = np.zeros((args.size, args.size, 3), dtype=np.uint8)
    boxes = [
        _draw_box(arr, rng, args.size, int(rng.integers(0, args.num_classes)),
                  args.num_classes) for _ in range(args.boxes_per_image)
    ]
    name = f"img_{i:05d}.png"
    Image.fromarray(arr).save(root / "images" / name)
    images.append({
        "id": i + 1,
        "file_name": name,
        "width": args.size,
        "height": args.size,
    })
    for x, y, w, h, cid in boxes:
      anns.append({
          "id": ann_id,
          "image_id": i + 1,
          "category_id": cid,
          "bbox": [x, y, w, h],
          "area": w * h,
          "iscrowd": 0,
      })
      ann_id += 1

  coco = {
      "images":
          images,
      "annotations":
          anns,
      "categories": [{
          "id": c + 1,
          "name": f"thing_{c}"
      } for c in range(args.num_classes)],
  }
  (root / "annotations.json").write_text(json.dumps(coco))
  print(f"wrote {args.num_images} images, {len(anns)} boxes, "
        f"{args.num_classes} classes -> {root}")


if __name__ == "__main__":
  main()
