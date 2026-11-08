"""End-to-end multi-process smoke driver (run under torchrun).

Runs the real ``genml_kit.cli.train.main()`` on a tiny synthetic dataset with
a mock model, 2 ranks, gloo/CPU.  Writes a per-rank marker file with the
observed world size, rank and shard size so the parent test can assert that
the distributed path actually engaged (and sharded the data).

Invoked by ``tests/test_distributed_e2e.py`` via
``torchrun --nproc_per_node=2``.
"""
import contextlib
import logging
import os
import pathlib
import sys
from collections import namedtuple
from unittest.mock import patch

import numpy as np
import torch
from datasets import Dataset
from PIL import Image

from genml_kit.utils.logging import GlogFormatter

LogitsOutput = namedtuple("LogitsOutput", ["logits"])
MockConfig = namedtuple("MockConfig", ["id2label"])


class TinyModel(torch.nn.Module):

  def __init__(self, num_labels=3):
    super().__init__()
    self.fc = torch.nn.Linear(3 * 64 * 64, num_labels)
    self.config = MockConfig(id2label={str(i): f"class_{i}" for i in range(num_labels)})

  def forward(self, pixel_values, **kwargs):
    return LogitsOutput(logits=self.fc(pixel_values.flatten(1)))


class TinyProcessor:

  image_mean = [0.5, 0.5, 0.5]
  image_std = [0.5, 0.5, 0.5]

  def __call__(self, image, return_tensors="pt"):
    import torchvision.transforms.functional as F
    from PIL import Image as PILImage

    if isinstance(image, torch.Tensor):
      image = PILImage.fromarray(
          (image.permute(1, 2, 0).numpy() * 255).astype(np.uint8))
    image = F.resize(image, [64, 64])
    image = F.to_tensor(image)
    image = F.normalize(image, mean=self.image_mean, std=self.image_std)
    return {"pixel_values": image}


def _make_dataset(n_total=12, num_classes=3, size=64):
  imgs = [
      Image.fromarray(np.random.randint(0, 256, (size, size, 3), dtype=np.uint8))
      for _ in range(n_total)
  ]
  labels = [i % num_classes for i in range(n_total)]
  return Dataset.from_dict({"image": imgs, "label": labels})


def _run(tmp_path):
  ds = _make_dataset()
  ckpt_base = str(tmp_path / "ckpts" / "model")
  test_args = [
      "train.py",
      "--model",
      "dummy/tiny-model",
      "--dataset",
      "dummy/dataset",
      "--epochs",
      "1",
      "--batch_size",
      "4",
      "--image_size",
      "64",
      "--checkpoint",
      ckpt_base,
      "--log_every",
      "1",
      "--num_workers",
      "0",
      "--log_level",
      "INFO",
      "--dist_arg",
      "shard=true",
  ]
  with (
      patch("sys.argv", test_args),
      patch("datasets.load_dataset", return_value=ds),
      patch("genml_kit.models.load_processor", return_value=TinyProcessor()),
      patch("genml_kit.models.load_model", return_value=TinyModel(num_labels=3)),
  ):
    from genml_kit.cli.train import main
    main()

  # Read the rank/world from the launcher env: the process group is torn down
  # by main()'s finally, so dist.is_initialized() would be False here.
  rank = int(os.environ.get("RANK", "0"))
  world = int(os.environ.get("WORLD_SIZE", "1"))
  (tmp_path / f"rank{rank}.marker").write_text(f"rank={rank} world={world}")
  return ckpt_base


def main():
  root = logging.getLogger()
  handlers_before = root.handlers[:]
  tmp_path = None
  try:
    tmp_path = __import__("pathlib").Path(os.environ["DIST_E2E_TMPDIR"])
    _run(tmp_path)
  except Exception:
    import traceback
    tb = traceback.format_exc()
    with contextlib.suppress(Exception):
      (pathlib.Path(os.environ["DIST_E2E_TMPDIR"]) /
       f"error.rank{os.environ.get('RANK', '?')}.log").write_text(tb)
    sys.stderr.write(tb)
    raise
  finally:
    for h in root.handlers[:]:
      root.removeHandler(h)
      if isinstance(h.formatter, GlogFormatter):
        h.close()
    root.handlers[:] = handlers_before


if __name__ == "__main__":
  main()
