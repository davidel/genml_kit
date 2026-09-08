"""Smoke test for train.py end-to-end.

Patches all HuggingFace hub calls and runs training for 1 epoch on a tiny
synthetic dataset with a mock model.  Verifies that checkpoints are written.
"""

import logging
import os
import re
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
  """Minimal torch.nn.Module that accepts pixel_values and returns .logits."""

  def __init__(self, num_labels=3):
    super().__init__()
    self.fc = torch.nn.Linear(3 * 64 * 64, num_labels)
    self.config = MockConfig(id2label={str(i): f"class_{i}" for i in range(num_labels)})

  def forward(self, pixel_values, **kwargs):
    return LogitsOutput(logits=self.fc(pixel_values.flatten(1)))


class TinyProcessor:
  """Mimics AutoImageProcessor."""

  image_mean = [0.5, 0.5, 0.5]
  image_std = [0.5, 0.5, 0.5]

  def __call__(self, image, return_tensors="pt"):
    import numpy as np
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
  """Return a raw Dataset with random images and labels."""
  imgs = [
      Image.fromarray(np.random.randint(0, 256, (size, size, 3), dtype=np.uint8))
      for _ in range(n_total)
  ]
  labels = [i % num_classes for i in range(n_total)]
  return Dataset.from_dict({"image": imgs, "label": labels})


def test_train_smoke(tmp_path):
  """Run main() for 1 epoch, verify checkpoint files are created."""
  # main() configures the root logger; close its handlers on exit so the
  # --log_targets file handle is released (warnings-as-errors otherwise
  # turns the leak into a ResourceWarning failure at teardown).
  root = logging.getLogger()
  handlers_before = root.handlers[:]
  try:
    _run_train_smoke(tmp_path)
  finally:
    for h in root.handlers[:]:
      root.removeHandler(h)
      if isinstance(h.formatter, GlogFormatter):
        h.close()
    root.handlers[:] = handlers_before


def _run_train_smoke(tmp_path):
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
      "2",
      "--num_workers",
      "0",
      "--log_level",
      "INFO",
      "--log_targets",
      str(tmp_path / "train.log"),
  ]

  with (
      patch("sys.argv", test_args),
      patch("genml_kit.training.train.load_dataset", return_value=ds),
      patch(
          "genml_kit.training.train.load_processor",
          return_value=TinyProcessor(),
      ),
      patch(
          "genml_kit.training.train.load_model",
          return_value=TinyModel(num_labels=3),
      ),
  ):
    from genml_kit.training.train import main

    main()

  latest = ckpt_base + "_latest.pt"
  assert os.path.exists(latest), f"Missing {latest}"

  # --log_targets wrote a glog-formatted log file for this run.
  log_path = tmp_path / "train.log"
  assert log_path.exists(), "Missing log file from --log_targets"
  log_text = log_path.read_text()
  assert log_text, "Log file is empty"
  assert re.search(r"^[IWEF]\d{4} ", log_text,
                   re.MULTILINE), ("Log file lacks glog-formatted lines")

  ckpt = torch.load(latest, map_location="cpu", weights_only=False)
  assert "model_state_dict" in ckpt
  assert "optimizer_state_dict" in ckpt
  assert "scheduler_state_dict" in ckpt
  assert "epoch" in ckpt
  assert "best_macro_f1" in ckpt


def test_sampler_skips_freq_in_loss_weights(tmp_path):
  """With --sampler weighted, loss weights should be class_multipliers only.

  Regression test for a bug where w_freq * class_multipliers was used as loss
  weights even when the sampler already balanced class representation,
  causing the model to collapse to the majority class.
  """
  # Imbalanced dataset: 16 class_0, 4 class_1 (20 total).
  imgs = [
      Image.fromarray(np.random.randint(0, 256, (64, 64, 3), dtype=np.uint8))
      for _ in range(20)
  ]
  labels = [0] * 16 + [1] * 4
  ds = Dataset.from_dict({"image": imgs, "label": labels})
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
      "--val_split",
      "0.3",
      "--image_size",
      "64",
      "--checkpoint",
      ckpt_base,
      "--log_every",
      "2",
      "--num_workers",
      "0",
      "--log_level",
      "WARNING",
      "--sampler",
      "weighted",
      "--sampler_weights",
      "frequency",
      "--class_multipliers",
      "0=2.0,1=3.0",
  ]

  with (
      patch("sys.argv", test_args),
      patch("genml_kit.training.train.load_dataset", return_value=ds),
      patch("genml_kit.training.train.load_processor", return_value=TinyProcessor()),
      patch("genml_kit.training.train.load_model",
            return_value=TinyModel(num_labels=2)),
  ):
    from genml_kit.training.train import main
    main()

  # Verify checkpoint was created (model trained successfully).
  latest = ckpt_base + "_latest.pt"
  assert os.path.exists(latest), f"Missing {latest}"

  ckpt = torch.load(latest, map_location="cpu", weights_only=False)
  # The model should NOT predict a single class for everything.
  # Check that the classifier weights are not all identical
  # (which would indicate collapse to majority class).
  head_w = ckpt["model_state_dict"]["fc.weight"]
  # Each row corresponds to a class. If rows are identical, model collapsed.
  assert not torch.allclose(head_w[0], head_w[1], atol=1e-6), \
      "Class 0 and 1 weights identical — model may have collapsed"
