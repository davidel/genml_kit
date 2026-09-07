"""Smoke test for pretrain.py end-to-end.

Registers a tiny self-supervised method and runs ``main()`` for 1 epoch on
a small on-disk imagefolder fixture (no HuggingFace hub calls).  Verifies
that checkpoints are written, that the method-specific state round-trips
through the saver's ``extra_fn``, and that a resumed run restores it.
"""

import logging
import os
from unittest.mock import patch

import torch
import torchvision.transforms.functional as TF
from PIL import Image

from scdiag.logging_utils import GlogFormatter
from scdiag.pretrain_methods.base import PretrainMethod
from scdiag.pretrain_methods.registry import register_method

TINY_MARKER = "tiny-smoke-state"


class TinyBackbone(torch.nn.Module):
  """Minimal conv net that trains fast on CPU."""

  def __init__(self):
    super().__init__()
    self.conv = torch.nn.Conv2d(3, 4, kernel_size=3, stride=2, padding=1)
    self.fc = torch.nn.Linear(4 * 16 * 16, 8)

  def forward(self, x):
    return self.fc(self.conv(x).flatten(1))


@register_method
class TinyPretrainMethod(PretrainMethod):
  """Reconstruction-free stub method exercising the generic harness."""

  NAME = "tiny_smoke"
  needs_labels = False

  def add_args(self, parser):
    pass

  def build(self, args, base_model, device):
    return base_model

  def train_step(self, model, images, global_step, labels=None):
    out = model(images)
    loss = out.square().mean()
    return loss, {}

  def get_checkpoint_state(self, model, args):
    return {"tiny_marker": TINY_MARKER}

  def load_checkpoint_state(self, model, state, args):
    model.loaded_tiny_marker = state.get("tiny_marker")

  def build_transform(self, image_size):
    size = (image_size, image_size)

    def _transform(image):
      image = TF.resize(image, size)
      return TF.to_tensor(image)

    return _transform


def _make_imagefolder(root, n_per_class=2, size=32):
  """Create a depth-2 imagefolder (root/<class>/<file>.jpg) fixture."""
  for cls in ("a", "b"):
    cls_dir = root / cls
    cls_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_per_class):
      img = Image.fromarray(
          torch.randint(0, 256, (size, size, 3), dtype=torch.uint8).numpy())
      img.save(cls_dir / f"{cls}_{i}.jpg")


def test_pretrain_smoke(tmp_path):
  """Run main() for 1 epoch, verify checkpoint contents."""
  # main() configures the root logger; close its handlers on exit so the
  # --log_targets file handle is released (warnings-as-errors otherwise
  # turns the leak into a ResourceWarning failure at teardown).
  root = logging.getLogger()
  handlers_before = root.handlers[:]
  try:
    _run_pretrain_smoke(tmp_path)
  finally:
    for h in root.handlers[:]:
      root.removeHandler(h)
      if isinstance(h.formatter, GlogFormatter):
        h.close()
    root.handlers[:] = handlers_before


def _run_pretrain_smoke(tmp_path):
  images_dir = tmp_path / "images"
  _make_imagefolder(images_dir)
  ckpt_base = str(tmp_path / "ckpts" / "model")

  test_args = [
      "pretrain.py",
      "--method",
      "tiny_smoke",
      "--datasets",
      str(images_dir),
      "--model",
      "dummy/tiny-backbone",
      "--epochs",
      "1",
      "--batch_size",
      "2",
      "--image_size",
      "32",
      "--checkpoint",
      ckpt_base,
      "--lr",
      "1e-3",
      "--log_every",
      "1",
      "--num_workers",
      "0",
      "--log_level",
      "INFO",
      "--log_targets",
      str(tmp_path / "pretrain.log"),
  ]

  with (
      patch("sys.argv", test_args),
      patch("scdiag.pretrain.load_model", return_value=TinyBackbone()),
  ):
    from scdiag.pretrain import main

    main()

  latest = ckpt_base + "_latest.pt"
  assert os.path.exists(latest), f"Missing {latest}"

  ckpt = torch.load(latest, map_location="cpu", weights_only=False)
  assert ckpt["epoch"] == 0
  assert ckpt["optimizer_state_dict"] is not None
  # The saver's extra_fn must have merged the method-specific state.
  assert ckpt.get("method_state") == {"tiny_marker": TINY_MARKER}
  assert "model_state_dict" in ckpt


def test_pretrain_resume_smoke(tmp_path):
  """Run main() twice; the second run must resume and restore method state."""
  root = logging.getLogger()
  handlers_before = root.handlers[:]
  try:
    _run_pretrain_smoke(tmp_path)
    _run_pretrain_smoke(tmp_path, resume=True)
  finally:
    for h in root.handlers[:]:
      root.removeHandler(h)
      if isinstance(h.formatter, GlogFormatter):
        h.close()
    root.handlers[:] = handlers_before


def _run_pretrain_smoke(tmp_path, resume=False):
  images_dir = tmp_path / "images"
  _make_imagefolder(images_dir)
  ckpt_base = str(tmp_path / "ckpts" / "model")

  test_args = [
      "pretrain.py",
      "--method",
      "tiny_smoke",
      "--datasets",
      str(images_dir),
      "--model",
      "dummy/tiny-backbone",
      "--epochs",
      "1",
      "--batch_size",
      "2",
      "--image_size",
      "32",
      "--checkpoint",
      ckpt_base,
      "--lr",
      "1e-3",
      "--log_every",
      "1",
      "--num_workers",
      "0",
      "--log_level",
      "INFO",
      "--log_targets",
      str(tmp_path / "pretrain.log"),
  ]
  if resume:
    test_args += ["--resume"]

  # create_model_report() is called with the model right before the loop
  # starts; use it to capture the (mutated) model for assertions.
  captured = {}

  def _report_spy(model):
    captured["model"] = model
    return "report"

  with (
      patch("sys.argv", test_args),
      patch("scdiag.pretrain.load_model", return_value=TinyBackbone()),
      patch("scdiag.pretrain.create_model_report", side_effect=_report_spy),
  ):
    from scdiag.pretrain import main

    main()

  if resume:
    # open_resume_context() must have loaded _latest.pt and the method
    # must have received its state via load_checkpoint_state().
    model = captured["model"]
    assert getattr(model, "loaded_tiny_marker", None) == TINY_MARKER
    ckpt = torch.load(ckpt_base + "_latest.pt", map_location="cpu", weights_only=False)
    # 1 epoch was already completed in the first run, so the second run
    # (also capped at --epochs 1) trains nothing and re-saves epoch 0.
    assert ckpt["epoch"] == 0
