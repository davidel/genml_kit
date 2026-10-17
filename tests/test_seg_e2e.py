"""End-to-end segmentation test: pipeline -> method -> loss -> eval -> registry."""

import argparse

import numpy as np
import pytest
import torch
from PIL import Image

from genml_kit.methods import build_method
from genml_kit.pipelines import build_pipeline


def _write_dataset(root, n=8, size=32, num_classes=3, seed=0):
  rng = np.random.default_rng(seed)
  (root / "images").mkdir(parents=True, exist_ok=True)
  (root / "masks").mkdir(parents=True, exist_ok=True)
  for i in range(n):
    Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(
        root / "images" / f"img_{i:03d}.png")
    mask = rng.integers(0, num_classes, (size, size), dtype=np.uint8)
    Image.fromarray(mask).save(root / "masks" / f"img_{i:03d}.png")


def _args(root, **overrides):
  base = dict(
      pipeline="segmentation",
      method="segmentation",
      dataset=f"segfolder/{root}",
      mask_column=None,
      image_column="image",
      val_split=0.25,
      cache_dir=None,
      model="seg_wrapper:timm/resnet18",
      model_arg={
          "decoder": "unet:depth=4",
          "pretrained": False
      },
      image_size=32,
      batch_size=4,
      num_workers=0,
      seed=0,
      device="cpu",
      seg_num_classes=None,
      seg_ignore_index=-1,
      seg_augmentation_script=None,
      loss=None,
      loss_args=None,
      freeze=None,
      lora=False,
      grad_checkpoint=False,
      source_checkpoint=None,
      param_init=None,
  )
  base.update(overrides)
  return argparse.Namespace(**base)


def _build(root, **overrides):
  args = _args(root, **overrides)
  pipeline = build_pipeline(args.pipeline)
  method = build_method(args.method)
  method.prepare_transforms(args, args.device)
  pipeline.build_loader(args, mode="train", method=method)
  pipeline.build_loader(args, mode="val", method=method)
  method.wire_data(args, pipeline)
  model = method.build_model(args, args.device)
  return args, pipeline, method, model


def test_registry_knows_seg_wrapper():
  from genml_kit.models.registry import MODELS
  assert MODELS.contains("seg_wrapper")


def test_pipeline_and_method_contract(tmp_path):
  _write_dataset(tmp_path)
  args, pipeline, method, model = _build(tmp_path)

  assert method._num_classes == 3
  blob = next(iter(pipeline.train_loader))
  blob = pipeline.to_device(blob, args.device)
  # Image is (B, C, H, W); the mask is a (B, H, W) long tensor in meta.
  assert blob.data.dim() == 4
  assert blob.meta["mask"].dim() == 3
  assert blob.meta["mask"].dtype == torch.long

  out = method.train_step(model, blob, 0)
  assert torch.isfinite(out.loss)

  metrics = method.evaluate(model, pipeline.val_loader, args.device, pipeline.to_device)
  for key in ("miou", "mdice", "pixel_acc", "loss"):
    assert key in metrics
    assert 0.0 <= metrics[key] <= 1.0 or key == "loss"


def test_num_classes_inferred_excludes_ignore(tmp_path):
  # Masks contain a 255 void label; the inferred class count must ignore it.
  rng = np.random.default_rng(0)
  (tmp_path / "images").mkdir()
  (tmp_path / "masks").mkdir()
  for i in range(6):
    Image.fromarray(rng.integers(0, 255, (16, 16, 3),
                                 dtype=np.uint8)).save(tmp_path / "images" / f"{i}.png")
    mask = rng.integers(0, 3, (16, 16), dtype=np.uint8)
    mask[:4, :] = 255
    Image.fromarray(mask).save(tmp_path / "masks" / f"{i}.png")
  args, pipeline, method, model = _build(tmp_path, image_size=16, seg_ignore_index=255)
  assert method._num_classes == 3


def test_explicit_num_classes_override(tmp_path):
  _write_dataset(tmp_path)
  args, pipeline, method, model = _build(tmp_path, seg_num_classes=5)
  assert method._num_classes == 5
  blob = next(iter(pipeline.train_loader))
  out = method.train_step(model, pipeline.to_device(blob, args.device), 0)
  assert torch.isfinite(out.loss)


def test_checkpoint_state_roundtrip(tmp_path):
  _write_dataset(tmp_path)
  args, pipeline, method, model = _build(tmp_path)
  state = method.get_checkpoint_state(model, args)
  assert state["num_classes"] == 3

  fresh = build_method("segmentation")
  fresh.load_checkpoint_state(model, state, args)
  assert fresh._num_classes == 3


def test_training_step_decreases_loss(tmp_path):
  _write_dataset(tmp_path, n=8, seed=1)
  args, pipeline, method, model = _build(tmp_path)
  blob = pipeline.to_device(next(iter(pipeline.train_loader)), args.device)
  opt = torch.optim.SGD(model.parameters(), lr=0.1)
  before = float(method.train_step(model, blob, 0).loss.detach())
  for _ in range(8):
    out = method.train_step(model, blob, 0)
    opt.zero_grad()
    out.loss.backward()
    opt.step()
  after = float(method.train_step(model, blob, 0).loss.detach())
  assert after < before


def test_freeze_keeps_encoder_params_fixed(tmp_path):
  """``--freeze`` lists the *trainable* patterns (repo semantics).

  So ``--freeze 'decoder.*'`` freezes the encoder and keeps the decoder
  trainable -- the CNN-encoder adaptation recipe.
  """
  _write_dataset(tmp_path)
  args, pipeline, method, model = _build(tmp_path, freeze="decoder.*")
  encoder_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
  decoder_params = [p for n, p in model.named_parameters() if n.startswith("decoder.")]
  assert encoder_params and decoder_params
  assert all(not p.requires_grad for p in encoder_params)
  assert all(p.requires_grad for p in decoder_params)


@pytest.mark.slow
def test_segmentation_learns_a_simple_pattern(tmp_path):
  """Slow smoke test: the model overfits a trivial foreground/background split."""
  size = 32
  rng = np.random.default_rng(0)
  (tmp_path / "images").mkdir()
  (tmp_path / "masks").mkdir()
  for i in range(16):
    img = np.zeros((size, size, 3), dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    # Left half bright/foreground, right half dark/background.
    img[:, :size // 2] = 255
    mask[:, :size // 2] = 1
    img += rng.integers(0, 20, img.shape, dtype=np.uint8)
    Image.fromarray(img).save(tmp_path / "images" / f"{i}.png")
    Image.fromarray(mask).save(tmp_path / "masks" / f"{i}.png")

  args, pipeline, method, model = _build(tmp_path, image_size=size)
  blob = pipeline.to_device(next(iter(pipeline.train_loader)), args.device)
  opt = torch.optim.Adam(model.parameters(), lr=1e-3)
  for _ in range(30):
    out = method.train_step(model, blob, 0)
    opt.zero_grad()
    out.loss.backward()
    opt.step()
  metrics = method.evaluate(model, pipeline.val_loader, args.device, pipeline.to_device)
  assert metrics["miou"] > 0.3
