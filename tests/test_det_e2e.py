"""End-to-end detection test: pipeline -> method -> loss -> eval -> registry."""

import argparse
import json

import numpy as np
import pytest
import torch
from PIL import Image

from genml_kit.methods import build_method
from genml_kit.pipelines import build_pipeline


def _write_dataset(root, n=8, size=64, num_classes=2, boxes_per_image=2, seed=0):
  """Write a tiny COCO detection corpus with axis-aligned boxes."""
  rng = np.random.default_rng(seed)
  (root / "images").mkdir(parents=True, exist_ok=True)
  images = []
  annotations = []
  ann_id = 1
  for i in range(n):
    Image.fromarray(rng.integers(0, 255, (size, size, 3), dtype=np.uint8)).save(
        root / "images" / f"img_{i:03d}.png")
    images.append({
        "id": i + 1,
        "file_name": f"img_{i:03d}.png",
        "width": size,
        "height": size
    })
    for _ in range(boxes_per_image):
      w = int(size // 4)
      h = int(size // 4)
      x = int(rng.integers(0, size - w))
      y = int(rng.integers(0, size - h))
      annotations.append({
          "id": ann_id,
          "image_id": i + 1,
          "category_id": int(rng.integers(1, num_classes + 1)),
          "bbox": [x, y, w, h],
          "area": w * h,
          "iscrowd": 0,
      })
      ann_id += 1
  categories = [{"id": c + 1, "name": f"class_{c}"} for c in range(num_classes)]
  (root / "annotations.json").write_text(
      json.dumps({
          "images": images,
          "annotations": annotations,
          "categories": categories
      }))


def _args(root, **overrides):
  base = dict(
      pipeline="detection",
      method="detection",
      dataset=f"detfolder/{root}",
      image_column="image",
      val_split=0.25,
      cache_dir=None,
      model="det_wrapper:timm/resnet18",
      model_arg={
          "head": "fcos_head:num_convs=1",
          "pretrained": False
      },
      image_size=64,
      batch_size=2,
      num_workers=0,
      seed=0,
      device="cpu",
      det_annotation_file="annotations.json",
      det_image_subdir="images",
      det_num_classes=None,
      det_score_thresh=0.05,
      det_nms_thresh=0.5,
      det_max_detections=100,
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


def test_registry_knows_det_wrapper():
  from genml_kit.models.registry import MODELS
  assert MODELS.contains("det_wrapper")


def test_pipeline_and_method_contract(tmp_path):
  _write_dataset(tmp_path)
  args, pipeline, method, model = _build(tmp_path)

  assert method._num_classes == 2
  blob = next(iter(pipeline.train_loader))
  blob = pipeline.to_device(blob, args.device)
  # Images are a (B, C, H, W) tensor; boxes/labels stay as per-image lists.
  assert blob.data.dim() == 4
  assert isinstance(blob.meta["boxes"], list)
  assert isinstance(blob.meta["labels"], list)
  assert blob.meta["boxes"][0].shape[1] == 4

  out = method.train_step(model, blob, 0)
  assert torch.isfinite(out.loss)

  metrics = method.evaluate(model, pipeline.val_loader, args.device, pipeline.to_device)
  for key in ("map", "map_50", "map_75", "loss"):
    assert key in metrics


def test_explicit_num_classes_override(tmp_path):
  _write_dataset(tmp_path)
  args, pipeline, method, model = _build(tmp_path, det_num_classes=5)
  assert method._num_classes == 5
  blob = next(iter(pipeline.train_loader))
  out = method.train_step(model, pipeline.to_device(blob, args.device), 0)
  assert torch.isfinite(out.loss)


def test_checkpoint_state_roundtrip(tmp_path):
  _write_dataset(tmp_path)
  args, pipeline, method, model = _build(tmp_path)
  state = method.get_checkpoint_state(model, args)
  assert state["num_classes"] == 2

  fresh = build_method("detection")
  fresh.load_checkpoint_state(model, state, args)
  assert fresh._num_classes == 2


def test_decode_predictions_shapes(tmp_path):
  from genml_kit.methods.detection import decode_predictions

  _write_dataset(tmp_path)
  args, pipeline, method, model = _build(tmp_path)
  blob = pipeline.to_device(next(iter(pipeline.train_loader)), args.device)
  out = model(pixel_values=blob.data)
  decoded = decode_predictions(out.logits, score_thresh=0.0)
  assert len(decoded) == blob.data.shape[0]
  for det in decoded:
    assert det["boxes"].shape[1] == 4
    assert det["boxes"].shape[0] == det["scores"].shape[0] == det["labels"].shape[0]


def test_freeze_keeps_encoder_params_fixed(tmp_path):
  """``--freeze`` lists the *trainable* patterns (repo semantics).

  So ``--freeze 'head.*'`` freezes the encoder and keeps the head trainable.
  """
  _write_dataset(tmp_path)
  args, pipeline, method, model = _build(tmp_path, freeze="head.*")
  encoder_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
  head_params = [p for n, p in model.named_parameters() if n.startswith("head.")]
  assert encoder_params and head_params
  assert all(not p.requires_grad for p in encoder_params)
  assert all(p.requires_grad for p in head_params)


@pytest.mark.slow
def test_detection_learns_a_simple_pattern(tmp_path):
  """Slow smoke test: overfit a single fixed box per image.

  The same bright square is placed at the same location in every image, so a
  model with enough capacity should drive the mAP well above the random
  baseline.
  """
  torch.manual_seed(0)
  np.random.seed(0)
  size = 64
  rng = np.random.default_rng(0)
  (tmp_path / "images").mkdir()
  images = []
  annotations = []
  for i in range(16):
    img = rng.integers(0, 40, (size, size, 3), dtype=np.uint8)
    # Bright square at a fixed location: an easy, learnable target.
    img[16:48, 16:48] = 255
    Image.fromarray(img).save(tmp_path / "images" / f"{i}.png")
    images.append({"id": i + 1, "file_name": f"{i}.png", "width": size, "height": size})
    annotations.append({
        "id": i + 1,
        "image_id": i + 1,
        "category_id": 1,
        "bbox": [16, 16, 32, 32],
        "area": 32 * 32,
        "iscrowd": 0,
    })
  (tmp_path / "annotations.json").write_text(
      json.dumps({
          "images": images,
          "annotations": annotations,
          "categories": [{
              "id": 1,
              "name": "square"
          }]
      }))

  args, pipeline, method, model = _build(tmp_path, image_size=size)
  baseline = method.evaluate(model, pipeline.val_loader, args.device,
                             pipeline.to_device)["map"]

  blob = pipeline.to_device(next(iter(pipeline.train_loader)), args.device)
  opt = torch.optim.Adam(model.parameters(), lr=1e-3)
  for _ in range(40):
    out = method.train_step(model, blob, 0)
    opt.zero_grad()
    out.loss.backward()
    opt.step()
  metrics = method.evaluate(model, pipeline.val_loader, args.device, pipeline.to_device)
  assert metrics["map"] >= baseline
