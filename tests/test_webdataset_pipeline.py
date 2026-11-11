"""Integration: a ``webdataset:`` spec through the real images pipeline.

``test_iterable_ensemble`` covers the spec parsing and the ensemble itself; this
drives the actual ``ImagesPipeline`` so the adapter/logging/collate chain is
exercised too -- the layer where a stream (rather than a map-style dataset) has
to be recognised and wrapped correctly.
"""

import io

import pytest
from PIL import Image
from torchvision.transforms import v2

from genml_kit.pipelines.images import ImagesPipeline
from tests.cli_args import train_args

webdataset = pytest.importorskip("webdataset")


def _make_shards(dirpath, n_shards=2, n_samples=4):
  pattern = str(dirpath / "shard-%04d.tar")
  for s in range(n_shards):
    with webdataset.TarWriter(pattern % s) as writer:
      for i in range(n_samples):
        buf = io.BytesIO()
        Image.new("RGB", (16, 16), color=(i * 20, s * 20, 0)).save(buf, "PNG")
        writer.write({"__key__": f"{s}-{i}", "png": buf.getvalue()})
  return str(dirpath / "shard-{0000..0001}.tar")


class _ToTensor(v2.Transform):
  """PIL -> float tensor, no resize (a resize would need a real image size)."""

  def forward(self, img):
    import torch
    tensor = v2.functional.to_image(img)
    return v2.functional.to_dtype(tensor, torch.float32, scale=True)


def _args(**overrides):
  pinned = dict(
      pipeline="images",
      batch_size=2,
      num_workers=0,
  )
  pinned.update(overrides)
  return train_args(**pinned)


class TestWebdatasetThroughImagesPipeline:

  def test_loader_yields_image_batches(self, tmp_path):
    spec = "webdataset:" + _make_shards(tmp_path)
    pipeline = ImagesPipeline()
    pipeline.build_loader(_args(datasets=[spec]),
                          needs_labels=False,
                          transform=_ToTensor())
    batches = list(pipeline.train_loader)
    assert batches
    assert sum(len(b.data) for b in batches) == 8

  def test_streaming_ensemble_uses_iterable_adapter(self, tmp_path):
    from genml_kit.pipelines.images import _IterableDictAdapter
    spec = "webdataset:" + _make_shards(tmp_path)
    pipeline = ImagesPipeline()
    pipeline.build_loader(_args(datasets=[spec]),
                          needs_labels=False,
                          transform=_ToTensor())
    assert isinstance(pipeline.train_dataset, _IterableDictAdapter)

  def test_build_pretrain_dataset_uses_iterable_selector(self, tmp_path):
    # ``build_pretrain_dataset`` is the other ensemble entry point; it must pick
    # the iterable field selector for a stream, or the loader would index it.
    from genml_kit.datasets.field_selector import IterableFieldSelectorDataset
    from genml_kit.pipelines.images import build_pretrain_dataset
    spec = "webdataset:" + _make_shards(tmp_path)
    dataset, ensemble = build_pretrain_dataset(_args(datasets=[spec]),
                                               needs_labels=False)
    assert ensemble.streaming
    assert isinstance(dataset, IterableFieldSelectorDataset)
    assert next(iter(dataset))["image"] is not None

  def test_batch_tensor_shape(self, tmp_path):
    spec = "webdataset:" + _make_shards(tmp_path)
    pipeline = ImagesPipeline()
    pipeline.build_loader(_args(datasets=[spec], batch_size=3),
                          needs_labels=False,
                          transform=_ToTensor())
    batch = next(iter(pipeline.train_loader))
    # (batch, channels, height, width).  The pipeline's own transform resizes to
    # ``--image_size`` (448 by default), so the spatial dims are that, not the
    # 16x16 source size.
    assert batch.data.shape[0] == 3
    assert batch.data.shape[1] == 3
    assert batch.data.shape[2] == batch.data.shape[3]
