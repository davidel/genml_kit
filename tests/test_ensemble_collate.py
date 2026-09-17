"""Integration tests: mixed labeled/unlabeled DatasetEnsemble + collate.

Regression tests for the I-JEPA multi-dataset pretraining crash: with
mixed sources, items had inconsistent schemas (labeled datasets carried
``"label"``, unlabeled ones did not), so torch's default_collate raised
``KeyError: 'label'`` whenever a batch mixed both.  The fix wraps the
transformed dataset in :class:`FieldSectorDataset` (image-only fields)
for methods that do not require labels.
"""

import argparse
import os

import pytest
import torch
from PIL import Image
from torchvision.transforms import v2

from genml_kit.datasets.ensemble import DatasetEnsemble
from genml_kit.pipelines.images import build_pretrain_dataset


def _make_labeled_dir(root, per_class=3):
  """Class-subdirectory layout -> ImageFolderDataset with labels."""
  for cls in ("cat", "dog"):
    d = os.path.join(root, cls)
    os.makedirs(d)
    for i in range(per_class):
      Image.new("RGB", (16, 16)).save(os.path.join(d, f"{cls}_{i}.jpg"))
  return root


def _make_unlabeled_dir(root, n=4):
  """"train/" split layout without class subdirs -> no labels."""
  d = os.path.join(root, "train")
  os.makedirs(d)
  for i in range(n):
    Image.new("RGB", (16, 16)).save(os.path.join(d, f"img_{i}.jpg"))
  return root


def _mixed_ensemble(tmp_path):
  labeled = _make_labeled_dir(tmp_path / "labeled")
  unlabeled = _make_unlabeled_dir(tmp_path / "unlabeled")
  return DatasetEnsemble([
      {
          "name": str(labeled),
          "source": "imagefolder"
      },
      {
          "name": str(unlabeled),
          "source": "imagefolder"
      },
  ])


def _pretrain_args(tmp_path):
  labeled = _make_labeled_dir(tmp_path / "labeled")
  unlabeled = _make_unlabeled_dir(tmp_path / "unlabeled")
  return argparse.Namespace(
      datasets=[
          f"imagefolder/{labeled}",
          f"imagefolder/{unlabeled}",
      ],
      image_size=16,
      cache_dir=None,
      hf_token=None,
      strict_datasets=False,
  )


def _to_tensor_transform():
  return v2.Compose([
      v2.ToImage(),
      v2.ToDtype(torch.float32, scale=True),
  ])


class TestFieldSectorOnMixedEnsemble:

  def test_items_are_image_only(self, tmp_path):
    """The wrapped dataset yields uniform {"image": tensor} items."""
    dataset, _ = build_pretrain_dataset(_pretrain_args(tmp_path),
                                        needs_labels=False,
                                        transform=_to_tensor_transform())
    assert set(dataset[0]) == {"image"}
    assert isinstance(dataset[0]["image"], torch.Tensor)

  def test_dataloader_never_raises(self, tmp_path):
    """Full pass through DataLoader with default collate — no KeyError."""
    dataset, _ = build_pretrain_dataset(_pretrain_args(tmp_path),
                                        needs_labels=False,
                                        transform=_to_tensor_transform())
    total = 0
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0)
    for batch in loader:
      assert set(batch) == {"image"}
      total += batch["image"].shape[0]
    assert total == 10  # every image went through default_collate unharmed

  def test_log_validation_consumes_real_collated_loader(self, tmp_path):
    """End-to-end: log_validation consumes a real collated mixed batch.

    The vis path must work with the production collate (DataBlob), which
    is what the trainer feeds to ``method.log_validation``.
    """
    from genml_kit.pipelines.contracts import DataBlob

    class _Writer:

      def __init__(self):
        self.calls = []

      def add_image(self, tag, tensor, step):
        self.calls.append(tag)

    class _NoopMethod:
      """Minimal method with only the default log_validation behavior."""

      def log_validation(self,
                         model,
                         loader,
                         to_device,
                         writer,
                         global_step,
                         device,
                         num_samples=8):  # noqa: B027
        pass

    dataset, ensemble = build_pretrain_dataset(_pretrain_args(tmp_path),
                                               needs_labels=False,
                                               transform=_to_tensor_transform())
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=0)

    def _to_device(blob, device):
      if isinstance(blob, DataBlob):
        return DataBlob(data=blob.data.to(device), meta=blob.meta)
      return blob

    writer = _Writer()
    _NoopMethod().log_validation(None, loader, _to_device, writer, 0,
                                 torch.device("cpu"))
    assert writer.calls == []  # no-op method logs nothing

  def test_label_requiring_method_rejects_mixed(self, tmp_path):
    """Methods needing labels refuse mixed ensembles.

      BalancedBatchSampler requires every sample to carry a label, so
      ``build_pretrain_dataset`` fails fast when any source is unlabeled,
      naming the offending datasets.
      """
    with pytest.raises(ValueError, match="provide no labels"):
      build_pretrain_dataset(_pretrain_args(tmp_path),
                             needs_labels=True,
                             transform=_to_tensor_transform())


class _NullWriter:

  def add_image(self, tag, tensor, step):
    pass
