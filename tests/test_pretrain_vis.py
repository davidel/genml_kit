"""Tests for SimMIMMethod.log_validation.

Regression tests for the vis-logging contract: production loaders use
``_images_collate`` and yield ``DataBlob`` namedtuples (not dicts), so
``log_validation`` must unpack ``blob.data`` rather than indexing by a
column name.  The base ``Method.log_validation`` is a silent no-op.
"""

import torch
import torch.nn as nn

from genml_kit.methods import get_method
from genml_kit.pipelines.contracts import DataBlob


class _Writer:
  """Minimal TensorBoard writer stand-in recording add_image calls."""

  def __init__(self):
    self._images = []

  def add_image(self, tag, tensor, global_step):
    self._images.append((tag, tensor.clone(), global_step))


class _FakeSimMIM(nn.Module):
  """Minimal object with the interface SimMIMMethod.log_validation needs.

  ``forward(images, mask)`` returns ``(output, target)`` where *output*
  is the flattened patch tensor ``unpatchify`` can reshape back to
  ``(B, C, H, W)``.
  """

  def __init__(self, patch_size=4, img_size=16, channels=3):
    super().__init__()
    self.patch_size = patch_size
    self.mask_ratio = 0.5
    self.in_channels = channels
    num_patches = (img_size // patch_size)**2
    self._out_dim = num_patches * patch_size * patch_size * channels
    self._img_size = img_size

  def forward(self, images, mask):
    b = images.shape[0]
    patch = self.patch_size
    h = w = self._img_size // patch
    # unpatched (B, C, h, w, patch, patch) -> flat (B, N, patch^2*C)
    num_patches = h * w
    dim = patch * patch * self.in_channels
    recon = torch.zeros(b, num_patches, dim)
    return recon, images


def _blob_loader(num_batches=1, batch_size=2, size=16):
  """A loader yielding DataBlob batches (production collate semantics)."""

  class _Loader:

    def __iter__(self):
      for _ in range(num_batches):
        yield DataBlob(data=torch.randn(batch_size, 3, size, size),
                       meta={"labels": None})

  return _Loader()


def test_base_log_validation_is_noop():
  """Methods without vis support log nothing (silent no-op)."""
  method = get_method("dino")()
  writer = _Writer()
  loader = _blob_loader()

  def _to_device(blob, device):
    return blob

  method.log_validation(None, loader, _to_device, writer, 0, torch.device("cpu"))
  assert writer._images == []


def test_simmim_logs_recon_pair():
  """SimMIM logs one original + one reconstructed image."""
  method = get_method("simmim")()
  writer = _Writer()
  loader = _blob_loader(batch_size=2, size=16)

  def _to_device(blob, device):
    return DataBlob(data=blob.data.to(device), meta=blob.meta)

  model = _FakeSimMIM()
  method.log_validation(model,
                        loader,
                        _to_device,
                        writer,
                        7,
                        torch.device("cpu"),
                        num_samples=2)
  tags = [tag for tag, _, _ in writer._images]
  assert "recon/original" in tags
  assert "recon/reconstructed" in tags
  assert all(step == 7 for _, _, step in writer._images)


def test_simmim_logs_first_view_of_multiview():
  """A multi-view blob logs the first view only."""
  method = get_method("simmim")()
  writer = _Writer()

  class _Loader:

    def __iter__(self):
      yield DataBlob(data=(torch.randn(2, 3, 16, 16), torch.randn(2, 3, 16, 16)),
                     meta={"labels": None})

  def _to_device(blob, device):
    return DataBlob(data=tuple(d.to(device) for d in blob.data), meta=blob.meta)

  model = _FakeSimMIM()
  method.log_validation(model,
                        _Loader(),
                        _to_device,
                        writer,
                        0,
                        torch.device("cpu"),
                        num_samples=2)
  original = [t for tag, t, _ in writer._images if tag == "recon/original"][0]
  # Single image (C, H, W) after [0] indexing.
  assert original.ndim == 3
