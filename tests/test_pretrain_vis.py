"""Tests for genml_kit.pretrain.cli.log_validation_images.

Regression tests for the loader contract: pretrain loaders yield
dicts (``{image_column: tensor, ...}``), so the visualization helper
must unpack the batch by column name rather than slicing it directly.
"""

import torch
import torch.nn as nn

from genml_kit.pretrain.cli import log_validation_images


class _Writer:
  """Minimal TensorBoard writer stand-in recording add_image calls."""

  def __init__(self):
    self.images = []

  def add_image(self, tag, tensor, step):
    self.images.append((tag, tensor, step))


class _FakeModel(nn.Module):
  """Tiny module tracking eval/train transitions."""

  def __init__(self):
    super().__init__()
    self.lin = nn.Linear(2, 2)

  @property
  def in_eval(self):
    return not self.training


class _NoValidateMethod:
  """Method without pixel-space visualisations (e.g. I-JEPA)."""

  def validate(self, model, images, num_samples):
    return None


class _ReconMethod:
  """Method returning a reconstruction for each input image."""

  def validate(self, model, images, num_samples):
    return torch.clamp(images, 0, 1)


def _dict_loader(image_column, batch, labels=None):
  """One-shot loader yielding a single dict batch."""

  class _Loader:

    def __iter__(self):
      item = {image_column: batch}
      if labels is not None:
        item["label"] = labels
      yield item

  return _Loader()


def _tensor_loader(batch):
  """One-shot loader yielding a bare tensor (legacy contract)."""

  class _Loader:

    def __iter__(self):
      yield batch

  return _Loader()


class TestLogValidationImages:

  def test_dict_batch_does_not_raise(self):
    """Regression: dict batches must not be sliced as tensors."""
    model = _FakeModel()
    loader = _dict_loader("image", torch.rand(16, 3, 8, 8))
    writer = _Writer()

    # Must not raise KeyError: slice(...) — and must restore train mode.
    log_validation_images(_NoValidateMethod(),
                          model,
                          loader,
                          writer,
                          0,
                          torch.device("cpu"),
                          image_column="image")
    assert not model.in_eval

  def test_dict_batch_logs_reconstructions(self):
    model = _FakeModel()
    batch = torch.rand(4, 3, 8, 8)
    loader = _dict_loader("image", batch)
    writer = _Writer()

    log_validation_images(_ReconMethod(),
                          model,
                          loader,
                          writer,
                          42,
                          torch.device("cpu"),
                          image_column="image")
    assert len(writer.images) == 2
    tags = [tag for tag, _, _ in writer.images]
    assert tags == ["recon/original", "recon/reconstructed"]
    steps = [step for _, _, step in writer.images]
    assert steps == [42, 42]

  def test_dict_batch_slices_num_samples(self):
    model = _FakeModel()
    batch = torch.rand(16, 3, 8, 8)
    loader = _dict_loader("image", batch)
    writer = _Writer()

    captured = {}

    def _capture(model_arg, images, num_samples):
      captured["num"] = images.shape[0]
      return None

    method = _NoValidateMethod()
    method.validate = _capture
    log_validation_images(method,
                          model,
                          loader,
                          writer,
                          0,
                          torch.device("cpu"),
                          image_column="image",
                          num_samples=8)
    assert captured["num"] == 8

  def test_train_mode_restored_when_no_recon(self):
    """validate() returning None must not leave the model in eval mode."""
    model = _FakeModel()
    loader = _dict_loader("image", torch.rand(4, 3, 8, 8))
    writer = _Writer()

    log_validation_images(_NoValidateMethod(),
                          model,
                          loader,
                          writer,
                          0,
                          torch.device("cpu"),
                          image_column="image")
    assert model.training

  def test_train_mode_restored_on_recon(self):
    model = _FakeModel()
    loader = _dict_loader("image", torch.rand(4, 3, 8, 8))
    writer = _Writer()

    log_validation_images(_ReconMethod(),
                          model,
                          loader,
                          writer,
                          0,
                          torch.device("cpu"),
                          image_column="image")
    assert model.training

  def test_tuple_batch_dual_view(self):
    """Dual-view transforms yield tuples of tensors — both sliced."""
    model = _FakeModel()
    loader = _dict_loader("image", (torch.rand(4, 3, 8, 8), torch.rand(4, 3, 8, 8)))
    writer = _Writer()
    seen = {}

    def _capture(model_arg, images, num_samples):
      seen["type"] = type(images)
      seen["lens"] = [v.shape[0] for v in images]
      return None

    method = _NoValidateMethod()
    method.validate = _capture
    log_validation_images(method,
                          model,
                          loader,
                          writer,
                          0,
                          torch.device("cpu"),
                          image_column="image",
                          num_samples=2)
    assert seen["lens"] == [2, 2]
    assert not model.in_eval
