"""Tests for encoder feature-extraction fallback semantics (#3).

``encode_with_backbone`` must fall back to a plain forward pass only for
"encoder not supported" failures, and must propagate genuine runtime
failures (in particular ``torch.cuda.OutOfMemoryError``).
"""

import pytest
import torch

from genml_kit.models.encoder_utils import encode_with_backbone


class _FailingHookEncoder(torch.nn.Module):
  """Encoder whose hook-based extraction raises *exc*; forward is a no-op."""

  def __init__(self, exc):
    super().__init__()
    self._exc = exc

  def forward(self, images):  # pragma: no cover - only reached on fallback
    return images.mean(dim=(2, 3))


class TestEncodeWithBackboneFallback:

  def _assert_fallback_used(self, exc, monkeypatch):
    encoder = _FailingHookEncoder(exc)

    def _boom(model, pixel_values):
      raise exc

    monkeypatch.setattr(
        "genml_kit.training.model_utils.extract_backbone_features",
        _boom,
    )
    return encoder

  def test_attribute_error_falls_back_to_forward(self, monkeypatch):
    encoder = self._assert_fallback_used(AttributeError("no classifier"), monkeypatch)
    imgs = torch.ones(2, 3, 8, 8)
    assert torch.allclose(encode_with_backbone(encoder, imgs), imgs.mean(dim=(2, 3)))

  def test_type_error_falls_back(self, monkeypatch):
    encoder = self._assert_fallback_used(TypeError("bad dtype"), monkeypatch)
    imgs = torch.ones(2, 3, 8, 8)
    assert torch.allclose(encode_with_backbone(encoder, imgs), imgs.mean(dim=(2, 3)))

  def test_value_error_falls_back(self, monkeypatch):
    encoder = self._assert_fallback_used(ValueError("bad shape"), monkeypatch)
    imgs = torch.zeros(1, 3, 4, 4)
    assert torch.allclose(encode_with_backbone(encoder, imgs), imgs.mean(dim=(2, 3)))

  def test_runtime_error_propagates(self, monkeypatch):
    encoder = self._assert_fallback_used(RuntimeError("boom"), monkeypatch)
    with pytest.raises(RuntimeError, match="boom"):
      encode_with_backbone(encoder, torch.ones(1, 3, 4, 4))

  def test_oom_error_propagates(self, monkeypatch):
    encoder = self._assert_fallback_used(torch.cuda.OutOfMemoryError(), monkeypatch)
    with pytest.raises(torch.cuda.OutOfMemoryError):
      encode_with_backbone(encoder, torch.ones(1, 3, 4, 4))

  def test_success_path_does_not_call_forward(self, monkeypatch):
    encoder = torch.nn.Module()

    def _good(model, pixel_values):
      return torch.ones(1, 8)

    monkeypatch.setattr("genml_kit.training.model_utils.extract_backbone_features",
                        _good)
    out = encode_with_backbone(encoder, torch.ones(1, 3, 4, 4))
    assert out.shape == (1, 8)
