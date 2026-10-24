"""Tests for encoder feature-extraction fallback semantics (#3).

``encode_with_backbone`` must fall back to a plain forward pass only for
"encoder not supported" failures, and must propagate genuine runtime
failures (in particular ``torch.cuda.OutOfMemoryError``).
"""

import pytest
import torch

from genml_kit.models.encoder_utils import (
    encode_with_backbone,
    require_matching_features,
)


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


class TestModelOutputFallback:
  """Regression for #9: .logits-only containers must not AttributeError."""

  def _run_two_paths(self):
    from genml_kit.models.registry import ModelOutput

    class _LogitsOnly(torch.nn.Module):

      def forward(self, images):
        return ModelOutput(images)

    class _PoolerAndLogits(torch.nn.Module):

      def forward(self, images):
        out = torch.nn.Module()
        out.logits = images
        out.pooler_output = images + 1
        return out

    class _HiddenAndLogits(torch.nn.Module):

      def forward(self, images):
        out = torch.nn.Module()
        out.logits = images
        out.last_hidden_state = images
        return out

    return _LogitsOnly(), _PoolerAndLogits(), _HiddenAndLogits()

  def test_logits_only_returns_logits(self, monkeypatch):
    enc, _, _ = self._run_two_paths()
    monkeypatch.setattr("genml_kit.training.model_utils.extract_backbone_features",
                        lambda m, x: (_ for _ in ()).throw(AttributeError("no hook")))
    imgs = torch.ones(2, 3, 8, 8)
    out = encode_with_backbone(enc, imgs)
    assert torch.equal(out, imgs)

  def test_pooler_output_wins(self, monkeypatch):
    _, enc, _ = self._run_two_paths()
    monkeypatch.setattr("genml_kit.training.model_utils.extract_backbone_features",
                        lambda m, x: (_ for _ in ()).throw(ValueError("no hook")))
    imgs = torch.ones(2, 3, 8, 8)
    out = encode_with_backbone(enc, imgs)
    assert torch.equal(out, imgs + 1)

  def test_last_hidden_state_mean(self, monkeypatch):
    _, _, enc = self._run_two_paths()
    monkeypatch.setattr("genml_kit.training.model_utils.extract_backbone_features",
                        lambda m, x: (_ for _ in ()).throw(TypeError("no hook")))
    imgs = torch.ones(2, 3, 8, 8)
    out = encode_with_backbone(enc, imgs)
    assert torch.equal(out, imgs.mean(dim=1))


class TestBareRawFallbackContract:
  """§25: the bare-`raw` fallback returns the encoder output verbatim.

  A backbone whose forward yields an un-pooled tensor must come back
  unchanged — callers relying on the ``(B, D)`` contract are responsible
  for pooling, and ``require_matching_features`` is the loud guard.
  """

  def test_raw_output_returned_verbatim(self, monkeypatch):
    monkeypatch.setattr("genml_kit.training.model_utils.extract_backbone_features",
                        lambda m, x: (_ for _ in ()).throw(AttributeError("no hook")))

    class _SpatialEncoder(torch.nn.Module):

      def forward(self, images):
        # Un-pooled, non-(B, D): returned as-is by the bare fallback.
        return images + 1

    imgs = torch.ones(2, 3, 8, 8)
    out = encode_with_backbone(_SpatialEncoder(), imgs)
    assert out.shape == imgs.shape
    assert torch.equal(out, imgs + 1)


class TestRequireMatchingFeatures:
  """§25: mismatched feature shapes must fail loudly, not broadcast."""

  def test_equal_shapes_pass_and_return_first(self):
    a = torch.zeros(2, 4)
    b = torch.ones(2, 4)
    assert require_matching_features(a, b) is a
    assert require_matching_features(a, b, torch.zeros(2, 4)) is a

  def test_broadcast_trap_raises(self):
    # (B, 4) vs (B, 1, 4): torch would broadcast these silently.
    with pytest.raises(ValueError, match="Feature shapes must match"):
      require_matching_features(torch.zeros(2, 4), torch.zeros(2, 1, 4))

  def test_mismatch_message_includes_shapes_and_names(self):
    with pytest.raises(ValueError) as excinfo:
      require_matching_features(torch.zeros(2, 4),
                                torch.zeros(2, 3),
                                names=("student", "teacher"))
    msg = str(excinfo.value)
    assert "student:(2, 4)" in msg
    assert "teacher:(2, 3)" in msg

  def test_fewer_than_two_tensors_raises(self):
    with pytest.raises(ValueError, match="at least two tensors"):
      require_matching_features(torch.zeros(2, 4))
    with pytest.raises(ValueError, match="at least two tensors"):
      require_matching_features()
