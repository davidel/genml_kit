"""Tests for the ``build_transformer_encoder`` utility."""

import warnings

import torch

from genml_kit.utils.transformer import build_transformer_encoder


def _make_layer(d_model, nhead, **kwargs):
  return torch.nn.TransformerEncoderLayer(
      d_model=d_model,
      nhead=nhead,
      dim_feedforward=d_model * 4,
      dropout=0.0,
      batch_first=True,
      **kwargs,
  )


def test_returns_encoder_with_expected_structure():
  encoder = build_transformer_encoder(_make_layer(8, 2), num_layers=3)
  assert isinstance(encoder, torch.nn.TransformerEncoder)
  assert len(encoder.layers) == 3


def test_final_norm_is_passed_through():
  norm = torch.nn.LayerNorm(8)
  encoder = build_transformer_encoder(_make_layer(8, 2), num_layers=2, norm=norm)
  assert encoder.norm is norm


def test_layers_have_distinct_initialization():
  torch.manual_seed(0)
  encoder = build_transformer_encoder(_make_layer(8, 2), num_layers=4)
  weights = [layer.linear1.weight for layer in encoder.layers]
  for i in range(len(weights)):
    for j in range(i + 1, len(weights)):
      assert not torch.equal(weights[i], weights[j])


def test_pre_norm_construction_emits_no_warnings():
  from genml_kit.pretrain.methods.ijepa import _Predictor

  with warnings.catch_warnings():
    warnings.simplefilter("error")
    predictor = _Predictor(
        embed_dim=32,
        num_patches=16,
        depth=2,
        num_heads=4,
        predictor_dim=32,
    )
  assert len(predictor.transformer.layers) == 2


def test_cls_attention_encoder_layers_differ():
  from genml_kit.training.classifiers.cls_attention import Classifier

  net = Classifier(
      num_labels=7,
      hidden_size=64,
      num_heads=4,
      num_encoder_layers=2,
  )
  weights = [layer.linear1.weight for layer in net.encoder.layers]
  assert not torch.equal(weights[0], weights[1])
