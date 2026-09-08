"""Tests for the XGBoost-on-backbone pipeline (#9).

``train_xgboost_on_backbone`` takes an explicit API (no ``args``
namespace).  The torch/model side is mocked; the real XGBClassifier
trains on synthetic features so the orchestration is genuinely exercised.
"""

import contextlib
import inspect
import os
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from genml_kit.training.xgb_pipeline import train_xgboost_on_backbone


class TestExplicitApi:

  def test_no_args_parameter(self):
    """The signature must not take an ``args`` namespace."""
    sig = inspect.signature(train_xgboost_on_backbone)
    assert "args" not in sig.parameters
    assert "xgb_kwargs" not in sig.parameters
    for name in ("checkpoint_dir", "model_spec", "image_size", "output_path"):
      assert name in sig.parameters
    # Explicit hyperparameters with CLI-matching defaults.
    defaults = {n: p.default for n, p in sig.parameters.items()}
    assert defaults["use_gpu"] is False
    assert defaults["max_depth"] == 6
    assert defaults["n_estimators"] == 200
    assert defaults["learning_rate"] == 0.1
    assert defaults["subsample"] == 0.8
    assert defaults["colsample_bytree"] == 0.8
    assert defaults["min_child_weight"] == 1
    assert defaults["gamma"] == 0.0
    assert defaults["reg_alpha"] == 0.0


def _fake_collect_features(model, dataset, device, batch_size=128):
  """Return synthetic 2-class features regardless of the dataset."""
  features = np.random.RandomState(0).randn(20, 4).astype(np.float32)
  labels = np.arange(20) % 2
  return features, labels


class TestEndToEnd:

  def _pipeline_patches(self, tmp_path):
    fake_model = MagicMock()
    fake_processor = object()
    return (
        patch("genml_kit.io.checkpointing.select_best_checkpoint",
              return_value=str(tmp_path / "ckpt_best.pt")),
        patch("genml_kit.training.model_utils.load_model_for_inference",
              return_value=(fake_model, fake_processor)),
        patch("genml_kit.training.model_utils.build_val_transform", return_value=None),
        patch("genml_kit.training.model_utils.collect_features",
              side_effect=_fake_collect_features),
    )

  @staticmethod
  def _fake_proxy(ds, transform=None):
    proxy = MagicMock()
    proxy.id2label = {0: "benign", 1: "malignant"}
    return proxy

  def test_trains_and_saves_model(self, tmp_path):
    out_path = str(tmp_path / "xgb_model.json")
    fake_ds = object()  # opaque; only passed through to HFDatasetProxy

    with contextlib.ExitStack() as stack:
      for cm in self._pipeline_patches(tmp_path):
        stack.enter_context(cm)
      stack.enter_context(
          patch("genml_kit.training.xgb_pipeline.HFDatasetProxy",
                side_effect=self._fake_proxy))
      train_xgboost_on_backbone(
          fake_ds,
          fake_ds,
          torch.device("cpu"),
          num_labels=2,
          checkpoint_dir=str(tmp_path / "ckpt"),
          model_spec="convvit",
          image_size=32,
          output_path=out_path,
          batch_size=4,
          n_estimators=5,
      )

    assert os.path.exists(out_path), "XGBoost model was not saved"

  def test_wraps_datasets_with_val_transform(self, tmp_path):
    """Train and val datasets must go through HFDatasetProxy wrapping."""
    out_path = str(tmp_path / "xgb_model.json")
    fake_ds = object()
    proxy_calls = []

    def _proxy(ds, transform=None):
      proxy_calls.append((ds, transform))
      return self._fake_proxy(ds, transform)

    with contextlib.ExitStack() as stack:
      for cm in self._pipeline_patches(tmp_path):
        stack.enter_context(cm)
      stack.enter_context(
          patch("genml_kit.training.xgb_pipeline.HFDatasetProxy", side_effect=_proxy))
      train_xgboost_on_backbone(
          fake_ds,
          fake_ds,
          torch.device("cpu"),
          num_labels=2,
          checkpoint_dir=str(tmp_path / "ckpt"),
          model_spec="convvit",
          image_size=64,
          output_path=out_path,
          n_estimators=5,
      )

    assert len(proxy_calls) == 2

  def test_no_checkpoint_short_circuits(self, tmp_path):
    """No resolvable checkpoint -> return without loading or saving."""
    out_path = str(tmp_path / "xgb_model.json")
    fake_ds = object()

    with (patch(
        "genml_kit.io.checkpointing.select_best_checkpoint",
        return_value=None,
    ), patch("genml_kit.training.model_utils.load_model_for_inference") as load_mock):
      result = train_xgboost_on_backbone(
          fake_ds,
          fake_ds,
          torch.device("cpu"),
          num_labels=2,
          checkpoint_dir=str(tmp_path / "empty"),
          model_spec="convvit",
          image_size=32,
          output_path=out_path,
      )

    assert result is None
    load_mock.assert_not_called()
    assert not os.path.exists(out_path)
