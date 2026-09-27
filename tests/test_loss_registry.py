"""Tests for the loss registry (LOSSES, load_loss, external scripts)."""

import argparse
import os
import tempfile

import pytest
import torch

from genml_kit.losses import DEFAULT_LOSS, LOSSES, load_loss
from genml_kit.losses.focal import CombinedFocalLoss
from genml_kit.methods.classification import ClassificationMethod
from genml_kit.utils.args import json_dict_type


def _write_script(body):
  """Write *body* to a temp .py file; return its path."""
  fd, path = tempfile.mkstemp(suffix=".py")
  with os.fdopen(fd, "w") as f:
    f.write(body)
  return path


class TestBuiltinLosses:

  def test_default_is_focal(self):
    assert DEFAULT_LOSS == "focal"
    loss = load_loss()
    assert isinstance(loss, CombinedFocalLoss)

  def test_none_defaults_to_focal(self):
    loss = load_loss(None)
    assert isinstance(loss, CombinedFocalLoss)

  def test_registered_focal(self):
    assert LOSSES.contains("focal")
    loss = load_loss("focal", gamma=0.5, label_smoothing=0.1)
    assert isinstance(loss, CombinedFocalLoss)
    assert loss.gamma == 0.5
    assert loss.label_smoothing == 0.1

  def test_rl_bundles_registered(self):
    for name in ("ppo", "sac", "dqn"):
      assert LOSSES.contains(name), f"missing bundle {name}"
    ppo = load_loss("ppo")
    assert set(("policy", "value", "entropy")) <= set(ppo)
    sac = load_loss("sac")
    assert set(("q", "policy", "alpha")) <= set(sac)
    dqn = load_loss("dqn")
    assert "td" in dqn


class TestScriptLoss:

  def test_script_loss(self):
    path = _write_script("import torch.nn as nn\n"
                         "class MyLoss(nn.Module):\n"
                         "  def __init__(self, alpha):\n"
                         "    super().__init__()\n"
                         "    self.alpha = alpha\n"
                         "  def forward(self, logits, targets):\n"
                         "    return logits.mean()\n"
                         "def build_loss(alpha=1.0, **kwargs):\n"
                         "  return MyLoss(alpha=alpha)\n")
    loss = load_loss(path, alpha=2.5)
    assert isinstance(loss, torch.nn.Module)
    assert loss.alpha == 2.5

  def test_script_missing_build_loss(self):
    path = _write_script("def not_build_loss():\n    pass\n")
    with pytest.raises(ValueError):
      load_loss(path)

  def test_script_url_dispatch(self, monkeypatch):
    # A URL string triggers the script branch, not the registry lookup.
    # Monkeypatch load_extern so no real network is touched; assert the
    # script path (build_loss) is used for a URL spec.
    from genml_kit import losses as losses_mod

    def fake_load_extern(path_or_url, fn_name):
      assert path_or_url == "https://example.invalid/loss.py"
      assert fn_name == "build_loss"
      return lambda **kw: ("script_loss", kw)

    monkeypatch.setattr(losses_mod.registry, "load_extern", fake_load_extern)
    result = load_loss("https://example.invalid/loss.py", alpha=2)
    assert result == ("script_loss", {"alpha": 2})


class TestUnknownLoss:

  def test_unknown_name_fatal(self):
    with pytest.raises(ValueError) as excinfo:
      load_loss("nope")
    assert "nope" in str(excinfo.value)
    assert "focal" in str(excinfo.value)


class TestArgsHelpers:

  def test_json_dict_type_parses(self):
    assert json_dict_type('{"gamma": 0.5}') == {"gamma": 0.5}
    assert json_dict_type("") == {}
    assert json_dict_type(None) == {}

  def test_json_dict_type_rejects_list(self):
    with pytest.raises(argparse.ArgumentTypeError):
      json_dict_type("[1, 2]")

  def test_build_criterion_script(self):
    path = _write_script("import torch.nn as nn\n"
                         "class MyLoss(nn.Module):\n"
                         "  def forward(self, logits, targets):\n"
                         "    return logits.mean()\n"
                         "def build_loss(**kwargs):\n"
                         "  return MyLoss()\n")
    args = argparse.Namespace(loss=path, loss_args=None)
    method = ClassificationMethod()
    criterion = method.build_criterion(args)
    assert isinstance(criterion, torch.nn.Module)

  def test_build_criterion_default(self):
    args = argparse.Namespace(loss=None, loss_args=None)
    method = ClassificationMethod()
    criterion = method.build_criterion(args, class_weights=torch.ones(3))
    assert isinstance(criterion, CombinedFocalLoss)
