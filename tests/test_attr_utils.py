"""Tests for genml_kit.attr_utils — safe dotted-attribute access utilities."""

import torch
import torch.nn as nn

from genml_kit.utils.attr import MISSING, get_attribute, maybe_call, maybe_setattr


class TestGetAttribute:

  def test_single_level(self):
    obj = nn.Linear(4, 2)
    assert get_attribute(obj, 'weight') is obj.weight

  def test_dotted_path(self):
    model = nn.Sequential(nn.Linear(4, 2), nn.ReLU())
    assert get_attribute(model, '0.weight') is model[0].weight

  def test_missing_single(self):
    assert get_attribute(nn.Linear(4, 2), 'nonexistent') is MISSING

  def test_missing_dotted(self):
    assert get_attribute(nn.Linear(4, 2), 'foo.bar') is MISSING

  def test_none_value_treated_as_missing(self):
    """An attribute whose value is literally None is treated as MISSING."""

    class Pseudo:
      x = None

    assert get_attribute(Pseudo(), 'x') is MISSING


class TestMaybeCall:

  def test_calls_existing_method(self):
    t = torch.tensor([1.0, 2.0, 3.0])
    result = maybe_call(t, 'tolist')
    assert result == [1.0, 2.0, 3.0]

  def test_returns_missing_for_missing_method(self):
    assert maybe_call(torch.tensor([1.0]), 'nonexistent') is MISSING

  def test_passes_args_and_kwargs(self):
    t = torch.tensor([1.0, 2.0, 3.0])
    result = maybe_call(t, 'unsqueeze', 0)
    assert result.shape == (1, 3)

  def test_deeply_nested(self):
    container = nn.Sequential(nn.Sequential(nn.ReLU()))
    result = maybe_call(container, '0.0.train')
    # train() returns self
    assert result is container[0][0]

  def test_missing_in_middle_of_path(self):
    container = nn.Sequential(nn.ReLU())
    assert maybe_call(container, '0.weight') is MISSING

  def test_non_callable_returns_missing(self):
    t = torch.tensor([1.0])
    assert maybe_call(t, 'shape') is MISSING


class TestMaybeSetattr:

  def test_sets_existing_attr(self):

    class Obj:
      use_grad_checkpoint = False

    obj = Obj()
    maybe_setattr(obj, 'use_grad_checkpoint', True)
    assert obj.use_grad_checkpoint is True

  def test_returns_previous_value(self):

    class Obj:
      use_grad_checkpoint = False

    obj = Obj()
    prev = maybe_setattr(obj, 'use_grad_checkpoint', True)
    assert prev is False

  def test_returns_missing_when_path_absent(self):
    result = maybe_setattr(nn.Linear(4, 2), 'missing.path.value', True)
    assert result is MISSING

  def test_dotted_setattr(self):

    class Inner:
      flag = False

    class Outer:
      inner = Inner()

    obj = Outer()
    maybe_setattr(obj, 'inner.flag', True)
    assert obj.inner.flag is True

  def test_does_not_create_new_attrs(self):
    model = nn.Linear(4, 2)
    maybe_setattr(model, 'brand.new.attr', True)
    assert not hasattr(model, 'brand')


class TestEnableGradCheckpointing:

  def test_custom_model_unwrapped(self):
    from genml_kit.models.convvit.model import CustomPatchTransformer
    from genml_kit.training.model_utils import enable_grad_checkpointing

    model = CustomPatchTransformer(num_classes=2, img_size=32)
    assert model.use_grad_checkpoint is False
    enable_grad_checkpointing(model)
    assert model.use_grad_checkpoint is True

  def test_custom_model_wrapped(self):
    from genml_kit.models.convvit.loader import ConvViTAdapter
    from genml_kit.models.convvit.model import CustomPatchTransformer
    from genml_kit.training.model_utils import enable_grad_checkpointing

    inner = CustomPatchTransformer(num_classes=2, img_size=32)

    class FakeConfig:
      id2label = {0: 'a', 1: 'b'}
      label2id = {'a': 0, 'b': 1}

    wrapped = ConvViTAdapter(inner, FakeConfig())
    assert wrapped.model.use_grad_checkpoint is False
    enable_grad_checkpointing(wrapped)
    assert wrapped.model.use_grad_checkpoint is True

  def test_timm_model(self):
    from unittest.mock import MagicMock

    from genml_kit.training.model_utils import enable_grad_checkpointing

    model = MagicMock()
    model.set_grad_checkpointing = MagicMock()
    # Make it look like a real timm model (no sub-attributes that confuse detection)
    del model.model
    del model.backbone
    del model.gradient_checkpointing_enable
    del model.use_grad_checkpoint

    enable_grad_checkpointing(model)
    model.set_grad_checkpointing.assert_called_once_with(enable=True)

  def test_unknown_model_logs_warning(self, caplog):
    import logging
    from unittest.mock import MagicMock

    from genml_kit.training.model_utils import enable_grad_checkpointing

    model = MagicMock(spec=[])  # empty spec = no real attributes

    with caplog.at_level(logging.WARNING):
      enable_grad_checkpointing(model)

    assert "no known mechanism found" in caplog.text
