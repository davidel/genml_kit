"""Tests asserting ``cache_dir`` is forwarded through the registry.

The registry's ``load_model`` / ``load_processor`` accept ``cache_dir`` and
must pass it to the underlying backend.  For the HuggingFace paths this is
a ``from_pretrained(..., cache_dir=...)`` call; for registered custom
loaders the value must survive the registry dispatch.  These tests pin the
forwarding behaviour for ``timm`` and ``cls_model_wrapper`` without any
network access.
"""

from unittest.mock import MagicMock, patch

from genml_kit.models import load_model, load_processor


class TestTimmCacheDir:
  """``timm.create_model`` must receive the caller's ``cache_dir``."""

  def test_model_forwards_cache_dir(self):
    fake = MagicMock()
    with patch("timm.create_model", return_value=fake) as create:
      load_model(
          "timm:resnet18",
          num_labels=3,
          cache_dir="/tmp/hf-cache",
          pretrained=False,
      )
    assert create.call_args.kwargs.get("cache_dir") == "/tmp/hf-cache"

  def test_processor_forwards_cache_dir(self):
    import timm

    # ``wraps`` runs the real create_model (pretrained=False, no download)
    # so ``pretrained_cfg`` is populated for resolve_data_config, while
    # still letting us inspect the forwarded kwargs.
    with patch("timm.create_model", wraps=timm.create_model) as create:
      load_processor("timm:resnet18", image_size=224, cache_dir="/tmp/hf-cache")
    assert create.call_args.kwargs.get("cache_dir") == "/tmp/hf-cache"


class TestClsModelWrapperCacheDir:
  """``ClsModelWrapper`` must forward ``cache_dir`` to the backbone load."""

  def test_model_forwards_cache_dir(self):
    from transformers import ViTConfig, ViTModel

    config = ViTConfig(
        image_size=224,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=128,
        num_labels=3,
    )
    tiny_backbone = ViTModel(config)

    with patch(
        "genml_kit.models.cls_model_wrapper.model.AutoModel.from_pretrained",
        return_value=tiny_backbone,
    ) as from_pretrained:
      load_model(
          "cls_model_wrapper:fake-backbone",
          num_labels=3,
          classifier="mlp",
          cache_dir="/tmp/hf-cache",
      )
    assert from_pretrained.call_args.kwargs.get("cache_dir") == "/tmp/hf-cache"


class TestRegistryCustomLoaderCacheDir:
  """A registered custom loader receives ``cache_dir`` from the registry."""

  def test_custom_loader_receives_cache_dir(self):
    from genml_kit.models.registry import MODELS

    captured = {}

    @MODELS.register("cache_dir_probe")
    def _probe(*, cache_dir=None, **kwargs):
      captured["cache_dir"] = cache_dir
      return MagicMock()

    try:
      load_model(
          "cache_dir_probe",
          num_labels=3,
          cache_dir="/tmp/hf-cache",
      )
    finally:
      MODELS.unregister("cache_dir_probe")

    assert captured["cache_dir"] == "/tmp/hf-cache"
