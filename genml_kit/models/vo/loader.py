"""VO similarity models registered under ``vo/<profile>`` in the registry.

Registration happens at import (see ``genml_kit.models.__init__``), which
makes ``load_model("vo/npu-small", ...)`` and friends work through the
standard registry path.
"""

import collections

import torch

from genml_kit.models.registry import register_model
from genml_kit.models.vo.processor import VOProcessor
from genml_kit.models.vo.vo_similar import VOSimilarityConfig, VOSimilarityNet

VOModelBundle = collections.namedtuple("VOModelBundle", ["model", "processor"])


@register_model("vo/npu-small")
def load_vo_npu_small(*, image_size=256, in_ch=1, cost_range=6, **kwargs):
  return _load_vo(
      VOSimilarityConfig(profile="npu-small",
                         in_ch=in_ch,
                         cost_range=cost_range,
                         cost_scale=8), image_size)


@register_model("vo/edge-mid")
def load_vo_edge_mid(*, image_size=256, in_ch=1, cost_range=6, **kwargs):
  return _load_vo(
      VOSimilarityConfig(profile="edge-mid",
                         in_ch=in_ch,
                         cost_range=cost_range,
                         cost_scale=8), image_size)


@register_model("vo/station")
def load_vo_station(*, image_size=256, in_ch=3, cost_range=6, **kwargs):
  return _load_vo(
      VOSimilarityConfig(profile="station",
                         in_ch=in_ch,
                         cost_range=cost_range,
                         cost_scale=8), image_size)


def _load_vo(config, image_size, device="cpu"):
  """Instantiate the network and probe it once so Lazy modules resolve.

  ``num_labels``/``id2label``/``label2id``/``checkpoint_path`` arrive from
  the registry's uniform call signature and are irrelevant here (VO is
  not a classifier); they are consumed via ``kwargs`` upstream.
  """
  del image_size
  model = VOSimilarityNet(config)
  model(torch.zeros(1, config.in_ch, 64, 64), torch.zeros(1, config.in_ch, 64, 64))
  model.to(device)
  return VOModelBundle(model, VOProcessor())
