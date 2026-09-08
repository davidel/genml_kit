"""BYOL: Bootstrap Your Own Latent (Grill et al., NeurIPS 2020)."""

import torch
from torchvision.transforms import v2

from genml_kit.models.byol import BYOL
from genml_kit.pretrain.augmentations.dual_view import DualViewTransform
from genml_kit.pretrain.methods.base import PretrainMethod
from genml_kit.pretrain.methods.registry import register_method
from genml_kit.training.model_utils import set_train_mode


@register_method
class BYOLMethod(PretrainMethod):
  """Self-supervised contrastive pre-training via BYOL."""

  NAME = "byol"
  needs_labels = False

  @classmethod
  def add_args(cls, parser):
    g = parser.add_argument_group("BYOL")
    g.add_argument("--byol_proj_dim",
                   type=int,
                   default=256,
                   help="Projection head output dimension.")
    g.add_argument("--byol_proj_hidden",
                   type=int,
                   default=2048,
                   help="Projection head hidden dimension.")
    g.add_argument("--byol_predictor_hidden",
                   type=int,
                   default=2048,
                   help="Predictor MLP hidden dimension.")
    g.add_argument("--byol_momentum",
                   type=float,
                   default=0.996,
                   help="Initial EMA momentum for target encoder.")
    g.add_argument("--byol_final_momentum",
                   type=float,
                   default=1.0,
                   help="Final EMA momentum (ramped up over training).")

  def build_transform(self, image_size):
    base = v2.Compose([
        v2.Resize(image_size, interpolation=v2.InterpolationMode.BICUBIC),
        v2.CenterCrop(image_size),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomApply([
            v2.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1),
        ],
                       p=0.8),
        v2.RandomGrayscale(p=0.2),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
    ])
    return DualViewTransform(base)

  def build(self, args, encoder, device):
    return BYOL(
        encoder,
        proj_dim=args.byol_proj_dim,
        proj_hidden=args.byol_proj_hidden,
        predictor_hidden=args.byol_predictor_hidden,
    ).to(device)

  def train_step(self, model, images, global_step, *, labels=None):
    set_train_mode(model, "train")
    loss, info = model(images)
    momentum = self._current_momentum(global_step, model)
    model.update_momentum(momentum)
    return loss, info

  def _current_momentum(self, global_step, model):
    total = getattr(model, "_total_steps", 0)
    if total <= 0:
      return self._momentum_end()
    ratio = min(global_step / total, 1.0)
    start = self._momentum_start()
    end = self._momentum_end()
    return end + (start - end) * (1.0 - ratio)

  def _momentum_start(self):
    return getattr(self, "_byol_momentum", 0.996)

  def _momentum_end(self):
    return getattr(self, "_byol_final_momentum", 1.0)

  def on_epoch_end(self, model, epoch, writer):
    pass

  def get_checkpoint_state(self, model, args):
    return {
        "method": "byol",
        "momentum": self._momentum_start(),
        "final_momentum": self._momentum_end(),
    }

  def load_checkpoint_state(self, model, state, args):
    self._byol_momentum = state.get("momentum", args.byol_momentum)
    self._byol_final_momentum = state.get("final_momentum", args.byol_final_momentum)

  def validate(self, model, images, num_samples):
    return None
