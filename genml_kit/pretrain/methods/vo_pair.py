"""VO pair-consistency pre-training method (optional, plan section 12.3).

Self-supervised encoder pre-training for VO through the existing pretrain
harness: the method re-uses the synthetic pair generator and trains the
encoder so that predicted similarities stay consistent with composed GT
transforms.  Registered as a standard ``PretrainMethod``.
"""

from genml_kit.pretrain.methods.base import PretrainMethod
from genml_kit.pretrain.methods.registry import register_method
from genml_kit.training.vo.train_vo import photometric_residual


@register_method
class VOPairMethod(PretrainMethod):
  """Pair-consistency pre-training for the VO front-end.

  train_step receives a collated pair batch (image_a/image_b plus GT
  metadata) and minimizes the photometric consistency of the model's
  own alignment prediction -- no GT similarity is consumed here, which
  is what makes the stage self-supervised.
  """

  NAME = "vo_pair"

  def add_args(self, parser):
    parser.add_argument("--vo_stage",
                        type=int,
                        default=1,
                        help="photometric stage (1 = on)")

  def build_transform(self, args):
    return None

  def build(self, args, encoder, device):
    from genml_kit.models.vo.vo_similar import VOSimilarityConfig, VOSimilarityNet

    model = VOSimilarityNet(
        VOSimilarityConfig(profile=getattr(args, "vo_profile", "npu-small"),
                           in_ch=getattr(args, "in_ch", 1)))
    model.to(device)
    return model

  def train_step(self, model, batch, global_step, *, labels=None):
    image_a = batch["image_a"]
    image_b = batch["image_b"]
    out = model(image_a, image_b)
    residual = photometric_residual(image_a, image_b, out["params"])
    loss = residual.mean()
    return loss, {"vo_photo": loss.detach()}
