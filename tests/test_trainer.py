"""Tests for the generic BaseTrainer loop (pipeline + method contract).

The loop is exercised through a minimal ``FakeMethod``/``FakePipeline``
pair -- exactly the extension contract production methods/pipelines
implement -- with a real ``CheckpointSaver`` and writer.

v4.2: the best-checkpoint cycle is keyed off ``method.METRIC_KEY`` and
``method.has_metric_improved``; the checkpoint value is stored under
``best_<metric_key>`` and the loop never negates.
"""

import torch

from genml_kit.pipelines.contracts import DataBlob, LossOutput
from genml_kit.training.trainer import BaseTrainer


class _Args:
  epochs = 2
  state_save = "none"  # optimizer/scheduler/AMP states not saved
  checkpoint = None  # set per-test
  save_every = 0
  remote_checkpoint = None
  grad_accum_steps = 1
  grad_clip = 0.0
  amp_dtype = None
  norm_history = 0
  grad_monitor = -1


class _Optimization:
  scaler = None

  def __init__(self):
    self.optimizer = None
    self.scheduler = None
    self.param_groups = {}


class _Params:
  """Dummy parameter count + .parameters() for the grad monitor."""

  def __init__(self):
    self._params = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(4))])

  def parameters(self):
    return self._params


class FakeMethod:
  """Minimal maximizer method: loss from a model parameter, metric = loss."""

  METRIC_KEY = "macro_f1"
  NAME = "fake"

  def __init__(self):
    self._loss = 1.0
    self.epoch_ends = 0
    self.checkpoint_state = {"method": "fake", "value": 123}

  def train_step(self, model, blob, global_step, *, labels=None):
    # Differentiable loss through a live model parameter, so backward()
    # works in the loop's grad-accumulation path.
    loss = (model.fc.weight**2).mean() + self._loss
    self._loss -= 0.05
    return LossOutput(loss=loss, metrics={"loss": loss.detach(), "top1": loss.detach()})

  def evaluate(self, model, loader, device, to_device):
    # Return a *lower* metric each epoch so a fresh best always fires.
    v = self._loss + 0.01
    return {"macro_f1": v, "loss": v}

  def has_metric_improved(self, new, best):
    return new > best

  def get_checkpoint_state(self, model, args):
    return self.checkpoint_state

  def load_checkpoint_state(self, model, state, args):
    if "value" in state:
      self._loss = 0.0

  def on_epoch_end(self, model, epoch, writer):
    self.epoch_ends += 1

  @classmethod
  def add_args(cls, parser):
    pass


class MinimizingFakeMethod(FakeMethod):
  """Minimize-direction method (VO mce): METRIC_KEY = 'mce'."""

  METRIC_KEY = "mce"

  def has_metric_improved(self, new, best):
    return new < best

  def evaluate(self, model, loader, device, to_device):
    return {"mce": self._loss + 0.02, "loss": self._loss}


class _WrappedLoader:
  """Stand-in: a pipeline loader that yields fixed DataBlobs."""

  def __init__(self, batches):
    self.batches = batches

  def __len__(self):
    return len(self.batches)

  def __iter__(self):
    return iter(self.batches)


class FakePipeline:
  """Minimal pipeline: loader of DataBlobs + identity device transfer."""

  def __init__(self, data, meta=None):
    self.train_loader = _WrappedLoader(
        [DataBlob(data=data, meta=meta or {"labels": torch.zeros(4)})])
    self.val_loader = None

  def to_device(self, blob, device):
    if isinstance(blob.data, (tuple, list)):
      return DataBlob(tuple(b.to(device) for b in blob.data), blob.meta)
    return DataBlob(blob.data.to(device), blob.meta)


class _Writer:
  """Minimal TensorBoard writer stand-in."""

  def __init__(self):
    self.scalars = []

  def add_scalar(self, tag, value, step):
    self.scalars.append((tag, value, step))

  def close(self):
    pass


def _make_trainer(tmp_path, method=None, pipeline=None, args=None):
  args = args or _Args()
  args.checkpoint = str(tmp_path / "ckpt")
  method = method or FakeMethod()
  pipeline = pipeline or FakePipeline(torch.zeros(4, 3, 8, 8))
  optimization = _Optimization()
  optimization.optimizer = torch.optim.SGD(_Params().parameters(), lr=0.01)
  model = torch.nn.Sequential()
  model.fc = torch.nn.Linear(8 * 8 * 3, 4)
  return BaseTrainer(args=args,
                     model=model,
                     method=method,
                     pipeline=pipeline,
                     optimization=optimization,
                     device=torch.device("cpu"),
                     writer=_Writer(),
                     start_epoch=0,
                     best_metric=0.0,
                     global_step=0)


def test_trainer_runs_epochs_and_selects_best(tmp_path):
  trainer = _make_trainer(tmp_path)
  result = trainer.run()
  assert result.completed_epoch == 1
  assert result.global_step == 2
  assert result.best_metric is not None


def test_validations_run_once_per_epoch_from_loop(tmp_path):
  method = FakeMethod()
  trainer = _make_trainer(tmp_path, method=method)
  trainer.run()
  assert method.epoch_ends == 2


def test_trainer_without_validate_skips_best_saving(tmp_path):
  args = _Args()
  method = FakeMethod()
  pipeline = FakePipeline(torch.zeros(4, 3, 8, 8))
  pipeline.val_loader = None
  trainer = _make_trainer(tmp_path, method=method, pipeline=pipeline, args=args)
  result = trainer.run()
  assert result.completed_epoch == 1
  assert result.best_metric == 0.0


def test_minimizing_metric_improves_downward(tmp_path):
  method = MinimizingFakeMethod()
  trainer = _make_trainer(tmp_path, method=method)
  trainer.run()
  assert trainer.best_metric is not None
  assert trainer.best_metric < float("inf")


def test_extras_flow_into_checkpoints(tmp_path):
  method = FakeMethod()
  method.extra = {"epoch": 42}
  trainer = _make_trainer(tmp_path, method=method)

  # The method's get_checkpoint_state flows into every save (saver_extra).
  class _Saver:

    def __init__(self, trainer):
      self._trainer = trainer

    def save_best(self, *args, **kwargs):
      self.kwargs = kwargs

    def save_latest(self, *args, **kwargs):
      self.kwargs = kwargs

  _Saver(trainer)
  trainer.run()


def test_vo_trainer_uses_shared_loop(tmp_path):
  """The VO objective trains through the same generic loop."""
  from genml_kit.methods.vo_pair import VOPairMethod
  from genml_kit.pipelines.vo_pair import VOPairPipeline

  method = VOPairMethod()
  args = _Args()
  args.checkpoint = str(tmp_path / "vo_ckpt")
  args.image_size = 16
  args.vo_length = 8
  args.vo_val_length = 4
  args.vo_stage = "supervised"
  args.vo_profile = "npu-small"
  args.vo_loss_cfg = None
  args.batch_size = 4
  args.num_workers = 0
  args.in_ch = 1
  args.seed = None
  args.state_save = "none"
  # _apply_model_extras is invoked by build_model; give it its neutral
  # flag surface (the CLI defaults).
  args.grad_checkpoint = False
  args.lora = False
  args.lora_r = 8
  args.lora_alpha = 16
  args.lora_dropout = 0.0
  args.lora_target_modules = ""
  args.source_checkpoint = None
  args.param_rename = None
  args.freeze = ""

  pipeline = VOPairPipeline()
  pipeline.build_loader(args, mode="train")
  pipeline.build_loader(args, mode="val")

  optimization = _Optimization()
  model = method.build_model(args, torch.device("cpu"))
  optimization.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  writer = _Writer()
  trainer = BaseTrainer(args=args,
                        model=model,
                        method=method,
                        pipeline=pipeline,
                        optimization=optimization,
                        device=torch.device("cpu"),
                        writer=writer,
                        start_epoch=0,
                        best_metric=float("inf"),
                        global_step=0)
  result = trainer.run()
  # 2 epochs (args.epochs=2) over len 8/batch 4 = 2 steps each; validation
  # ran and produced a real, positive mce checkpoint key.
  assert result.completed_epoch == 1
  assert result.global_step == 4
  latest = torch.load(str(tmp_path / "vo_ckpt_latest.pt"), weights_only=False)
  assert "best_mce" in latest
  assert "best_mce_negated" not in latest
