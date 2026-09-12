"""Tests for the BaseTrainer loop mechanics and its concrete trainers.

The loop is exercised through a minimal ``FakeTrainer`` subclass --
exactly the extension point production trainers use -- plus the VO
trainer as a real-consumer integration test.
"""

import collections

import torch

from genml_kit.training.trainer import BaseTrainer, TrainingResult
from genml_kit.training.vo.train_vo import VOTrainer


class _Args:
  epochs = 2
  state_save = "none"  # save model weights only
  checkpoint = "/tmp"
  remote_checkpoint = None
  save_every = 0
  grad_accum_steps = 1
  vo_stage = 0
  grad_monitor = -1  # disabled
  norm_history = 0
  trend_top_n = 0


class _Optim:
  """Minimal optimization stand-in satisfying the saver's attributes."""

  def __init__(self, params):
    self.optimizer = torch.optim.SGD(params, lr=0.01)
    self.scheduler = None
    self.scaler = None


class _Model(torch.nn.Module):

  def __init__(self):
    super().__init__()
    self.lin = torch.nn.Linear(2, 1)

  def forward(self, x):
    return self.lin(x)


class _Writer:

  def __init__(self):
    self.scalars = []

  def add_scalar(self, tag, value, step):
    self.scalars.append((tag, value, step))

  def close(self):
    pass


class _FakeMetrics(collections.namedtuple("_FakeMetrics", ["score"])):
  pass


class FakeTrainer(BaseTrainer):
  """Minimal concrete trainer: counts calls, validates a constant."""

  BEST_METRIC = "score"
  BEST_METRIC_KEY = "best_score"

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.calls = {"epochs": 0, "validations": 0, "epoch_ends": 0}
    self.score = 0.42

  def train_epoch(self, epoch, saver, step, monitor):
    self.calls["epochs"] += 1
    self.epoch = epoch
    return 0.5, step + 1

  def validate(self):
    self.calls["validations"] += 1
    return _FakeMetrics(self.score)

  def epoch_end(self):
    self.calls["epoch_ends"] += 1


def _make(tmp_path, trainer_cls=FakeTrainer, **kwargs):
  args = _Args()
  args.checkpoint = str(tmp_path / "ckpt")
  model = _Model()
  trainer = trainer_cls(args, model, _Optim(model.parameters()),
                        torch.device("cpu"), _Writer(),
                        start_epoch=0, best_metric=0.0, global_step=0,
                        **kwargs)
  return trainer, args


def test_trainer_runs_epochs_and_selects_best(tmp_path):
  trainer, _ = _make(tmp_path)
  result = trainer.run()
  assert isinstance(result, TrainingResult)
  assert trainer.calls["epochs"] == 2
  assert trainer.calls["validations"] == 2
  assert trainer.calls["epoch_ends"] == 2
  assert result.completed_epoch == 1
  assert result.best_metric == 0.42
  best = torch.load(str(tmp_path / "ckpt_best.pt"), weights_only=False)
  assert best["best_score"] == 0.42


def test_validations_run_once_per_epoch_from_loop(tmp_path):
  """The base loop is the single validate() caller per epoch.

  A concrete trainer whose train_epoch also calls validate() would double
  the validation cost; the loop must be the only driver.
  """
  calls = {"train_epoch": 0, "validate": 0}

  class CountingTrainer(BaseTrainer):

    BEST_METRIC = "score"
    BEST_METRIC_KEY = "best_score"

    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

    def train_epoch(self, epoch, saver, step, monitor):
      calls["train_epoch"] += 1
      self.epoch = epoch
      return 0.5, step + 1

    def validate(self):
      calls["validate"] += 1
      return _FakeMetrics(0.42)

  trainer, _ = _make(tmp_path, CountingTrainer)
  trainer.run()
  # Exactly one validation per completed epoch, driven by the loop.
  assert calls["validate"] == 2
  assert calls["train_epoch"] == 2


def test_trainer_without_validate_skips_best_saving(tmp_path):

  class NoValidateTrainer(FakeTrainer):

    def validate(self):
      return None

  trainer, _ = _make(tmp_path, NoValidateTrainer)
  result = trainer.run()
  assert result.best_metric == 0.0
  assert result.completed_epoch == 1
  assert not (tmp_path / "ckpt_best.pt").exists()
  assert (tmp_path / "ckpt_latest.pt").exists()


def test_minimizing_metric_improves_downward(tmp_path):
  """The VO pattern: lower metric wins, init inf means first wins."""

  class MinimizingTrainer(FakeTrainer):

    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.best_metric = float("inf")
      self.scores = [3.2, 1.5, 9.9]  # last epoch must NOT become best

    def has_metric_improved(self, old, new):
      return new < old

    def validate(self):
      self.calls["validations"] += 1
      return _FakeMetrics(self.scores[self.calls["validations"] - 1])

  args = _Args()
  args.epochs = 3
  args.checkpoint = str(tmp_path / "ckpt")
  model = _Model()
  trainer = MinimizingTrainer(args, model, _Optim(model.parameters()),
                              torch.device("cpu"), _Writer(),
                              start_epoch=0, best_metric=float("inf"),
                              global_step=0)
  result = trainer.run()
  assert result.best_metric == 1.5  # 9.9 never displaces the best
  best = torch.load(str(tmp_path / "ckpt_best.pt"), weights_only=False)
  assert best["best_score"] == 1.5


def test_extras_flow_into_checkpoints(tmp_path):
  """saver_extra lands in every checkpoint; ckpt_extra in best/exit saves."""

  class ExtraTrainer(FakeTrainer):

    def saver_extra(self):
      return {"method_state": {"seen_batches": 7}}

    def ckpt_extra(self, best, step):
      return {"blob": f"best={best}@{step}"}

  trainer, _ = _make(tmp_path, ExtraTrainer)
  trainer.run()
  latest = torch.load(str(tmp_path / "ckpt_latest.pt"), weights_only=False)
  assert latest["method_state"] == {"seen_batches": 7}
  best = torch.load(str(tmp_path / "ckpt_best.pt"), weights_only=False)
  # The best save happens right after epoch 0's validation, at step 1.
  assert best["blob"] == "best=0.42@1"


def test_vo_trainer_uses_shared_loop(tmp_path):
  """The VO trainer extends BaseTrainer, not a copy.

  It must return the shared TrainingResult, drive epochs through it,
  and store the best MCE honestly (positive pixels under ``best_mce``,
  not the historical negated convention).
  """
  from genml_kit.datasets.vo_pairs import VOPairDataset
  from genml_kit.models.registry import load_model

  dataset = VOPairDataset(length=4, size=(64, 64), seed=0)
  bundle = load_model("vo/npu-small", num_labels=0, image_size=64)
  model = bundle.model
  loader = torch.utils.data.DataLoader(dataset, batch_size=2)

  class _Loaders:
    pass

  container = _Loaders()
  container.train_loader = loader
  container.val_loader = loader

  args = _Args()
  args.checkpoint = str(tmp_path / "vo_ckpt")
  writer = _Writer()

  trainer = VOTrainer(args,
                      model,
                      container,
                      _Optim(model.parameters()),
                      torch.device("cpu"),
                      writer,
                      start_epoch=0,
                      global_step=0)
  result = trainer.run()
  assert isinstance(result, TrainingResult)
  assert result.completed_epoch == 1
  # Lower-is-better with inf init: the first validation must win and
  # land as a real, positive corner error in the checkpoint.
  assert result.best_metric == trainer.best_metric
  assert 0.0 < result.best_metric < float("inf")
  latest = torch.load(str(tmp_path / "vo_ckpt_latest.pt"), weights_only=False)
  assert 0.0 < latest["best_mce"] < float("inf")
  assert "best_mce_negated" not in latest
  # Validation logging rides the trainer's epoch, not the old -1 sentinel.
  assert any(tag == "VO/mce_val" for tag, _, _ in writer.scalars)
