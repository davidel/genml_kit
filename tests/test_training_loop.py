"""Tests for the shared training-loop skeleton and its adapters."""

import torch

from genml_kit.training.loop import TrainingResult, run_training_loop
from genml_kit.training.vo.train_vo import run_vo_training


class _Args:
  epochs = 2
  state_save = "none"  # save model weights only
  checkpoint = "/tmp"
  remote_checkpoint = None
  save_every = 0
  lora = False
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


def test_loop_runs_epochs_and_selects_best(tmp_path):
  args = _Args()
  args.checkpoint = str(tmp_path / "ckpt")
  model = _Model()
  optim = _Optim(model.parameters())
  calls = {"epochs": 0, "validations": 0}

  def epoch_fn(epoch, saver, step, monitor):
    calls["epochs"] += 1
    return 0.5, step + 1

  def validate_fn():
    calls["validations"] += 1
    return (0.1, 0.42)  # index 1 is the "higher is better" metric

  result = run_training_loop(
      args,
      model,
      optim,
      torch.device("cpu"),
      train_epoch_fn=epoch_fn,
      validate_fn=validate_fn,
      best_index=1,
      best_metric_key="best_macro_f1",
  )
  assert isinstance(result, TrainingResult)
  assert calls["epochs"] == 2
  assert calls["validations"] == 2
  assert result.completed_epoch == 1
  assert result.best_metric == 0.42


def test_loop_without_validate_skips_best_saving(tmp_path):
  args = _Args()
  args.checkpoint = str(tmp_path / "ckpt")
  model = _Model()
  optim = _Optim(model.parameters())
  result = run_training_loop(
      args,
      model,
      optim,
      torch.device("cpu"),
      train_epoch_fn=lambda e, s, g, m: (0.5, g + 1),
      validate_fn=None,
  )
  assert result.best_metric == 0.0
  assert result.completed_epoch == 1


def test_vo_trainer_uses_shared_loop(tmp_path):
  """The VO trainer is a consumer of run_training_loop, not a copy.

  It must return the shared TrainingResult and drive epochs through it.
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

  class _Writer:

    def add_scalar(self, *a, **k):
      pass

    def close(self):
      pass

  result = run_vo_training(args,
                           model,
                           container,
                           _Optim(model.parameters()),
                           torch.device("cpu"),
                           _Writer(),
                           start_epoch=0,
                           global_step=0)
  assert isinstance(result, TrainingResult)
  assert result.completed_epoch == 1
