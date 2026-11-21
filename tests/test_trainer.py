"""Tests for the generic BaseTrainer loop (pipeline + method contract).

The loop is exercised through a minimal ``FakeMethod``/``FakePipeline``
pair -- exactly the extension contract production methods/pipelines
implement -- with a real ``CheckpointSaver`` and writer.

v4.2: the best-checkpoint cycle is keyed off ``method.METRIC_KEY`` and
``method.has_metric_improved``; the checkpoint value is stored under
``best_<metric_key>`` and the loop never negates.
"""

import types

import torch

from genml_kit.pipelines.contracts import DataBlob, LossOutput, ValidationResult
from genml_kit.training.trainer import BaseTrainer
from tests.logging_helpers import log_capture


class _Args:
  epochs = 2
  cache_dir = None
  # Optimizer/scheduler/AMP states not saved.
  state_save = "none"
  # Set per-test.
  checkpoint = None
  save_every = 0
  remote_checkpoint = None
  grad_accum_steps = 1
  grad_clip = 0.0
  amp_dtype = None
  # EMA flags (mixin-registered on the real parser; off by default).
  ema_decay = 0.0
  ema_final_decay = None
  ema_warmup_steps = 0
  ema_eval = False
  log_every = 50
  vis_every = 0
  norm_history = 0
  grad_monitor = -1
  dist_arg = {}
  max_batches = None
  max_train_time = None


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
    return ValidationResult({"macro_f1": v, "loss": v}, 4.0)

  def has_metric_improved(self, new, best):
    return new > best

  def get_checkpoint_state(self, model, args):
    return self.checkpoint_state

  def load_checkpoint_state(self, model, state, args):
    if "value" in state:
      self._loss = 0.0

  def on_epoch_end(self, model, epoch, logger):
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
    return ValidationResult({"mce": self._loss + 0.02, "loss": self._loss}, 4.0)


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
        [DataBlob(data=data, meta=meta or {"labels": torch.zeros(4)}, size=4)])
    self.val_loader = None

  def to_device(self, blob, device):
    if isinstance(blob.data, (tuple, list)):
      data = tuple(b.to(device) for b in blob.data)
    else:
      data = blob.data.to(device)
    return blob._replace(data=data)


class _Logger:
  """Minimal MetricsLogger stand-in (semantic API + passthrough)."""

  def __init__(self):
    self.scalars = []

  def log_step(self, tag, value, *, step, resolution=None, kind=None):
    self.scalars.append((tag, value, step))

  def log_epoch(self, tag, value, *, epoch, kind=None):
    self.scalars.append((tag, value, epoch))

  def log_artifact(self, tag, payload_kind, payload, *, step=None):
    pass

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
                     logger=_Logger(),
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


def test_new_best_line_keeps_subpixel_precision(tmp_path):
  """The "New best" line must not round a pixel metric to 0.00.

  Real VO mce values sit in the 1e-3..1e-1 range, so the old ``.2f``
  format printed ``0.00 -> 0.00`` every epoch even when the underlying
  value improved materially.  The fake method below therefore returns
  sub-pixel values, matching the production metric's scale.
  """

  class _SubpixelMce(MinimizingFakeMethod):
    """Reports a realistic, monotonically improving sub-pixel mce."""

    def __init__(self):
      super().__init__()
      self._vals = [0.0032, 0.0028, 0.0025]
      self._i = 0

    def evaluate(self, model, loader, device, to_device):
      v = self._vals[min(self._i, len(self._vals) - 1)]
      self._i += 1
      return ValidationResult({"mce": v, "loss": 0.01}, 4.0)

  pipeline = FakePipeline(torch.zeros(4, 3, 8, 8))
  # Validation must run, otherwise the best-metric cycle never fires.
  pipeline.val_loader = _WrappedLoader(
      [DataBlob(data=torch.zeros(4, 3, 8, 8), meta={"labels": torch.zeros(4)}, size=4)])
  trainer = _make_trainer(tmp_path, method=_SubpixelMce(), pipeline=pipeline)
  # Seed the sentinel the fresh-run branch would use, so the first
  # epoch's mce counts as an improvement and the line actually fires.
  trainer.best_metric = float("inf")
  with log_capture(level="INFO") as logs:
    trainer.run()
  new_best = [ln for ln in logs.text.splitlines() if "New best" in ln]
  assert new_best, "expected a 'New best' line"
  body = "\n".join(new_best)
  # The real regression: sub-pixel mce collapsed to a constant.
  assert "0.00 -> 0.00" not in body
  assert "0.0032 -> 0.0028" in body


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


def test_trainer_calls_set_epoch_on_batch_sampler(tmp_path):
  """The epoch index reaches a batch sampler that exposes ``set_epoch``.

  The length-bucketing sampler varies its per-epoch shuffle by epoch index; the
  loop must tell it which epoch it is (a no-op for loaders without one).
  """

  class _Sampler:

    def __init__(self):
      self.epochs = []

    def set_epoch(self, epoch):
      self.epochs.append(epoch)

  class _LoaderWithSampler(_WrappedLoader):

    def __init__(self, batches):
      super().__init__(batches)
      self.batch_sampler = _Sampler()

  pipeline = FakePipeline(torch.zeros(4, 3, 8, 8))
  pipeline.train_loader = _LoaderWithSampler(pipeline.train_loader.batches)
  trainer = _make_trainer(tmp_path, pipeline=pipeline)
  trainer.run()
  # args.epochs == 2 in the fake args.
  assert pipeline.train_loader.batch_sampler.epochs == [0, 1]


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
  args.seeded_pairs = 0
  args.pitch_deg = 45.0
  args.terrain = "field"
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
  args.param_init = None
  args.freeze = ""

  pipeline = VOPairPipeline()
  pipeline.build_loader(args, mode="train")
  pipeline.build_loader(args, mode="val")

  optimization = _Optimization()
  model = method.build_model(args, torch.device("cpu"))
  optimization.optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  logger = _Logger()
  trainer = BaseTrainer(args=args,
                        model=model,
                        method=method,
                        pipeline=pipeline,
                        optimization=optimization,
                        device=torch.device("cpu"),
                        logger=logger,
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


class TestEstimatedBatches:
  """Sample-estimate -> per-rank batch-count conversion (streaming progress)."""

  class _Loader:
    """A length-less loader carrying an optional batch size and dataset."""

    def __init__(self, dataset, batch_size=None):
      self.dataset = dataset
      if batch_size is not None:
        self.batch_size = batch_size

  class _Dataset:

    def __init__(self, count):
      self._count = count

    def num_samples(self):
      return self._count

  def test_converts_samples_to_batches(self):
    from genml_kit.training.trainer import _estimated_batches

    loader = self._Loader(self._Dataset(100), batch_size=8)
    assert _estimated_batches(loader) == 13  # ceil(100 / 8)

  def test_none_without_an_estimate(self):
    from genml_kit.training.trainer import _estimated_batches

    loader = self._Loader(self._Dataset(None), batch_size=8)
    assert _estimated_batches(loader) is None

  def test_none_without_a_batch_size(self):
    # A batch-sampler / packing stream loader has ``batch_size=None`` and no
    # batch size on its dataset; a wrong denominator is worse than none, so the
    # estimate is withheld.
    from genml_kit.training.trainer import _estimated_batches

    loader = self._Loader(self._Dataset(100))
    assert _estimated_batches(loader) is None

  def test_batch_size_falls_back_to_the_dataset(self):
    # A streaming batch pipe reaches the trainer as a ``batch_size=None``
    # loader (it emits whole batches); its per-batch size lives on the dataset,
    # so that is consulted rather than giving up.
    from genml_kit.training.trainer import _estimated_batches

    dataset = self._Dataset(100)
    dataset.batch_size = 4
    loader = self._Loader(dataset)
    assert _estimated_batches(loader) == 25  # ceil(100 / 4)

  def test_divides_across_ranks(self, monkeypatch):
    from genml_kit.training import trainer as trainer_mod

    monkeypatch.setattr(trainer_mod, "world_size", lambda: 4)
    loader = self._Loader(self._Dataset(100), batch_size=5)
    # ceil(100 / 5 / 4) == 5 batches per rank.
    assert trainer_mod._estimated_batches(loader) == 5


class _LengthlessLoader:
  """A streaming loader: no ``__len__`` (``TypeError`` means "unknown")."""

  def __init__(self, batches):
    self.batches = batches
    self.dataset = None

  def __iter__(self):
    return iter(self.batches)

  def __len__(self):
    raise TypeError("streaming loader has no length")


def _capture_total_batches(monkeypatch):
  """Run one trainer epoch and return the reporter's ``total_batches``."""
  captured = {}

  class _SpyReporter:

    def __init__(self, *, total_batches, **kwargs):
      del kwargs
      captured["total_batches"] = total_batches

    def step(self, *args, **kwargs):
      pass

    def summary(self):
      pass

    def epoch_avg_loss(self):
      return types.SimpleNamespace(loss=0.0)

  monkeypatch.setattr("genml_kit.training.trainer.ImageTrainReporting", _SpyReporter)
  return captured


def test_pipeline_batch_estimate_overrides_the_stream_heuristic(tmp_path, monkeypatch):
  """A pipeline that knows its item->batch policy wins over ``samples/batch``.

  This is the packed-stream fix: ``_estimated_batches`` would divide items by
  ``batch_size`` (wrong under ``wrapped`` packing, which counts windows), so
  the pipeline's own estimate must take precedence.
  """
  captured = _capture_total_batches(monkeypatch)
  pipeline = FakePipeline(torch.zeros(4, 3, 8, 8))
  pipeline.train_loader = _LengthlessLoader(pipeline.train_loader.batches)
  pipeline.val_loader = None
  pipeline.estimate_train_batches = lambda: 777
  trainer = _make_trainer(tmp_path, pipeline=pipeline)
  trainer.run()
  assert captured["total_batches"] == 777


def test_exact_length_still_beats_the_pipeline_estimate(tmp_path, monkeypatch):
  """A real ``len(loader)`` (random access) is preferred over any estimate."""
  captured = _capture_total_batches(monkeypatch)
  pipeline = FakePipeline(torch.zeros(4, 3, 8, 8))
  pipeline.val_loader = None
  pipeline.estimate_train_batches = lambda: 777
  trainer = _make_trainer(tmp_path, pipeline=pipeline)
  trainer.run()
  # The fake loader has one batch; its exact length must win.
  assert captured["total_batches"] == len(pipeline.train_loader)


def _multi_batch_pipeline(n_batches):
  """A pipeline whose train loader yields *n_batches* fixed blobs."""
  pipeline = FakePipeline(torch.zeros(4, 3, 8, 8))
  pipeline.train_loader = _WrappedLoader([
      DataBlob(data=torch.zeros(4, 3, 8, 8), meta={"labels": torch.zeros(4)}, size=4)
      for _ in range(n_batches)
  ])
  pipeline.val_loader = None
  return pipeline


def test_max_batches_truncates_every_epoch_but_runs_all_epochs(tmp_path):
  # max_batches is per-epoch: it caps each epoch's inner loop, while the outer
  # loop still runs every epoch.  2 epochs x 2 batches = 4 optimizer steps, the
  # same as the uncapped 2-epoch baseline would give with 2-batch loaders.
  args = _Args()
  args.max_batches = 2
  pipeline = _multi_batch_pipeline(5)
  trainer = _make_trainer(tmp_path, args=args, pipeline=pipeline)
  result = trainer.run()
  assert result.completed_epoch == 1
  assert result.global_step == 4
  assert result.stop_reason == "max_batches"


def test_no_cap_drains_the_loader(tmp_path):
  args = _Args()
  args.max_batches = None
  pipeline = _multi_batch_pipeline(3)
  trainer = _make_trainer(tmp_path, args=args, pipeline=pipeline)
  result = trainer.run()
  # 2 epochs x 3 batches, one optimizer step each (grad_accum_steps == 1).
  assert result.global_step == 6
  assert result.stop_reason == "epochs"


def test_max_train_time_stops_the_run(tmp_path, monkeypatch):
  args = _Args()
  args.epochs = 100
  args.max_train_time = 1.0
  args.max_batches = None
  pipeline = _multi_batch_pipeline(4)
  trainer = _make_trainer(tmp_path, args=args, pipeline=pipeline)

  # Fake clock: each reading advances 0.4s, so the run expires early and the
  # outer loop breaks well before 100 epochs.
  ticks = {"t": 0.0}

  def clock():
    ticks["t"] += 0.4
    return ticks["t"]

  trainer.budget._clock = clock
  result = trainer.run()
  assert result.stop_reason == "max_time"
  assert result.completed_epoch < 99


def test_capped_epoch_still_runs_tail_flush_and_validation(tmp_path):
  # A capped epoch must land its partial accumulation window (tail flush) and
  # still validate, exactly like a loader-exhausted epoch.
  method = FakeMethod()
  args = _Args()
  # 3 batches per epoch with an accumulation window of 2 -> the cap leaves a
  # partial trailing window that only closes on the post-loop tail flush.
  args.grad_accum_steps = 2
  args.max_batches = 3
  pipeline = _multi_batch_pipeline(10)
  trainer = _make_trainer(tmp_path, method=method, args=args, pipeline=pipeline)
  result = trainer.run()
  assert method.epoch_ends == 2
  # 3 micro-batches / window 2 -> ceil = 2 optimizer steps per epoch.
  assert result.global_step == 4


class _LogitsMethod(FakeMethod):
  """Method that publishes model logits so the reporter tracks accuracy."""

  #: 4 samples, class i predicts i in a 4-way head -> all correct.
  LOGITS = torch.eye(4) * 10.0

  def train_step(self, model, blob, global_step, *, labels=None):
    loss = (model.fc.weight**2).mean() + self._loss
    self._loss -= 0.05
    return LossOutput(loss=loss, metrics={"loss": loss.detach()}, logits=self.LOGITS)


def test_method_logits_and_blob_targets_drive_reporting(tmp_path):
  """The loop wires ``LossOutput.logits`` + ``blob.meta['labels']`` to the reporter.

  This is the seam that was previously dead: the trainer fabricated ``logits``
  via ``getattr(loss_out, 'logits', None)`` (always ``None``, so accuracy never
  tracked).  Now the method publishes typed logits, the input-side targets come
  from the blob meta, and ``ImageTrainReporting`` accumulates both.
  """
  batch = DataBlob(data=torch.zeros(4, 3, 8, 8),
                   meta={"labels": torch.arange(4)},
                   size=4)
  pipeline = FakePipeline(torch.zeros(4, 3, 8, 8))
  pipeline.train_loader = _WrappedLoader([batch])
  args = _Args()
  args.log_every = 1
  trainer = _make_trainer(tmp_path,
                          method=_LogitsMethod(),
                          pipeline=pipeline,
                          args=args)
  with log_capture(level="INFO") as logs:
    trainer.run()
  # All four predictions are correct -> the window accuracy is 100%.
  assert "top1=100.00%" in logs.text
  assert "macro_f1=" in logs.text


def test_grad_accum_last_batch_closes_partial_window_in_loop(tmp_path):
  """Known-length epochs close the trailing partial window in-loop.

  5 batches, accum=2 -> windows [2, 2, 1]; the last window closes at batch
  index 4 in-loop, so the post-loop tail flush must not fire and the optimizer
  step count stays ceil(5 / 2) == 3 per epoch (6 over two epochs).
  """
  args = _Args()
  args.grad_accum_steps = 2
  trainer = _make_trainer(tmp_path, pipeline=_multi_batch_pipeline(5), args=args)
  result = trainer.run()
  # 3 steps/epoch x 2 epochs; ``completed_epoch`` is the last epoch index,
  # matching the existing tests' convention.
  assert result.global_step == 6
  assert result.completed_epoch == 1


def test_grad_accum_stream_tail_flush_closes_partial_window(tmp_path):
  """A stream (length-less loader) defers its trailing window to the tail flush.

  accum=3 over 5 batches -> windows [3, 2]; the partial tail window is closed
  by the post-loop flush, so the step count is still ceil(5 / 3) == 2 per epoch
  (4 over two epochs).  Regression for the stream path of the no_sync fix.
  """
  pipeline = FakePipeline(torch.zeros(4, 3, 8, 8))
  pipeline.val_loader = None
  pipeline.train_loader = _LengthlessLoader([
      DataBlob(data=torch.zeros(4, 3, 8, 8), meta={"labels": torch.zeros(4)}, size=4)
      for _ in range(5)
  ])
  args = _Args()
  args.grad_accum_steps = 3
  trainer = _make_trainer(tmp_path, pipeline=pipeline, args=args)
  result = trainer.run()
  # 2 steps/epoch x 2 epochs; ``completed_epoch`` is the last epoch index.
  assert result.global_step == 4
  assert result.completed_epoch == 1
