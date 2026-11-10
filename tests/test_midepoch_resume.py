"""Mid-epoch resume (``--save_every``) for the supervised trainer.

Covers the behaviour restored/added on top of ``ec19359``:

* a periodic checkpoint is written every ``--save_every`` optimizer steps;
* it carries an ``in_progress`` marker (epoch + micro-step) and ``world_size``;
* a resume into that marker continues the epoch (same sample order as an
  uninterrupted run), fast-forwarded by a ``grad_accum_steps``-aligned amount;
* legacy checkpoints (no marker) resume at the next epoch boundary.

The loop is driven through a minimal ``FakeMethod``/``FakePipeline`` pair and a
recording loader that tags each batch with its index, so the resumed order can
be compared against the uninterrupted run.
"""

import contextlib

import torch

from genml_kit.io.checkpointing import resume_checkpoint
from genml_kit.pipelines.contracts import DataBlob, LossOutput
from genml_kit.training.trainer import BaseTrainer


class _Args:

  def __init__(self, epochs=1, save_every=0, grad_accum_steps=1):
    self.epochs = epochs
    self.cache_dir = None
    self.state_save = "none"
    self.checkpoint = None
    self.save_every = save_every
    self.remote_checkpoint = None
    self.grad_accum_steps = grad_accum_steps
    self.grad_clip = 0.0
    self.amp_dtype = None
    self.log_every = 50
    self.vis_every = 0
    self.norm_history = 0
    self.grad_monitor = -1
    self.dist_arg = {}


class _Optimization:
  scaler = None

  def __init__(self):
    self.optimizer = None
    self.scheduler = None


class _Params:
  """Dummy parameter holder for the optimizer + grad monitor."""

  def __init__(self):
    self._params = torch.nn.ParameterList([torch.nn.Parameter(torch.zeros(4))])

  def parameters(self):
    return self._params


class _RecordingLoader:
  """Loader that yields one DataBlob per index and records the drawn order."""

  def __init__(self, tags):
    self.tags = tags
    self.drawn = []

  def __len__(self):
    return len(self.tags)

  def __iter__(self):
    for tag in self.tags:
      self.drawn.append(tag)
      yield DataBlob(data=torch.zeros(4, 3, 8, 8),
                     meta={
                         "labels": torch.zeros(4),
                         "tag": tag
                     },
                     size=4)


class _RecordingMethod:
  """Records the batch tags it saw, in order; trivial differentiable loss."""

  METRIC_KEY = "macro_f1"
  NAME = "recording"

  def __init__(self, tags, interrupt_at=None):
    self._tags = tags
    self.seen = []
    self.completed = 0
    self._interrupt_at = interrupt_at
    self._count = 0

  def train_step(self, model, blob, global_step, *, labels=None):
    tag = blob.meta.get("tag") if isinstance(blob.meta, dict) else None
    self.seen.append(tag)
    self._count += 1
    # Raise *before* producing a loss so no flush/update happens for this
    # batch: it models an interrupt landing mid-iteration.
    if self._interrupt_at is not None and self._count == self._interrupt_at:
      raise KeyboardInterrupt
    loss = (model.fc.weight**2).mean()
    self.completed += 1
    return LossOutput(loss=loss, metrics={"loss": loss.detach()})

  def evaluate(self, model, loader, device, to_device):
    from genml_kit.pipelines.contracts import ValidationResult
    return ValidationResult({"macro_f1": 0.0, "loss": 0.0}, 4.0)

  def has_metric_improved(self, new, best):
    return new > best

  def get_checkpoint_state(self, model, args):
    return {}

  def load_checkpoint_state(self, model, state, args):
    pass

  def on_epoch_end(self, model, epoch, logger):
    pass


class _Pipeline:
  """Minimal pipeline exposing the recording loader."""

  def __init__(self, loader):
    self.train_loader = loader
    self.val_loader = None

  def to_device(self, blob, device):
    return blob


class _Logger:

  def log_step(self, *a, **k):
    pass

  def log_epoch(self, *a, **k):
    pass

  def log_artifact(self, *a, **k):
    pass

  def add_scalar(self, *a, **k):
    pass

  def close(self):
    pass


def _make_trainer(tmp_path, method, loader, args, start_epoch=0, resume_progress=None):
  args.checkpoint = str(tmp_path / "ckpt")
  optimization = _Optimization()
  optimization.optimizer = torch.optim.SGD(_Params().parameters(), lr=0.01)
  model = torch.nn.Sequential()
  model.fc = torch.nn.Linear(8 * 8 * 3, 4)
  return BaseTrainer(args=args,
                     model=model,
                     method=method,
                     pipeline=_Pipeline(loader),
                     optimization=optimization,
                     device=torch.device("cpu"),
                     logger=_Logger(),
                     start_epoch=start_epoch,
                     best_metric=0.0,
                     global_step=0,
                     resume_progress=resume_progress)


def test_periodic_save_records_in_progress_marker(tmp_path):
  tags = [f"b{i}" for i in range(6)]
  # Interrupt mid-epoch (a clean run's exit save overwrites the periodic file,
  # which is exactly why the periodic save exists: crash safety).
  args = _Args(epochs=1, save_every=2)
  loader = _RecordingLoader(tags)
  method = _RecordingMethod(tags, interrupt_at=4)
  trainer = _make_trainer(tmp_path, method, loader, args)
  with contextlib.suppress(KeyboardInterrupt):
    trainer.run()

  ckpt = torch.load(str(tmp_path / "ckpt_latest.pt"), weights_only=False)
  # ``world_size`` is always recorded (guard for the mismatch warning).
  assert ckpt["world_size"] == 1
  # The interrupt save carries the mid-epoch marker at the last flush boundary,
  # i.e. exactly the batches that completed.
  assert ckpt["in_progress"]["epoch"] == 0
  assert ckpt["in_progress"]["micro_step"] == method.completed
  assert method.completed == 3


def test_resume_continues_same_order_single_epoch(tmp_path):
  tags = [f"b{i}" for i in range(8)]
  # Uninterrupted reference run: capture the full draw order.
  ref = _RecordingMethod(tags)
  _make_trainer(tmp_path, ref, _RecordingLoader(tags), _Args(epochs=1)).run()

  # Interrupted run: save_every=2 writes markers; interrupt while drawing #4.
  method = _RecordingMethod(tags, interrupt_at=4)
  args = _Args(epochs=1, save_every=2)
  interrupted = _make_trainer(tmp_path, method, _RecordingLoader(tags), args)
  with contextlib.suppress(KeyboardInterrupt):
    interrupted.run()

  ckpt = torch.load(str(tmp_path / "ckpt_latest.pt"), weights_only=False)
  in_progress = ckpt.get("in_progress")
  assert in_progress is not None
  # Marker points to the last *flushed* boundary (grad_accum=1 -> step 3 or 4).
  assert in_progress["epoch"] == 0
  assert in_progress["micro_step"] >= 1

  _, start_epoch, _, extra = resume_checkpoint(str(tmp_path / "ckpt_latest.pt"),
                                               "unused.pt", torch.nn.Sequential(),
                                               "cpu")
  # Resume into the interrupted epoch, not the next one.
  assert start_epoch == 0

  resumed = _RecordingMethod(tags)
  trainer = _make_trainer(tmp_path,
                          resumed,
                          _RecordingLoader(tags),
                          _Args(epochs=1),
                          start_epoch=start_epoch,
                          resume_progress=extra.get("in_progress"))
  trainer.run()

  # The resumed run drew the tail of the reference order (fast-forwarded).
  skip = extra["in_progress"]["micro_step"]
  assert resumed.seen == ref.seen[skip:], (resumed.seen, ref.seen, skip)


def test_grad_accum_rounds_skip_down(tmp_path):
  tags = [f"b{i}" for i in range(8)]

  def _skip(micro_step, marker_epoch=0, start_epoch=0):
    trainer = _make_trainer(tmp_path,
                            _RecordingMethod(tags),
                            _RecordingLoader(tags),
                            _Args(epochs=1, grad_accum_steps=2),
                            start_epoch=start_epoch,
                            resume_progress={
                                "epoch": marker_epoch,
                                "micro_step": micro_step
                            })
    return trainer._resume_skip(start_epoch)

  # grad_accum=2: a mid-window marker rounds down to the window edge.
  assert _skip(3) == 2
  assert _skip(5) == 4
  assert _skip(1) == 0
  # A marker for a later epoch never skips the first epoch.
  assert _skip(6, marker_epoch=3) == 0


def test_legacy_checkpoint_resumes_next_epoch(tmp_path):
  tags = [f"b{i}" for i in range(4)]
  args = _Args(epochs=2)
  method = _RecordingMethod(tags)
  trainer = _make_trainer(tmp_path, method, _RecordingLoader(tags), args)
  trainer.run()
  ckpt = torch.load(str(tmp_path / "ckpt_latest.pt"), weights_only=False)
  # A clean, fully-completed run leaves no in-progress marker.
  assert ckpt.get("in_progress") is None
  assert ckpt["epoch"] == 1
