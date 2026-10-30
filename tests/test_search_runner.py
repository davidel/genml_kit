"""Integration tests: the subprocess runner against the fake trainer.

These exercise the real isolation boundary -- a genuine ``python
tests/fake_trainer.py`` subprocess, its ``metrics.jsonl`` read back by the
runner -- plus pruning, retries, resume, the scheduler and the CLI wiring.
No torch training code is touched.
"""

import os

import pytest

from genml_kit.cli import tune
from genml_kit.search.runner import (GpuPool, build_trial_argv, run_study)
from genml_kit.search.store import StudyStore
from genml_kit.search.study import StudySpec
from genml_kit.search.trial import (STATUS_COMPLETED, STATUS_FAILED, STATUS_PRUNED,
                                    Trial)

FAKE_TRAINER = os.path.join(os.path.dirname(__file__), "fake_trainer.py")


def _spec(**overrides):
  data = {
      "objective": "val/macro_f1",
      "direction": "max",
      "sampler": "grid",
      "pruner": "none",
      "space": {
          "x": "0.1,0.2,0.3"
      },
      "max_trials": 3,
      "max_epochs_per_trial": 2,
  }
  data.update(overrides)
  return StudySpec(data)


def _run(tmp_path, spec, extra_argv=None, **kwargs):
  store = StudyStore.create(str(tmp_path / "study"), spec.to_dict())
  run_study(spec,
            store,
            extra_argv or [],
            train_module=FAKE_TRAINER,
            poll_interval=0.05,
            **kwargs)
  return store


class TestBuildTrialArgv:

  def test_bookkeeping_flags_present(self, tmp_path):
    spec = _spec()
    trial = Trial(trial_id=0, config={"x": 0.2}, seed=42)
    trial.trial_dir = str(tmp_path / "t0")
    argv = build_trial_argv(spec, trial, ["--base"])
    assert "--base" in argv
    assert "--log_dir" in argv
    assert "--metrics_sinks" in argv and "jsonl" in argv
    assert "--epochs" in argv and "2" in argv
    assert "--seed" in argv and "42" in argv
    assert "--x" in argv and "0.2" in argv


class TestSequentialRun:

  def test_completes_all_trials(self, tmp_path):
    store = _run(tmp_path, _spec())
    assert len(store.trials) == 3
    assert all(t.status == STATUS_COMPLETED for t in store.trials.values())

  def test_finds_the_best_config(self, tmp_path):
    store = _run(tmp_path, _spec())
    best = store.leaderboard(top=1)[0]
    assert best.config["x"] == 0.2

  def test_metrics_recorded_per_trial(self, tmp_path):
    store = _run(tmp_path, _spec())
    for trial in store.trials.values():
      assert trial.series  # at least one epoch observed
      assert trial.best is not None


class TestPruning:

  def test_asha_prunes_laggards(self, tmp_path):
    spec = _spec(sampler="random",
                 pruner="asha",
                 max_trials=6,
                 pruner_args={
                     "min_resource": 1,
                     "reduction_factor": 2
                 })
    store = _run(tmp_path, spec, extra_argv=["--epoch_sleep", "0.5"])
    statuses = [t.status for t in store.trials.values()]
    assert STATUS_PRUNED in statuses

  def test_pruned_trial_still_has_a_score(self, tmp_path):
    spec = _spec(sampler="random",
                 pruner="asha",
                 max_trials=6,
                 pruner_args={
                     "min_resource": 1,
                     "reduction_factor": 2
                 })
    store = _run(tmp_path, spec, extra_argv=["--epoch_sleep", "0.5"])
    for trial in store.trials.values():
      if trial.status == STATUS_PRUNED:
        assert trial.best is not None


class TestFailureAndRetries:

  def test_nonzero_exit_is_failed(self, tmp_path):
    spec = _spec(max_trials=1, retries=0)
    store = _run(tmp_path, spec, extra_argv=["--fail"])
    trial = store.trials[0]
    assert trial.status == STATUS_FAILED
    assert trial.exit_code == 1

  def test_retries_then_fails(self, tmp_path):
    spec = _spec(max_trials=1, retries=1)
    store = _run(tmp_path, spec, extra_argv=["--fail"])
    assert store.trials[0].status == STATUS_FAILED


class TestParallelRun:

  def test_workers_run_all_trials(self, tmp_path):
    store = _run(tmp_path, _spec(max_trials=3), workers=2)
    assert len(store.trials) == 3
    assert all(t.status == STATUS_COMPLETED for t in store.trials.values())


class TestResume:

  def test_resume_skips_completed_trials(self, tmp_path):
    study_dir = str(tmp_path / "study")
    store = StudyStore.create(study_dir, _spec().to_dict())
    run_study(_spec(max_trials=2),
              store, [],
              train_module=FAKE_TRAINER,
              poll_interval=0.05)
    assert len(store.trials) == 2
    # Resume: the budget is already met, so nothing new runs.
    resumed = StudyStore.open(study_dir)
    run_study(_spec(max_trials=2),
              resumed, [],
              train_module=FAKE_TRAINER,
              poll_interval=0.05)
    assert len(resumed.trials) == 2


class TestGpuPool:

  def test_acquire_release_roundtrip(self):
    pool = GpuPool([0, 1])
    gpus = {pool.acquire(), pool.acquire()}
    assert gpus == {0, 1}
    for gpu in gpus:
      pool.release(gpu)
    assert {pool.acquire(), pool.acquire()} == {0, 1}

  def test_defaults_to_single_none_slot(self):
    pool = GpuPool()
    assert pool.acquire() is None


class TestTuneCli:

  def test_run_parser_wires_forwarded_args(self):
    args = tune.parse_args([
        "run",
        "--space",
        "x=0.1,0.2",
        "--objective",
        "val/macro_f1",
        "--",
        "--pipeline",
        "images",
    ])
    assert args.command == "run"
    assert args.train_extra == ["--pipeline", "images"]
    argv = tune._train_argv(args)
    assert argv == ["--pipeline", "images"]

  def test_build_spec_from_cli(self):
    args = tune.parse_args([
        "run",
        "--space",
        "opt.lr=1e-4,1e-3",
        "--objective",
        "val/macro_f1",
        "--sampler",
        "grid",
        "--max_trials",
        "2",
        "--min_resource",
        "1",
    ])
    spec = tune._build_spec(args)
    assert spec.space.names == ["opt.lr"]
    assert spec.sampler == "grid"
    assert spec.max_trials == 2
    assert spec.pruner_args["min_resource"] == 1

  def test_cli_end_to_end(self, tmp_path):
    study_dir = str(tmp_path / "study")
    tune.main([
        "run",
        "--space",
        "x=0.1,0.2,0.3",
        "--objective",
        "val/macro_f1",
        "--sampler",
        "grid",
        "--pruner",
        "none",
        "--max_trials",
        "3",
        "--epochs",
        "1",
        "--study_dir",
        study_dir,
        "--python",
        os.environ.get("PYTHON", "python3"),
        "--train-module",
        FAKE_TRAINER,
    ])
    store = StudyStore.open(study_dir)
    assert len(store.trials) == 3

  def test_report_subcommand(self, tmp_path):
    study_dir = str(tmp_path / "study")
    store = StudyStore.create(study_dir, {"objective": "x", "direction": "max"})
    store.record(Trial(trial_id=0, config={"a": 1}, best=0.5))
    # Must not raise; prints the leaderboard.
    tune.main(["report", study_dir, "--top", "1"])


@pytest.mark.parametrize("tag", ["val/macro_f1", "epoch/loss"])
def test_objective_tag_is_configurable(tmp_path, tag):
  store = _run(tmp_path, _spec(objective=tag))
  assert all(t.best is not None for t in store.trials.values())
