"""Trial execution: subprocess isolation and the study scheduler.

The tuner never imports training internals.  Each trial is a fresh
``python -m genml_kit.cli.train ...`` subprocess whose only contract with the
tuner is (a) the argv it is handed and (b) the ``metrics.jsonl`` file its
JSONL sink already writes.  This keeps the search subsystem fully decoupled
from the training code -- the isolation boundary *is* the process boundary.

:class:`StudyRunner` is the scheduler: it asks the sampler for configs, runs
them (single process by default; ``workers > 1`` for a thread pool over
subprocesses, each pinned to a GPU from the pool), tails their metrics, prunes
hopeless trials via the pruner, and records every state change in the store.
"""

import logging
import os
import queue
import subprocess
import sys
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from genml_kit.search.errors import SearchSpaceExhausted
from genml_kit.search.pruners import build_pruner
from genml_kit.search.samplers import build_sampler
from genml_kit.search.trial import (STATUS_COMPLETED, STATUS_FAILED, STATUS_PRUNED,
                                    STATUS_RUNNING, Trial, read_objective_series)
from genml_kit.utils.logging import fatal

DEFAULT_TRAIN_MODULE = "genml_kit.cli.train"
_POLL_INTERVAL = 1.0
_TERMINATE_GRACE = 10.0


class GpuPool:
  """A blocking pool of ``CUDA_VISIBLE_DEVICES`` slots for trial pinning."""

  def __init__(self, gpus=None):
    self._gpus = list(gpus) if gpus else [None]
    self._queue = queue.Queue()
    for gpu in self._gpus:
      self._queue.put(gpu)

  def acquire(self):
    """Block until a slot is free; return the device id (or ``None``)."""
    return self._queue.get()

  def release(self, gpu):
    """Return a previously acquired slot to the pool."""
    self._queue.put(gpu)


def build_trial_argv(spec, trial, base_argv):
  """Compose the full ``genml-kit-train`` argv for *trial*.

  Order matters: fixed ``base_args`` first, then the sampled overrides, then
  the bookkeeping flags.  Later flags win under argparse, so the search
  subsystem always controls ``--log_dir``/``--checkpoint``/``--seed``/
  ``--epochs`` even if the user's ``base_args`` set them.
  """
  argv = list(base_argv)
  argv.extend(spec.space.decode_to_argv(trial.config))
  argv += [
      "--log_dir",
      trial.trial_dir,
      "--checkpoint",
      os.path.join(trial.trial_dir, "ckpt"),
      "--metrics_sinks",
      "jsonl",
      "--seed",
      str(trial.seed),
  ]
  if spec.max_epochs_per_trial > 0:
    argv += ["--epochs", str(spec.max_epochs_per_trial)]
  return argv


class SubprocessTrialRunner:
  """Runs one trial as an isolated subprocess, pruning it if asked."""

  def __init__(self,
               spec,
               base_argv,
               python=None,
               train_module=DEFAULT_TRAIN_MODULE,
               gpu_pool=None,
               poll_interval=_POLL_INTERVAL):
    self.spec = spec
    self.base_argv = base_argv
    self.python = python or sys.executable
    self.train_module = train_module
    self.gpu_pool = gpu_pool or GpuPool()
    self.poll_interval = poll_interval

  def run(self, trial, pruner):
    """Execute *trial*, updating it in place; return the finished trial."""
    attempt = 0
    while True:
      gpu = self.gpu_pool.acquire()
      try:
        ok = self._run_once(trial, pruner, gpu)
      finally:
        self.gpu_pool.release(gpu)
      if ok or trial.status == STATUS_PRUNED:
        return trial
      attempt += 1
      if attempt > self.spec.retries:
        trial.status = STATUS_FAILED
        return trial
      logging.warning("Trial %s failed (exit %s); retry %s/%s.", trial.trial_id,
                      trial.exit_code, attempt, self.spec.retries)
      trial.seed += 1000

  def _run_once(self, trial, pruner, gpu):
    """One subprocess attempt.  Returns ``True`` on a clean exit."""
    argv = build_trial_argv(self.spec, trial, self.base_argv)
    env = dict(os.environ)
    if gpu is not None:
      env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_path = os.path.join(trial.trial_dir, "trial.log")
    metrics_path = os.path.join(trial.trial_dir, "metrics.jsonl")
    trial.status = STATUS_RUNNING
    trial.series = []
    trial.best = None
    start = time.time()
    with open(log_path, "wb") as log_file:
      proc = subprocess.Popen(
          self._launch(argv),
          env=env,
          stdout=log_file,
          stderr=subprocess.STDOUT,
      )
      pruned = self._supervise(proc, trial, pruner, metrics_path)
    trial.duration = time.time() - start
    trial.exit_code = proc.returncode
    if pruned:
      trial.status = STATUS_PRUNED
      return True
    if proc.returncode == 0:
      trial.status = STATUS_COMPLETED
      return True
    trial.status = STATUS_FAILED
    return False

  def _launch(self, argv):
    """Compose the child argv: a module launch or a direct script path."""
    if self.train_module.endswith(".py"):
      return [self.python, self.train_module, *argv]
    return [self.python, "-m", self.train_module, *argv]

  def _supervise(self, proc, trial, pruner, metrics_path):
    """Poll the subprocess, ingest metrics, and prune when the policy says so."""
    pruned = False
    while proc.poll() is None:
      pruned = self._ingest(trial, pruner, metrics_path) or pruned
      if pruned:
        self._terminate(proc)
        break
      time.sleep(self.poll_interval)
    self._ingest(trial, pruner, metrics_path)  # final flush
    trial.best = trial.best_objective(self.spec.direction)
    return pruned

  def _ingest(self, trial, pruner, metrics_path):
    """Read any new epoch points and ask the pruner about them."""
    series = read_objective_series(metrics_path, self.spec.objective)
    if len(series) == len(trial.series):
      return False
    trial.series = series
    trial.best = trial.best_objective(self.spec.direction)
    return pruner.should_stop(trial.trial_id, trial.series)

  @staticmethod
  def _terminate(proc):
    """Ask the child to stop, escalating to SIGKILL after a grace period."""
    proc.terminate()
    try:
      proc.wait(timeout=_TERMINATE_GRACE)
    except subprocess.TimeoutExpired:  # pragma: no cover - defensive
      proc.kill()
      proc.wait()


class StudyRunner:
  """Schedules trials for a study until the budget or space is exhausted."""

  def __init__(self,
               spec,
               store,
               base_argv,
               workers=1,
               gpus=None,
               python=None,
               train_module=DEFAULT_TRAIN_MODULE,
               poll_interval=_POLL_INTERVAL):
    self.spec = spec
    self.store = store
    self.sampler = build_sampler(spec.sampler,
                                 spec.space,
                                 direction=spec.direction,
                                 seed=spec.seed,
                                 **spec.sampler_args)
    self.pruner = build_pruner(spec.pruner,
                               objective_is_min=(spec.direction == "min"),
                               **spec.pruner_args)
    self.runner = SubprocessTrialRunner(spec,
                                        base_argv,
                                        python=python,
                                        train_module=train_module,
                                        gpu_pool=GpuPool(gpus),
                                        poll_interval=poll_interval)
    self.workers = max(1, int(workers))
    self._store_lock = threading.Lock()
    self._sampler_lock = threading.Lock()

  def run(self):
    """Drive the sweep; returns the store when the budget is reached."""
    start_id = self.store.next_trial_id
    done = len(self.store.completed_ids())
    remaining = None if self.spec.max_trials <= 0 else max(0, self.spec.max_trials -
                                                           done)
    if self.workers == 1:
      self._run_sequential(start_id, remaining)
    else:
      self._run_parallel(start_id, remaining)
    return self.store

  # -- single-process path -------------------------------------------------

  def _run_sequential(self, start_id, remaining):
    next_id = start_id
    while remaining is None or remaining > 0:
      trial = self._make_trial(next_id, self._observed())
      if trial is None:
        break
      with self._store_lock:
        self.store.record(trial)
      self.runner.run(trial, self.pruner)
      with self._store_lock:
        self.store.record(trial)
      logging.info("Trial %s %s: best=%s", trial.trial_id, trial.status, trial.best)
      next_id += 1
      if remaining is not None:
        remaining -= 1

  # -- multi-process path --------------------------------------------------

  def _run_parallel(self, start_id, remaining):
    next_id = start_id
    running = {}
    with ThreadPoolExecutor(max_workers=self.workers) as pool:
      while (remaining is None or remaining > 0) or running:
        while (remaining is None or remaining > 0) and len(running) < self.workers:
          trial = self._make_trial(next_id, self._observed())
          if trial is None:
            remaining = 0
            break
          with self._store_lock:
            self.store.record(trial)
          running[pool.submit(self.runner.run, trial, self.pruner)] = trial
          next_id += 1
          if remaining is not None:
            remaining -= 1
        if not running:
          break
        finished, _ = wait(running, return_when=FIRST_COMPLETED)
        for future in finished:
          trial = running.pop(future)
          trial = future.result()
          with self._store_lock:
            self.store.record(trial)
          logging.info("Trial %s %s: best=%s", trial.trial_id, trial.status, trial.best)

  # -- helpers -------------------------------------------------------------

  def _observed(self):
    with self._store_lock:
      return self.store.observed()

  def _make_trial(self, trial_id, observed):
    """Ask the sampler for the next config; ``None`` when the space is done."""
    with self._sampler_lock:
      try:
        config = self.sampler.suggest(observed, n=1)[0]
      except SearchSpaceExhausted:
        logging.info("Search space exhausted after %s trials.", trial_id)
        return None
    trial = Trial(trial_id=trial_id, config=config, seed=self.spec.seed + trial_id)
    trial.trial_dir = os.path.join(self.store.study_dir, "trials", str(trial_id))
    os.makedirs(trial.trial_dir, exist_ok=True)
    return trial


def run_study(spec,
              store,
              base_argv,
              workers=1,
              gpus=None,
              python=None,
              train_module=DEFAULT_TRAIN_MODULE,
              poll_interval=_POLL_INTERVAL):
  """Convenience wrapper: build a :class:`StudyRunner` and run it."""
  if len(spec.space) == 0:
    fatal("Study has an empty search space; nothing to tune.", ValueError)
  return StudyRunner(spec,
                     store,
                     base_argv,
                     workers=workers,
                     gpus=gpus,
                     python=python,
                     train_module=train_module,
                     poll_interval=poll_interval).run()
