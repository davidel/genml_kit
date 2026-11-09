"""End-to-end multi-process smoke driver for the *RL* path (run under torchrun).

Runs the real ``genml_kit.cli.train.main()`` with ``--pipeline rl --method ppo``
on a tiny CartPole run, 2 ranks, gloo/CPU.  Writes a per-rank marker file with
the observed world size and rank so the parent test can assert that the
distributed path actually engaged.

This is the RL counterpart of ``dist_e2e_driver.py``.  The supervised driver
proves the DDP wiring for a ``DataLoader``-driven pipeline; this one proves it
for the RL path, where:

* the model is deliberately **not** DDP-wrapped (``RLMethod.wrap_model``),
* gradients are reduced explicitly by ``RLTrainer``,
* PPO rollouts are sharded across ranks and all-gathered,
* checkpoints are still written by rank 0 alone with portable keys.

Invoked by ``tests/test_rl_dist_e2e.py`` via ``torchrun --nproc_per_node=2``.
"""
import contextlib
import logging
import os
import pathlib
import sys
from unittest.mock import patch

from genml_kit.utils.logging import GlogFormatter


def _run(tmp_path):
  ckpt_base = str(tmp_path / "ckpts" / "model")
  test_args = [
      "train.py",
      "--pipeline",
      "rl",
      "--method",
      "ppo",
      "--env_id",
      "CartPole-v1",
      "--epochs",
      "1",
      "--steps_per_epoch",
      "4",
      "--ppo_rollout_len",
      "4",
      "--ppo_mini_batch_size",
      "2",
      "--ppo_epochs",
      "1",
      "--eval_episodes",
      "1",
      "--batch_size",
      "4",
      "--checkpoint",
      ckpt_base,
      "--log_every",
      "1",
      "--log_level",
      "INFO",
      "--dist_arg",
      "rollout_shard=true",
  ]
  with patch("sys.argv", test_args):
    from genml_kit.cli.train import main
    main()

  # Read the rank/world from the launcher env: the process group is torn down
  # by the time main() returns here.
  rank = int(os.environ.get("RANK", "0"))
  world = int(os.environ.get("WORLD_SIZE", "1"))
  (tmp_path / f"rank{rank}.marker").write_text(f"rank={rank} world={world}")
  return ckpt_base


def main():
  root = logging.getLogger()
  handlers_before = root.handlers[:]
  tmp_path = None
  try:
    tmp_path = pathlib.Path(os.environ["DIST_E2E_TMPDIR"])
    _run(tmp_path)
  except Exception:
    import traceback
    tb = traceback.format_exc()
    with contextlib.suppress(Exception):
      (pathlib.Path(os.environ["DIST_E2E_TMPDIR"]) /
       f"error.rank{os.environ.get('RANK', '?')}.log").write_text(tb)
    sys.stderr.write(tb)
    raise
  finally:
    for h in root.handlers[:]:
      root.removeHandler(h)
      if isinstance(h.formatter, GlogFormatter):
        h.close()
    root.handlers[:] = handlers_before


if __name__ == "__main__":
  main()
