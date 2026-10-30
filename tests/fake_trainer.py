"""A tiny stand-in for ``genml-kit-train`` used by the search tests.

It mimics the only surface the search subsystem depends on: it accepts
hyperparameter override flags plus ``--log_dir`` / ``--epochs`` for free (via
``parse_known_args``) and writes an epoch-scope scalar series to
``<log_dir>/metrics.jsonl`` in exactly the schema the real JSONL sink emits.

Several objective tags are emitted per epoch (mirroring the real reporter's
validation block), so a study's ``objective`` genuinely selects which tag to
track.  The ``val/macro_f1`` tag is a simple quadratic in ``--x`` peaking at
``x == 0.2``, which gives the samplers a known, findable optimum.
``--epoch_sleep`` slows the run so a pruner can intervene mid-training.
"""

import argparse
import json
import os
import sys
import time

_OBJECTIVE_TAGS = ("val/macro_f1", "epoch/loss")


def _score(x):
  return 1.0 - (x - 0.2)**2


def build_parser():
  parser = argparse.ArgumentParser()
  parser.add_argument("--log_dir", type=str, default="logs")
  parser.add_argument("--epochs", type=int, default=3)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--x", type=float, default=0.2)
  parser.add_argument("--epoch_sleep", type=float, default=0.0)
  parser.add_argument("--fail", action="store_true")
  parser.add_argument("--checkpoint", type=str, default=None)
  return parser


def main(argv=None):
  args, _ = build_parser().parse_known_args(argv)
  if args.fail:
    print("simulated failure", file=sys.stderr)
    return 1
  os.makedirs(args.log_dir, exist_ok=True)
  metrics_path = os.path.join(args.log_dir, "metrics.jsonl")
  base = _score(args.x)
  with open(metrics_path, "w", encoding="utf-8") as handle:
    for epoch in range(args.epochs):
      loss = 5.0 - 0.1 * epoch
      scalars = {"val/macro_f1": base + 0.001 * epoch, "epoch/loss": loss}
      for tag in _OBJECTIVE_TAGS:
        handle.write(
            json.dumps({
                "type": "scalar",
                "tag": tag,
                "value": scalars[tag],
                "scope": "epoch",
                "epoch": epoch,
            }) + "\n")
      handle.flush()
      print(f"epoch {epoch} {scalars}")
      if args.epoch_sleep > 0:
        time.sleep(args.epoch_sleep)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
