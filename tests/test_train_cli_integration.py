"""Integration tests for the real training CLI parser.

Regression guard for the ``--model_arg`` / ``--opt_arg`` / ``--sched_arg`` /
``--lr_group`` breakage introduced in 393cb45 (dropped ``nargs="+"``).

``tests/test_cli_utils.py`` exercises ``KVPairAction`` against a hand-built
parser that hardcodes ``nargs="+"``, so it could not catch the production
flag losing its ``nargs``.  These tests go through ``parse_args()`` -- the
exact builder used by the ``genml_kit-train`` entry point -- so the flag
declaration and the action are never allowed to drift apart again.
"""

from genml_kit.training.train import parse_args


def test_model_arg_multiple_tokens():
  """``--model_arg depth=6 num_heads=8`` collects all tokens."""
  args = parse_args(["--model_arg", "embed_dim=512", "num_heads=8"])
  assert args.model_arg == {"embed_dim": 512, "num_heads": 8}


def test_model_arg_spec_value():
  """``--model_arg decoder=unet:depth=4`` (segmentation decoder spec)."""
  args = parse_args(["--model_arg", "decoder=unet:depth=4"])
  assert args.model_arg == {"decoder": "unet:depth=4"}


def test_opt_arg_multiple_tokens():
  args = parse_args(["--opt_arg", "betas=0.9,0.999", "momentum=0.9"])
  assert args.opt_arg == {"betas": "0.9,0.999", "momentum": 0.9}


def test_sched_arg_multiple_tokens():
  args = parse_args(["--sched_arg", "T_max=50", "eta_min=1e-6"])
  assert args.sched_arg == {"T_max": 50, "eta_min": 1e-06}


def test_lr_group_multiple_tokens():
  args = parse_args(["--lr_group", "backbone.*=1e-5", "classifier.*=1e-3"])
  assert args.lr_group == {"backbone.*": 1e-05, "classifier.*": 0.001}


def test_kv_args_default_to_empty_dict():
  """Absent flags yield ``{}`` (safe for ``**args.opt_arg`` etc.)."""
  args = parse_args([])
  assert args.model_arg == {}
  assert args.opt_arg == {}
  assert args.sched_arg == {}
  assert args.lr_group == {}
