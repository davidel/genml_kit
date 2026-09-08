"""Tests for the scdiag → genml_kit restructuring.

Guards the two things the split changed: where code lives / what the
package is called, and the CLI contract change (``--dataset`` lost its
domain default and is now required).
"""

import importlib.util

import pytest


def test_package_imports_under_new_name():
  """The package imports as genml_kit and reports the new version."""
  genml_kit = importlib.import_module("genml_kit")
  assert genml_kit.__version__ == "0.1.0"


def test_entry_point_modules_resolve():
  """Every console-script target module imports cleanly."""
  from genml_kit.pretrain.cli import main as pretrain_main
  from genml_kit.training.infer import main as infer_main
  from genml_kit.training.train import main as train_main

  assert callable(train_main)
  assert callable(pretrain_main)
  assert callable(infer_main)


def test_pretrain_harness_is_not_train_harness():
  """cli.main and train.main are distinct functions (not aliased)."""
  from genml_kit.pretrain.cli import main as pretrain_main
  from genml_kit.training.train import main as train_main

  assert pretrain_main is not train_main


@pytest.mark.parametrize(
    "argv",
    [
        ["--help"],
        [],
        ["--model", "google/vit-base-patch16-224"],
    ],
)
def test_train_cli_requires_dataset(argv, capsys):
  """--dataset is required: --help works, all other argvs fail cleanly."""
  from genml_kit.training.train import parse_args

  if argv == ["--help"]:
    with pytest.raises(SystemExit) as excinfo:
      parse_args(argv)
    assert excinfo.value.code == 0
    assert "usage:" in capsys.readouterr().out
    return

  with pytest.raises(SystemExit) as excinfo:
    parse_args(argv)
  assert excinfo.value.code == 2
  err = capsys.readouterr().err
  assert "--dataset" in err


def test_train_cli_accepts_dataset():
  """The required --dataset flag parses and is preserved."""
  from genml_kit.training.train import parse_args

  args = parse_args(
      ["--model", "google/vit-base-patch16-224", "--dataset", "my-org/my-images"])
  assert args.dataset == "my-org/my-images"
  assert args.model == "google/vit-base-patch16-224"


def test_no_scdiag_submodules_left():
  """No scdiag.<submodule> is importable from the restructured tree."""
  assert importlib.util.find_spec("scdiag.datasets") is None
  assert importlib.util.find_spec("scdiag.models") is None
  assert importlib.util.find_spec("scdiag.train") is None
  assert importlib.util.find_spec("scdiag.pretrain") is None
  assert importlib.util.find_spec("scdiag.io") is None
  assert importlib.util.find_spec("scdiag.utils") is None
  assert importlib.util.find_spec("scdiag.losses") is None
  assert importlib.util.find_spec("scdiag.classifiers") is None
  assert importlib.util.find_spec("scdiag.augmentations") is None
  assert importlib.util.find_spec("scdiag.pretrain_methods") is None
  # "scdiag.models.uvito" is covered by "scdiag.models" above: a dotted
  # find_spec would raise ModuleNotFoundError here, not return None.
