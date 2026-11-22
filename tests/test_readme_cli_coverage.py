"""Guard: every CLI flag the parsers declare must be referenced by a README.

The README CLI suite only fails when a marked example breaks; a flag that
exists in ``genml_kit/cli`` but appears in *no* README fence can silently
drift (rename, change semantics, disappear) with green CI.  This test closes
that hole cheaply: it builds the real argparse parsers (registry-composed
owner args included), scans every fence and prose occurrence of a flag in
the READMEs, and requires

* every parser flag to be referenced somewhere (``--help``/``-h`` excluded),
* every referenced-but-unknown flag to be a documented allowlist entry.

The allowlist is data, not a blanket waiver: each entry needs a reason that
a reviewer can read.  Delete an entry when the flag gains an example.
"""

import argparse
import re

import pytest

from tests.readme_cli_harness import _README_GLOBS, REPO_ROOT

# Flags that intentionally have no README example.  Keys are ``module:flag``
# (module = train/infer/llm/tune/tokenizer); values are the reason.  Delete
# an entry when the flag gains a README example.
_FLAG_ALLOWLIST = {
    # --- SSL architecture internals: single-flag knobs with sane defaults;
    # --- the README documents the concepts and the main size/ratio flags.
    "train:--decoder_depth": "SimMIM decoder detail, default tuned per model",
    "train:--decoder_dim": "SimMIM decoder detail, default tuned per model",
    "train:--predictor_depth": "I-JEPA predictor depth, default tuned per model",
    "train:--predictor_dim": "I-JEPA predictor width, default tuned per model",
    "train:--dino_proj_hidden": "DINO projection-head width, default 2048",
    "train:--reward_head_hidden": "RL reward-head width, niche custom-model knob",
    # --- RL: mirrors of documented flags with distinct per-method prefixes.
    "train:--dqn_n_step": "covered by the method-agnostic --n_step in rl-dqn-replay",
    "train:--dqn_no_ddqn": "negative of the documented --dqn_ddqn",
    "train:--sac_alpha": "fixed-alpha alternative to the documented --sac_auto_alpha",
    "train:--sac_no_auto_alpha": "negative of the documented --sac_auto_alpha",
    # --- XGBoost training hyperparameters: the README points at the xgboost
    # --- docs; the pipeline is exercised end-to-end instead.
    "train:--xgb_colsample_bytree": "xgboost-native knob, passthrough",
    "train:--xgb_gamma": "xgboost-native knob, passthrough",
    "train:--xgb_learning_rate": "xgboost-native knob, passthrough",
    "train:--xgb_max_depth": "xgboost-native knob, passthrough",
    "train:--xgb_min_child_weight": "xgboost-native knob, passthrough",
    "train:--xgb_n_estimators": "xgboost-native knob, passthrough",
    "train:--xgb_reg_alpha": "xgboost-native knob, passthrough",
    "train:--xgb_subsample": "xgboost-native knob, passthrough",
    "train:--xgb_use_gpu": "xgboost-native knob, passthrough",
    # --- rarely-needed plumbing.
    "train:--in_ch": "input-channel override for non-RGB data, niche",
    "train:--no-grad_checkpoint": "negative of the documented --grad_checkpoint",
    "train:--tokenizer_arg": "processor kwargs for text pipelines, niche",
    "train:--warmup_steps": "step-granularity warmup, --warmup_epochs is documented",
    "llm:--tokenizer_arg": "tokenizer-construction kwargs, niche",
    "tune:--direction": "documented in prose tables; exercised via the study YAML path",
    "tune:--min_resource": "asha/hyperband detail, default fine",
    "tune:--pruner_arg": "pruner kwargs, passthrough",
    "tune:--python": "trial interpreter override, niche deployment knob",
    "tune:--reduction_factor": "hyperband detail, default fine",
    "tune:--retries": "retry policy, default fine",
    "tune:--sampler_arg": "sampler kwargs, passthrough",
    "tune:--train": "quoted train argv form; the -- separator form is documented",
    "tune:--train-module": "trial module override, niche deployment knob",
    "tokenizer:--force": "rebuild cache even if present, niche",
}


def _parser_surface(entry, module_name, builder_name="build_parser"):
  """Flags declared by a CLI module's parser, including subcommands."""
  module = __import__(module_name, fromlist=[builder_name])
  parser = getattr(module, builder_name)()
  surface = {"": set()}
  for action in parser._actions:
    for opt in action.option_strings:
      surface[""].add(opt)
    if isinstance(action, argparse._SubParsersAction):
      for sub_name, sub_parser in action.choices.items():
        surface.setdefault(sub_name, set())
        for sub_action in sub_parser._actions:
          for opt in sub_action.option_strings:
            surface[sub_name].add(opt)
  return surface


_SURFACE = {
    "train": _parser_surface("train", "genml_kit.cli.train"),
    "infer": _parser_surface("infer", "genml_kit.cli.infer"),
    "llm": _parser_surface("llm", "genml_kit.cli.llm"),
    "tune": _parser_surface("tune", "genml_kit.cli.tune"),
    "tokenizer": _parser_surface("tokenizer", "genml_kit.cli.tokenizer"),
}


def _readme_text():
  chunks = []
  for pattern in _README_GLOBS:
    for md in sorted(REPO_ROOT.glob(pattern)):
      chunks.append(md.read_text(encoding="utf-8"))
  return "\n".join(chunks)


def _referenced_flags():
  """Every ``--flag`` token appearing anywhere in the READMEs."""
  text = _readme_text()
  # A backtick immediately before the dashes is fine: prose references a flag
  # as ``--flag`` all over the READMEs.  Only reject matches glued to a word
  # character (e.g. the middle of an identifier or a URL).
  return set(re.findall(r"(?<![\w])--[A-Za-z0-9][A-Za-z0-9_.-]*", text))


@pytest.mark.parametrize("module,sub", [(m, s) for m in _SURFACE for s in _SURFACE[m]])
def test_every_flag_referenced_or_allowlisted(module, sub):
  surface = _SURFACE[module][sub]
  referenced = _referenced_flags()
  missing = []
  for flag in sorted(surface):
    if flag in ("-h", "--help"):
      continue
    key = f"{module}:{flag}"
    if flag in referenced:
      continue
    if key in _FLAG_ALLOWLIST:
      continue
    missing.append(flag)
  assert not missing, (
      f"flags of `genml-kit-{module}{' ' + sub if sub else ''}` are not "
      "referenced by any README and not allowlisted in "
      "tests/test_readme_cli_coverage.py::_FLAG_ALLOWLIST -- either document "
      f"them in a README example or waive them with a reason: {missing}")


def test_allowlist_entries_are_real():
  """No stale waivers: every entry must name an existing flag."""
  stale = []
  for key in _FLAG_ALLOWLIST:
    module, flag = key.split(":", 1)
    if module not in _SURFACE or flag not in _SURFACE.get(module, {}).get("", set()) \
        and not any(flag in flags for flags in _SURFACE.get(module, {}).values()):
      stale.append(key)
  assert not stale, f"stale allowlist entries (flag no longer exists): {stale}"


def test_allowlist_entries_have_reason():
  """Every waiver must carry a non-trivial reason string."""
  for key, reason in _FLAG_ALLOWLIST.items():
    assert reason and len(reason) > 10, f"waiver {key!r} needs a real reason"


def test_cache_dir_entry_set_matches_parsers():
  """The runner's ``--cache_dir`` injection list matches the real parsers."""
  from tests.readme_cli_harness import _CACHE_DIR_ENTRIES, _ENTRY_MODULES
  for name, module_name in sorted(_ENTRY_MODULES.items()):
    module = __import__(module_name, fromlist=["build_parser"])
    parser = module.build_parser()
    flags = {opt for action in parser._actions for opt in action.option_strings}
    has_flag = "--cache_dir" in flags
    in_set = name in _CACHE_DIR_ENTRIES
    assert has_flag == in_set, (
        f"{name}: declares --cache_dir={has_flag} but membership in "
        f"_CACHE_DIR_ENTRIES={in_set}; keep the set in sync with the parsers")
