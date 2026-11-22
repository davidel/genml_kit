"""Fast unit tests for the README CLI example harness.

No subprocess, no network: parses synthetic markdown, checks marker
validation errors, argv building/overrides, launcher rewrites, candidate
fence detection, and the service-availability model.
"""

import sys

import pytest
import yaml

from tests import services
from tests.readme_cli_harness import (
    _README_GLOBS,
    REPO_ROOT,
    CommandSpec,
    _train_command_count,
    _validate_overrides,
    build_argv,
    candidate_fences,
    read_markers,
)

# Fast unit tests (no subprocess, no network)


def _write(tmp_path, text, name="doc.md"):
  path = tmp_path / name
  path.write_text(text, encoding="utf-8")
  return path


_MARKER = "<!-- genml-cli-test\nnetwork: true\nextra: --epochs 1\n-->\n"

_EXCLUDE = "<!-- genml-cli-test-exclude: not a runnable command -->\n"


class TestReadMarkers:

  def test_marker_above_bash_fence_detected(self, tmp_path):
    path = _write(tmp_path, f"{_MARKER}```bash\ngenml-kit-train --epochs 5\n```\n")
    specs = read_markers(path)
    assert len(specs) == 1
    assert specs[0].commands == [["genml-kit-train", "--epochs", "5"]]
    assert specs[0].network is True

  def test_identical_fence_without_marker_ignored(self, tmp_path):
    path = _write(tmp_path, "```bash\ngenml-kit-train --epochs 5\n```\n")
    assert read_markers(path) == []

  def test_exclude_token_is_not_a_marker(self, tmp_path):
    path = _write(tmp_path, f"{_EXCLUDE}```bash\ngenml-kit-train --epochs 5\n```\n")
    assert read_markers(path) == []

  def test_marker_above_unlabeled_fence_raises(self, tmp_path):
    path = _write(tmp_path, f"{_MARKER}```\ngenml-kit-train --epochs 5\n```\n")
    with pytest.raises(ValueError, match="relabel"):
      read_markers(path)

  def test_marker_on_commandless_fence_raises(self, tmp_path):
    path = _write(tmp_path, f"{_MARKER}```bash\n# just a comment\n```\n")
    with pytest.raises(ValueError):
      read_markers(path)

  def test_non_core_command_raises(self, tmp_path):
    path = _write(tmp_path, f"{_MARKER}```bash\npip install genml_kit\n```\n")
    with pytest.raises(ValueError):
      read_markers(path)

  def test_underscore_entry_point_is_hard_error(self, tmp_path):
    path = _write(tmp_path, f"{_MARKER}```bash\ngenml_kit-train --epochs 5\n```\n")
    with pytest.raises(ValueError):
      read_markers(path)

  def test_heading_tracked(self, tmp_path):
    path = _write(
        tmp_path,
        f"## Quick Start\n\n{_MARKER}```bash\ngenml-kit-train --epochs 5\n```\n")
    assert read_markers(path)[0].heading == "## Quick Start"


class TestMarkerValidation:

  def test_unknown_key_raises(self, tmp_path):
    path = _write(
        tmp_path, "<!-- genml-cli-test\nnetwrok: true\n-->\n"
        "```bash\ngenml-kit-train --epochs 5\n```\n")
    with pytest.raises(ValueError, match="unknown marker key"):
      read_markers(path)

  def test_malformed_yaml_raises(self, tmp_path):
    path = _write(
        tmp_path, "<!-- genml-cli-test\nnetwork: [unclosed\n-->\n"
        "```bash\ngenml-kit-train --epochs 5\n```\n")
    with pytest.raises(yaml.YAMLError):
      read_markers(path)

  def test_service_implies_network(self, tmp_path):
    path = _write(
        tmp_path, "<!-- genml-cli-test\nservice: hf\n-->\n"
        "```bash\ngenml-kit-train --epochs 5\n```\n")
    spec = read_markers(path)[0]
    assert spec.service == "hf" and spec.network

  def test_unknown_service_raises(self, tmp_path):
    path = _write(
        tmp_path, "<!-- genml-cli-test\nservice: dropbox\n-->\n"
        "```bash\ngenml-kit-train --epochs 5\n```\n")
    with pytest.raises(ValueError, match="unknown service"):
      read_markers(path)

  def test_service_with_network_false_raises(self, tmp_path):
    path = _write(
        tmp_path, "<!-- genml-cli-test\nnetwork: false\n"
        "service: hf\n-->\n"
        "```bash\ngenml-kit-train --epochs 5\n```\n")
    with pytest.raises(ValueError):
      read_markers(path)

  def test_bad_timeout_raises(self, tmp_path):
    path = _write(
        tmp_path, "<!-- genml-cli-test\ntimeout: 0\n-->\n"
        "```bash\ngenml-kit-train --epochs 5\n```\n")
    with pytest.raises(ValueError, match="timeout"):
      read_markers(path)

  def test_multi_command_fence_ordering(self, tmp_path):
    path = _write(
        tmp_path, f"{_MARKER}```bash\n"
        "genml-kit-train --epochs 5 \\\n"
        "  --batch_size 8\n"
        "genml-kit-infer --image x.png\n```\n")
    cmds = read_markers(path)[0].commands
    assert cmds == [["genml-kit-train", "--epochs", "5", "--batch_size", "8"],
                    ["genml-kit-infer", "--image", "x.png"]]


class TestBuildArgv:

  def _spec(self, **kwargs):
    base = dict(md="README.md",
                line=1,
                heading="## H",
                name="n",
                commands=[["genml-kit-train", "--epochs", "5"]])
    base.update(kwargs)
    return CommandSpec(**base)

  def test_override_replace_keeps_position(self, tmp_path):
    spec = self._spec(
        commands=[["genml-kit-train", "--dataset", "hf/name", "--epochs", "5"]],
        override=["--dataset {tmp}/img"])
    argv = build_argv(spec, spec.commands[0], tmp_path)
    assert argv[:3] == [sys.executable, "-m", "genml_kit.cli.train"]
    assert argv[3:5] == ["--dataset", str(tmp_path / "img")]
    assert argv[5:] == ["--epochs", "5", "--cache_dir", str(tmp_path / "hf_cache")]

  def test_override_delete_removes_flag_and_value(self, tmp_path):
    spec = self._spec(
        commands=[["genml-kit-train", "--source_checkpoint", "/x.pt", "--epochs", "5"]],
        override=["-.--source_checkpoint"])
    argv = build_argv(spec, spec.commands[0], tmp_path)
    assert "--source_checkpoint" not in argv
    assert "/x.pt" not in argv

  def test_unmatched_override_raises(self, tmp_path):
    spec = self._spec(override=["--nonexistent x"])
    with pytest.raises(ValueError, match="matches no command"):
      _validate_overrides(spec)

  def test_cache_dir_not_duplicated(self, tmp_path):
    spec = self._spec(
        commands=[["genml-kit-train", "--cache_dir", "/given", "--epochs", "5"]])
    argv = build_argv(spec, spec.commands[0], tmp_path)
    assert argv.count("--cache_dir") == 1
    assert "/given" in argv

  def test_extra_appended(self, tmp_path):
    spec = self._spec(extra=["--epochs", "1", "--max_items", "16"])
    argv = build_argv(spec, spec.commands[0], tmp_path)
    assert "--max_items" in argv and "16" in argv
    # --cache_dir is injected *after* the limiter args.
    assert argv.index("--cache_dir") > argv.index("--max_items")

  def test_single_value_flag_does_not_swallow_positional(self, tmp_path):
    """A single-value flag stops after one token (block [31] infer)."""
    spec = self._spec(commands=[[
        "genml-kit-infer", "--checkpoint", "genml_kit_best.pt", "path/to/lesion.jpg"
    ]],
                      override=["--checkpoint {tmp}/chk/model_best.pt"])
    argv = build_argv(spec, spec.commands[0], tmp_path)
    assert "path/to/lesion.jpg" in argv
    assert argv[argv.index("--checkpoint") + 1] == str(tmp_path / "chk/model_best.pt")

  def test_extra_only_applied_to_train_commands(self, tmp_path):
    """Inference commands reject training-only flags (block [31])."""
    spec = self._spec(commands=[["genml-kit-train", "--epochs", "5"],
                                ["genml-kit-infer", "--image", "x.png"]],
                      extra=["--epochs", "1"])
    argv = build_argv(spec, spec.commands[1], tmp_path)
    assert "--epochs" not in argv

  def test_multi_train_fence_isolates_checkpoints(self, tmp_path):
    """Each training command gets its own prefix + --no-resume (blocks [30])."""
    spec = self._spec(commands=[["genml-kit-train", "--epochs", "5"],
                                ["genml-kit-train", "--epochs", "5"]])
    first = build_argv(spec, spec.commands[0], tmp_path, cmd_index=0)
    second = build_argv(spec, spec.commands[1], tmp_path, cmd_index=1)
    assert first[first.index("--checkpoint") +
                 1] != second[second.index("--checkpoint") + 1]
    assert "--no-resume" in first and "--no-resume" in second

  def test_single_train_fence_keeps_default_checkpoint(self, tmp_path):
    spec = self._spec(commands=[["genml-kit-train", "--epochs", "5"]])
    argv = build_argv(spec, spec.commands[0], tmp_path)
    assert "--checkpoint" not in argv and "--no-resume" not in argv


_LAUNCHER_MARKER = ("<!-- genml-cli-test\nlauncher: torchrun\nnproc: 2\n"
                    "-->\n")
_LAUNCHER_FENCE = ("```bash\ntorchrun --standalone --nproc_per_node=2 -m "
                   "genml_kit.cli.train --epochs 5\n```\n")


class TestLauncherFences:

  def test_launcher_fence_parsed(self, tmp_path):
    path = _write(tmp_path, f"{_LAUNCHER_MARKER}{_LAUNCHER_FENCE}")
    spec = read_markers(path)[0]
    assert spec.launcher == "torchrun"
    assert spec.nproc == 2

  def test_launcher_fence_is_candidate(self, tmp_path):
    path = _write(tmp_path, f"{_LAUNCHER_MARKER}{_LAUNCHER_FENCE}")
    assert [(m, x) for _, _, m, x in candidate_fences(path)] == [(True, False)]

  def test_unknown_launcher_raises(self, tmp_path):
    path = _write(
        tmp_path, "<!-- genml-cli-test\nlauncher: slurm\n-->\n"
        "```bash\nslurm -m genml_kit.cli.train --epochs 5\n```\n")
    with pytest.raises(ValueError, match="unknown launcher"):
      read_markers(path)

  def test_nproc_without_launcher_raises(self, tmp_path):
    path = _write(
        tmp_path, "<!-- genml-cli-test\nnproc: 2\n-->\n"
        "```bash\ngenml-kit-train --epochs 5\n```\n")
    with pytest.raises(ValueError, match="nproc requires a launcher"):
      read_markers(path)

  def test_bad_nproc_raises(self, tmp_path):
    path = _write(
        tmp_path, "<!-- genml-cli-test\nlauncher: torchrun\n"
        "nproc: 1\n-->\n"
        "```bash\ntorchrun -m genml_kit.cli.train --epochs 5\n```\n")
    with pytest.raises(ValueError, match="nproc"):
      read_markers(path)

  def test_core_command_in_launcher_fence_raises(self, tmp_path):
    path = _write(tmp_path, f"{_LAUNCHER_MARKER}"
                  "```bash\ngenml-kit-train --epochs 5\n```\n")
    with pytest.raises(ValueError, match="non-torchrun"):
      read_markers(path)

  def test_rewrite_uses_distributed_run(self, tmp_path):
    path = _write(tmp_path, f"{_LAUNCHER_MARKER}{_LAUNCHER_FENCE}")
    spec = read_markers(path)[0]
    argv = build_argv(spec, spec.commands[0], tmp_path)
    assert argv[:3] == [sys.executable, "-m", "torch.distributed.run"]
    assert "--standalone" in argv
    assert "--nproc_per_node=2" in argv
    # Inner entry module is rewritten to the importable module, then the
    # training limiter/override logic applies exactly as for a plain fence.
    assert argv[argv.index("-m", 3) + 1] == "genml_kit.cli.train"
    assert "--cache_dir" in argv

  def test_rewrite_injects_nproc_when_absent(self, tmp_path):
    path = _write(
        tmp_path, "<!-- genml-cli-test\nlauncher: torchrun\nnproc: 3\n-->\n"
        "```bash\ntorchrun --standalone -m genml_kit.cli.train --epochs 5\n```\n")
    spec = read_markers(path)[0]
    argv = build_argv(spec, spec.commands[0], tmp_path)
    assert "--nproc_per_node=3" in argv

  def test_rewrite_rejects_conflicting_nproc(self, tmp_path):
    path = _write(
        tmp_path, "<!-- genml-cli-test\nlauncher: torchrun\nnproc: 2\n-->\n"
        "```bash\ntorchrun --standalone --nproc_per_node=4 -m "
        "genml_kit.cli.train --epochs 5\n```\n")
    spec = read_markers(path)[0]
    with pytest.raises(ValueError, match="disagrees with marker nproc"):
      build_argv(spec, spec.commands[0], tmp_path)

  def test_launcher_command_count_deduces_train(self, tmp_path):
    path = _write(tmp_path, f"{_LAUNCHER_MARKER}{_LAUNCHER_FENCE}")
    spec = read_markers(path)[0]
    assert _train_command_count(spec) == 1


class TestCandidateFences:

  def test_marked_candidate_reported_marked(self, tmp_path):
    path = _write(tmp_path, f"{_MARKER}```bash\ngenml-kit-train --epochs 5\n```\n")
    assert [(m, x) for _, _, m, x in candidate_fences(path)] == [(True, False)]

  def test_unmarked_candidate_reported(self, tmp_path):
    path = _write(tmp_path, "```bash\ngenml-kit-train --epochs 5\n```\n")
    assert [(m, x) for _, _, m, x in candidate_fences(path)] == [(False, False)]

  def test_excluded_candidate_reported_excluded(self, tmp_path):
    path = _write(tmp_path, f"{_EXCLUDE}```bash\ngenml-kit-train --epochs 5\n```\n")
    assert [(m, x) for _, _, m, x in candidate_fences(path)] == [(False, True)]

  def test_non_core_fence_is_not_a_candidate(self, tmp_path):
    path = _write(tmp_path, "```bash\npip install genml_kit\n```\n")
    assert list(candidate_fences(path)) == []

  def test_unlabeled_fence_is_not_a_candidate(self, tmp_path):
    path = _write(tmp_path, "```\ngenml-kit-train --epochs 5\n```\n")
    assert list(candidate_fences(path)) == []

  def test_empty_exclude_reason_raises(self, tmp_path):
    path = _write(
        tmp_path, "<!-- genml-cli-test-exclude: -->\n"
        "```bash\ngenml-kit-train --epochs 5\n```\n")
    with pytest.raises(ValueError, match="empty genml-cli-test-exclude"):
      list(candidate_fences(path))


def test_all_candidate_fences_marked_or_excluded():
  """No README example may slip in untested and unexcluded.

  Every ``bash``/``sh`` fence that carries a core ``genml-kit-*`` command must
  be opted in with a marker or opted out with an exclusion comment.  This
  replaces the old magic-number count assertion: it catches a brand-new
  unmarked example, not just a dropped marker.
  """
  problems = []
  for pattern in _README_GLOBS:
    for md in sorted(REPO_ROOT.glob(pattern)):
      rel = md.relative_to(REPO_ROOT)
      for line, _info, marked, excluded in candidate_fences(md):
        if not marked and not excluded:
          problems.append(f"{rel}:{line}: core command in a fence that is "
                          "neither marked nor excluded")
  assert not problems, "\n".join(problems)


class TestServices:
  """Availability gate: credentials OR an endpoint override satisfy a service."""

  def test_credentials_satisfy(self):
    env = {"HF_TOKEN": "hf_abc"}
    assert services.has_credentials("hf", env)
    assert services.available("hf", env)

  def test_endpoint_override_satisfies_without_credentials(self):
    env = {"HF_ENDPOINT": "http://proxy/hf"}
    assert not services.has_credentials("hf", env)
    assert services.endpoint_override("hf", env) == "HF_ENDPOINT"
    assert services.available("hf", env)

  def test_blank_values_do_not_satisfy(self):
    env = {"HF_TOKEN": "  ", "HF_ENDPOINT": ""}
    assert not services.available("hf", env)

  def test_neither_satisfies(self):
    assert not services.available("hf", {})
    assert services.skip_reason("hf") == (
        "requires HF_TOKEN or HUGGING_FACE_HUB_TOKEN or HF_ENDPOINT")

  def test_alternative_tokens_each_satisfy(self):
    for var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
      assert services.has_credentials("hf", {var: "tok"})

  def test_credential_pair_requires_both(self):
    assert not services.available("s3", {"AWS_ACCESS_KEY_ID": "k"})
    assert not services.available("s3", {"AWS_SECRET_ACCESS_KEY": "s"})
    assert services.available("s3", {
        "AWS_ACCESS_KEY_ID": "k",
        "AWS_SECRET_ACCESS_KEY": "s"
    })

  def test_second_endpoint_alternative_satisfies(self):
    for var in ("AWS_ENDPOINT_URL_S3", "AWS_ENDPOINT_URL"):
      assert services.available("s3", {var: "http://emu/s3"})

  def test_require_skips_when_unavailable(self):
    with pytest.raises(pytest.skip.Exception):
      services.require("gcs", {})

  def test_require_passes_when_endpoint_present(self):
    services.require("gcs", {"STORAGE_EMULATOR_HOST": "http://emu"})
