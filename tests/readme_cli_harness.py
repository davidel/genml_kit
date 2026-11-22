"""Marker parsing and argv building for the README CLI example suite.

Extracted from the original monolithic ``tests/test_readme_cli.py`` so the
parsing/rewrite machinery can be unit-tested without the subprocess runner.
Terminology (unchanged from the original module docstring):

A markdown HTML comment (invisible on GitHub) immediately above a ```bash
fence opts that fence in.  Each marked fence is executed as a bounded smoke
test that writes everything under a removable temp dir.  The package is
imported via ``PYTHONPATH`` (no install); the ``genml-kit-*`` console scripts
are rewritten to ``python -m genml_kit...`` so the suite needs no editable
install.
"""

import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from tests import services

REPO_ROOT = Path(__file__).resolve().parents[1]

# Fence opener: leading whitespace, >=3 backticks/tildes, an info string.
_FENCE_RE = re.compile(r"^(?P<indent>\s*)(?P<ticks>`{3,}|~{3,})\s*"
                       r"(?P<info>[A-Za-z0-9_+.\-]*)\s*$")

# HTML comment carrying the marker (may span multiple lines).  The trailing
# ``(?![-\w])`` keeps the sibling ``genml-cli-test-exclude`` token from being
# mistaken for a marker (``\b`` alone matches between ``t`` and ``-``).
_MARKER_RE = re.compile(r"<!--\s*genml-cli-test(?![-\w])(?P<body>.*?)-->", re.DOTALL)

# Opt-out comment for a candidate fence that must not be executed.
_EXCLUDE_RE = re.compile(r"<!--\s*genml-cli-test-exclude\s*:\s*"
                         r"(?P<reason>.*?)-->", re.DOTALL)

# README command -> python -m module.  Only the *correct* hyphen spellings are
# recognised; the underscore typo (``genml_kit-train``) is deliberately absent
# so a marked fence that still uses it is a hard error (regression guard).
_ENTRY_MODULES = {
    "genml-kit-train": "genml_kit.cli.train",
    "genml-kit-infer": "genml_kit.cli.infer",
    "genml-kit-llm": "genml_kit.cli.llm",
    "genml-kit-tune": "genml_kit.cli.tune",
    "genml-kit-tokenizer": "genml_kit.cli.tokenizer",
}

# Launchers that run an entry *module* under ``python -m <launcher>``.  A
# fence headed by one of these is executed multi-process (opt-in via the
# ``launcher:`` marker key), never as a plain single-process command.
_LAUNCHERS = {
    "torchrun": "torch.distributed.run",
}

# Default process count for a ``launcher: torchrun`` fence.
_DEFAULT_NPROC = 2

_README_GLOBS = ["README.md", "*/README.md"]

_ALLOWED_KEYS = {
    "name", "skip", "network", "service", "gated", "timeout", "extra", "override",
    "extras", "env", "cwd", "launcher", "nproc"
}

# Fixture directory a ``gated: true`` fence has its HF dataset remapped to, so a
# gated example is exercisable without a token (or access grant).  Reuses the
# ``img`` ImageFolder fixture built by ``_prepare_fixtures``.
_GATED_FIXTURE = "{tmp}/img"
_GATED_DATASET_FLAGS = ("--dataset", "--datasets")
# The fixture's columns are ``image``/``label``; remap any explicit column flag.
_GATED_COLUMN_FLAGS = {"--image_column": "image", "--label_column": "label"}

_EXTRA_IMPORTS = {
    "lora": "peft",
    "llm": "transformers",
    "rl": "gymnasium",
    "timm": "timm",
}

_DEFAULT_TIMEOUT = 300


@dataclass
class CommandSpec:
  md: str  # repo-relative path, e.g. "README.md"
  line: int  # 1-based line of the opening fence
  heading: str  # nearest preceding "##/### ..." heading
  name: str  # marker name, or f"{md}:{line}"
  commands: list  # list[list[str]]: each inner list is one argv
  network: bool = False
  service: str | None = None
  gated: bool = False
  timeout: int = _DEFAULT_TIMEOUT
  extra: list = field(default_factory=list)  # argv appended to each cmd
  override: list = field(default_factory=list)  # override spec strings
  extras: list = field(default_factory=list)  # pip extras -> importorskip
  env: dict = field(default_factory=dict)
  cwd: str = "{tmp}"
  skip: str | None = None
  launcher: str | None = None  # e.g. "torchrun"; runs the command multi-process
  nproc: int = _DEFAULT_NPROC  # process count for a launcher fence


def _comment_above(lines, fence_idx):
  """Return the HTML comment block immediately above ``lines[fence_idx]``.

  Only blank lines may separate the closing ``-->`` from the fence.  Returns
  the raw comment text (``<!-- ... -->``), or None when no comment sits there.
  """
  j = fence_idx - 1
  while j >= 0 and lines[j].strip() == "":
    j -= 1
  if j < 0 or "-->" not in lines[j]:
    return None
  k = j
  while k >= 0 and "<!--" not in lines[k]:
    k -= 1
  if k < 0:
    return None
  block = "\n".join(lines[k:j + 1])
  if not block.rstrip().endswith("-->"):
    return None
  return block


def _marker_above(lines, fence_idx):
  """Return the marker body immediately above ``lines[fence_idx]`` or None."""
  block = _comment_above(lines, fence_idx)
  if block is None:
    return None
  match = _MARKER_RE.search(block)
  if match is None:
    return None
  return match.group("body")


def _exclude_reason_above(lines, fence_idx):
  """Return the exclusion reason above ``lines[fence_idx]``, or None.

  Raises when an exclusion comment is present but carries an empty reason.
  """
  block = _comment_above(lines, fence_idx)
  if block is None:
    return None
  match = _EXCLUDE_RE.search(block)
  if match is None:
    return None
  reason = match.group("reason").strip()
  if not reason:
    raise ValueError("empty genml-cli-test-exclude reason above the fence; "
                     "state why the example is not executed")
  return reason


def _collect_fence(lines, open_idx):
  """Return (body_lines, closing_idx) for the fence opening at *open_idx*."""
  ticks = _FENCE_RE.match(lines[open_idx]).group("ticks")
  char = ticks[0]
  body = []
  i = open_idx + 1
  while i < len(lines):
    stripped = lines[i].strip()
    if stripped and set(stripped) == {char} and len(stripped) >= len(ticks):
      return body, i
    body.append(lines[i])
    i += 1
  raise ValueError(f"unterminated fence at line {open_idx + 1}")


def _logical_lines(text):
  """Join backslash-continued lines of a fence body into logical commands."""
  logical = []
  buf = ""
  for raw in text.splitlines():
    piece = raw.rstrip()
    buf = f"{buf} {piece}" if buf else piece
    if buf.endswith("\\"):
      buf = buf[:-1]
      continue
    logical.append(buf)
    buf = ""
  if buf:
    logical.append(buf)
  return logical


def fence_has_core_command(text):
  """True if any logical line starts with a core entry point or a launcher.

  A launcher-headed line (``torchrun ...``) is a *candidate* too, so the
  ``test_all_candidate_fences_marked_or_excluded`` guard forces every such
  fence to be either executed (marker with ``launcher:``) or explicitly
  excluded.
  """
  for line in _logical_lines(text):
    stripped = line.strip()
    if stripped == "" or stripped.startswith("#"):
      continue
    tokens = shlex.split(stripped, posix=True)
    if tokens and (tokens[0] in _ENTRY_MODULES or tokens[0] in _LAUNCHERS):
      return True
  return False


def split_commands(text, launcher=None):
  """Split a fence body into one argv per command.

  A plain fence may only contain core ``genml-kit-*`` commands.  A fence
  marked ``launcher: <name>`` is instead expected to be built from
  ``<name>``-headed lines (e.g. ``torchrun --standalone ... -m
  genml_kit.cli.train ...``); a stray core command in such a fence is an
  error, and vice versa.
  """
  commands = []
  for line in _logical_lines(text):
    stripped = line.strip()
    if stripped == "" or stripped.startswith("#"):
      continue
    tokens = shlex.split(stripped, posix=True)
    if not tokens:
      continue
    if launcher is not None:
      if tokens[0] != launcher:
        raise ValueError(f"launcher fence ({launcher}) contains a "
                         f"non-{launcher} command: {stripped!r}")
    elif tokens[0] not in _ENTRY_MODULES:
      raise ValueError(f"non-core command inside a marked fence: {stripped!r}")
    commands.append(tokens)
  if not commands:
    raise ValueError("genml-cli-test marker on a fence with no genml-kit-* "
                     "command")
  return commands


def _parse_marker(rel, line, heading, marker_body, body_lines):
  data = yaml.safe_load(marker_body)
  if data is None:
    data = {}
  if not isinstance(data, dict):
    raise ValueError(f"{rel}:{line}: genml-cli-test marker body is not a "
                     "YAML mapping")
  unknown = set(data) - _ALLOWED_KEYS
  if unknown:
    raise ValueError(f"{rel}:{line}: unknown marker key(s): "
                     f"{sorted(unknown)}")

  network = bool(data.get("network", False))
  service = data.get("service")
  if service is not None:
    if service not in services.SERVICES:
      raise ValueError(f"{rel}:{line}: unknown service: {service!r} "
                       f"(expected one of {sorted(services.SERVICES)})")
    if "network" in data and not network:
      raise ValueError(f"{rel}:{line}: service: {service} with network: false")
    network = True

  timeout = data.get("timeout", _DEFAULT_TIMEOUT)
  if not isinstance(timeout, int) or isinstance(timeout, bool) or timeout <= 0:
    raise ValueError(f"{rel}:{line}: timeout must be a positive int")

  raw_extra = data.get("extra", "")
  if isinstance(raw_extra, str):
    extra = shlex.split(raw_extra, posix=True)
  elif isinstance(raw_extra, list):
    extra = list(raw_extra)
  else:
    raise ValueError(f"{rel}:{line}: extra must be a string or list")

  override = data.get("override", []) or []
  if not isinstance(override, list) or not all(isinstance(o, str) for o in override):
    raise ValueError(f"{rel}:{line}: override must be a list of strings")

  extras = data.get("extras", []) or []
  if isinstance(extras, str):
    extras = [extras]

  env = data.get("env", {}) or {}
  if not isinstance(env, dict):
    raise ValueError(f"{rel}:{line}: env must be a mapping")

  gated = bool(data.get("gated", False))
  if gated and not service and not network:
    raise ValueError(f"{rel}:{line}: gated: true requires a service: key or "
                     "network: true (the fence must name a Hub resource for "
                     "the remap to a local fixture to make sense)")

  launcher = data.get("launcher")
  if launcher is not None and launcher not in _LAUNCHERS:
    raise ValueError(f"{rel}:{line}: unknown launcher: {launcher!r} "
                     f"(expected one of {sorted(_LAUNCHERS)})")
  nproc = data.get("nproc", _DEFAULT_NPROC)
  if launcher is not None:
    if not isinstance(nproc, int) or isinstance(nproc, bool) or nproc < 2:
      raise ValueError(f"{rel}:{line}: nproc must be an int >= 2")
  elif "nproc" in data:
    raise ValueError(f"{rel}:{line}: nproc requires a launcher: key")

  name = data.get("name") or f"{rel}:{line}"
  skip = data.get("skip")
  commands = split_commands("\n".join(body_lines), launcher=launcher)

  spec = CommandSpec(
      md=rel,
      line=line,
      heading=heading,
      name=name,
      commands=commands,
      network=network,
      service=service,
      gated=gated,
      timeout=timeout,
      extra=extra,
      override=override,
      extras=list(extras),
      env=dict(env),
      cwd=data.get("cwd", "{tmp}"),
      skip=skip,
      launcher=launcher,
      nproc=nproc,
  )
  _validate_overrides(spec)
  return spec


def read_markers(md_path):
  """Parse a README, returning one :class:`CommandSpec` per marked fence."""
  md = Path(md_path)
  try:
    rel = str(md.resolve().relative_to(REPO_ROOT))
  except ValueError:
    rel = str(md)
  lines = md.read_text(encoding="utf-8").splitlines()

  specs = []
  heading = ""
  i = 0
  while i < len(lines):
    line = lines[i]
    if line.lstrip().startswith("#"):
      heading = line.strip()
    fence = _FENCE_RE.match(line)
    if fence is None:
      i += 1
      continue

    info = fence.group("info").lower()
    marker_body = _marker_above(lines, i)
    if marker_body is None:
      i += 1
      continue
    if info not in ("bash", "sh"):
      if info == "":
        raise ValueError(f"{rel}:{i + 1}: genml-cli-test marker on an "
                         "unlabeled fence; relabel it to ```bash")
      i += 1
      continue

    body, close = _collect_fence(lines, i)
    specs.append(_parse_marker(rel, i + 1, heading, marker_body, body))
    i = close + 1

  return specs


def candidate_fences(md_path):
  """Yield ``(line, info, marked, excluded)`` for every candidate fence.

  A *candidate* is a ``bash``/``sh`` fence whose body contains a core
  ``genml-kit-*`` command.  Unlike :func:`read_markers`, this
  does not skip unmarked fences: it reports whether each is opted in via a
  marker (``marked``) or opted out via an exclusion comment (``excluded``).
  """
  md = Path(md_path)
  lines = md.read_text(encoding="utf-8").splitlines()
  i = 0
  while i < len(lines):
    fence = _FENCE_RE.match(lines[i])
    if fence is None:
      i += 1
      continue
    if fence.group("info").lower() not in ("bash", "sh"):
      i += 1
      continue
    body, close = _collect_fence(lines, i)
    if fence_has_core_command("\n".join(body)):
      marked = _marker_above(lines, i) is not None
      excluded = _exclude_reason_above(lines, i) is not None
      yield i + 1, fence.group("info").lower(), marked, excluded
    i = close + 1


def _flag_index(cmd, flag):
  for idx, token in enumerate(cmd):
    if token == flag:
      return idx
  return None


# Flags that consume every following non-flag token (``nargs="+"``).  A
# single-value flag (the common case) must stop after one token, otherwise an
# override such as ``--checkpoint {tmp}/x.pt`` would swallow a trailing
# positional (block [31]'s ``genml-kit-infer ... path/to/lesion.jpg``).
_MULTI_VALUE_FLAGS = {
    "--datasets",
    "--models",
    "--lr_group",
    "--class_multipliers",
    "--freeze",
    "--model_arg",
    "--opt_arg",
    "--sched_arg",
}


def _value_end(cmd, flag_idx):
  """Return the index just past *flag* and its value token(s)."""
  if cmd[flag_idx] in _MULTI_VALUE_FLAGS:
    end = flag_idx + 1
    while end < len(cmd) and not cmd[end].startswith("-"):
      end += 1
    return end
  if flag_idx + 1 < len(cmd) and not cmd[flag_idx + 1].startswith("-"):
    return flag_idx + 2
  return flag_idx + 1


def _apply_override(cmd, entry):
  """Apply one override entry to *cmd* in place; return True if it matched.

  A non-matching command is normal: an entry is validated against the whole
  fence, and only the commands that carry the flag are edited (e.g. a fence
  whose Step 1 uses ``--datasets`` and Step 2 ``--dataset``).
  """
  if entry.startswith("-."):
    flag = entry[2:].strip()
    idx = _flag_index(cmd, flag)
    if idx is None:
      return False
    del cmd[idx:_value_end(cmd, idx)]
    return True

  tokens = shlex.split(entry, posix=True)
  if not tokens or not tokens[0].startswith("-"):
    raise ValueError(f"bad override entry: {entry!r}")
  flag = tokens[0]
  idx = _flag_index(cmd, flag)
  if idx is None:
    return False
  cmd[idx:_value_end(cmd, idx)] = tokens
  return True


def _gated_overrides(commands):
  """Return overrides remapping HF dataset flag(s) to the local fixture.

  A ``gated: true`` fence names a dataset that needs a token/access grant.
  Rather than skip it, point its ``--dataset``/``--datasets`` values at the
  synthesized ``img`` ImageFolder fixture so the example still executes
  (offline, no credentials) while the README keeps documenting the gated
  dataset verbatim.
  """
  overrides = []
  for cmd in commands:
    for flag in _GATED_DATASET_FLAGS:
      if flag in cmd:
        overrides.append(f"{flag} {_GATED_FIXTURE}")
    for flag in _GATED_COLUMN_FLAGS:
      if flag in cmd:
        overrides.append(f"{flag} {_GATED_COLUMN_FLAGS[flag]}")
  return overrides


def _validate_overrides(spec):
  """Every override entry must match at least one command in the fence."""
  for entry in spec.override:
    if not any(_apply_override(list(cmd), entry) for cmd in spec.commands):
      raise ValueError(f"{spec.md}: override {entry!r} matches no command "
                       f"in fence {spec.name!r}")


_MODULE_ENTRIES = {module: name for name, module in _ENTRY_MODULES.items()}

# Entry points that declare a ``--cache_dir`` flag: the runner injects a
# private per-test Hub cache into every command that can use one.  Kept as a
# literal set (instead of derived from the parsers) so parsing a README never
# imports the CLI modules (and transitively torch); a sync test in
# tests/test_readme_cli_coverage.py keeps it from drifting.
_CACHE_DIR_ENTRIES = frozenset({
    "genml-kit-train",
    "genml-kit-infer",
    "genml-kit-tokenizer",
})


def _split_launcher(command, launcher):
  """Split a launcher line into ``(launcher_flags, entry_name, inner_args)``.

  ``torchrun --standalone --nproc_per_node=2 -m genml_kit.cli.train --epochs 1``
  becomes ``(["--standalone", "--nproc_per_node=2"], "genml-kit-train",
  ["--epochs", "1"])``.  The entry module after ``-m`` is mapped back to its
  canonical ``genml-kit-*`` name so the shared limiter/override logic applies.
  """
  flags, i = [], 1
  while i < len(command):
    if command[i] == "-m":
      if i + 1 >= len(command):
        raise ValueError(f"launcher {launcher} line has a dangling -m")
      module = command[i + 1]
      entry = _MODULE_ENTRIES.get(module)
      if entry is None:
        raise ValueError(f"launcher {launcher} runs an unknown module: "
                         f"{module!r}")
      return flags, entry, list(command[i + 2:])
    flags.append(command[i])
    i += 1
  raise ValueError(f"launcher {launcher} line has no '-m <module>' target")


def _launcher_nproc(flags):
  """Return the ``--nproc_per_node`` value in *flags*, or None."""
  for idx, token in enumerate(flags):
    if token.startswith("--nproc_per_node="):
      return token.split("=", 1)[1]
    if token in ("--nproc_per_node", "--nproc-per-node") and idx + 1 < len(flags):
      return flags[idx + 1]
  return None


def _train_command_count(spec):
  """Number of training commands, unwrapping a launcher-headed fence."""
  count = 0
  for c in spec.commands:
    if not c:
      continue
    head = c[0]
    if spec.launcher is not None and head in _LAUNCHERS:
      head = _split_launcher(c, spec.launcher)[1]
    if head == "genml-kit-train":
      count += 1
  return count


def _rewrite_inner(spec, inner, run_root, cmd_index):
  """Apply overrides/extra/cache_dir/checkpoint logic to a core command.

  ``inner`` is ``[<genml-kit-*>, *args]`` regardless of whether the fence is
  plain or launcher-headed, so the two paths share one implementation.
  """
  overrides = (_gated_overrides([inner]) if spec.gated else []) + list(spec.override)
  for entry in overrides:
    _apply_override(inner, entry)
  if inner[0] == "genml-kit-train":
    inner.extend(spec.extra)
    if _train_command_count(spec) > 1:
      if "--checkpoint" not in inner:
        inner += ["--checkpoint", str(run_root / f"ckpt_{cmd_index}")]
      if not any(t in inner for t in ("--no-resume", "--resume")):
        inner.append("--no-resume")
  if inner[0] in _CACHE_DIR_ENTRIES and "--cache_dir" not in inner:
    inner += ["--cache_dir", str(run_root / "hf_cache")]
  return inner


def build_argv(spec, command, run_root, cmd_index=0):
  """Apply overrides/extra, inject --cache_dir, rewrite head, expand {tmp}.

  ``extra`` (the test-only limiter, e.g. ``--epochs 1``) is appended only to
  ``genml-kit-train`` commands: a fence may mix training and inference/LLM
  commands (block [31]), and the latter reject training-only flags.

  When a fence holds more than one training command (blocks [02], [29],
  [30], [31]) each gets a private ``--checkpoint`` prefix and ``--no-resume``:
  without it every command writes ``<cwd>/genml_kit_latest.pt`` and the next
  one auto-resumes its sibling's (incompatible) weights.

  A ``launcher:`` fence (``torchrun ...``) is rewritten to
  ``python -m torch.distributed.run <launcher flags> -m <module> <args>`` so it
  runs multi-process; the ``genml-kit-*`` name is recovered from the module so
  the same limiter/override/checkpoint logic applies to the inner command.
  """
  if spec.launcher is not None:
    flags, entry_name, inner = _split_launcher(list(command), spec.launcher)
    written = _launcher_nproc(flags)
    if written is not None and written != str(spec.nproc):
      raise ValueError(f"{spec.md}:{spec.line}: --nproc_per_node={written} "
                       f"disagrees with marker nproc={spec.nproc}")
    if written is None:
      flags.append(f"--nproc_per_node={spec.nproc}")
    if "--standalone" not in flags:
      flags.append("--standalone")
    inner = _rewrite_inner(spec, [entry_name] + inner, run_root, cmd_index)
    module = _ENTRY_MODULES[entry_name]
    cmd = [sys.executable, "-m", _LAUNCHERS[spec.launcher]] + flags + \
          ["-m", module] + inner[1:]
    return [str(token).replace("{tmp}", str(run_root)) for token in cmd]

  cmd = _rewrite_inner(spec, list(command), run_root, cmd_index)
  module = _ENTRY_MODULES[cmd[0]]
  cmd = [sys.executable, "-m", module] + cmd[1:]
  return [str(token).replace("{tmp}", str(run_root)) for token in cmd]


# Fixtures (synthesized under run_root)
