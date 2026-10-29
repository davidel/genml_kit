"""Verify the copy-pasteable CLI examples in the READMEs.

A markdown HTML comment (invisible on GitHub) immediately above a ```bash
fence opts that fence in.  Each marked fence is executed as a bounded smoke
test that writes everything under a removable temp dir.  The package is
imported via ``PYTHONPATH`` (no install); the ``genml-kit-*`` console scripts
are rewritten to ``python -m genml_kit...`` so the suite needs no editable
install.

Marker body (YAML)::

    <!-- genml-cli-test
    name: quick-start-cifar10
    network: true
    timeout: 600
    extra: --epochs 1 --max_items 16
    override:
      - --dataset {tmp}/img
      - -.--source_checkpoint
    extras: lora
    env: {HF_HOME: "{tmp}/hf"}
    -->

``network: true`` marks a fence that touches the internet.  ``service:
<name>`` additionally names an external dependency (``hf``, ``s3``, ``r2``,
``gcs``, ``github``, ``wandb``) and implies ``network: true``; the fence
then runs when that service is reachable, either through its credentials
or through an endpoint-override proxy/emulator (see ``tests/services.py``).

A fence that contains a core ``genml-kit-*`` command but must *not* be
executed opts out with a sibling comment instead::

    <!-- genml-cli-test-exclude: illustrative ellipsis, no dataset -->

``test_all_candidate_fences_marked_or_excluded`` requires every such fence to
carry one of the two comments, so a new README example cannot be added without
being either tested or explicitly excluded.

Fixture invocation is an implementation detail: :func:`_prepare_fixtures`
lazily builds only the catalogue fixtures whose canonical names
(``img``, ``seg``, ``det``, ``hf_seg``, ``text``,
``instr``, ``pref``, ``prompts``, ``chk``, ``teacher``, ``my_loss.py`` ...)
are actually referenced by the spec's overrides / commands / env / cwd.
Unreferenced fixtures are never built; unknown ``{tmp}/<name>`` references
are left alone so simple overrides keep working.
"""

import os
import re
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pytest
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
    "genml-kit-train": "genml_kit.training.train",
    "genml-kit-infer": "genml_kit.training.infer",
    "genml-kit-llm": "genml_kit.llm_cli",
}

_README_GLOBS = ["README.md", "*/README.md"]

_ALLOWED_KEYS = {
    "name", "skip", "network", "service", "gated", "timeout", "extra", "override",
    "extras", "env", "cwd"
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


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


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
  """True if any logical line of *text* starts with a core entry point."""
  for line in _logical_lines(text):
    stripped = line.strip()
    if stripped == "" or stripped.startswith("#"):
      continue
    tokens = shlex.split(stripped, posix=True)
    if tokens and tokens[0] in _ENTRY_MODULES:
      return True
  return False


def split_commands(text):
  """Split a fence body into one argv per core command."""
  commands = []
  for line in _logical_lines(text):
    stripped = line.strip()
    if stripped == "" or stripped.startswith("#"):
      continue
    tokens = shlex.split(stripped, posix=True)
    if not tokens:
      continue
    if tokens[0] not in _ENTRY_MODULES:
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
  if gated and not service:
    raise ValueError(f"{rel}:{line}: gated: true requires a service: key")

  name = data.get("name") or f"{rel}:{line}"
  skip = data.get("skip")
  commands = split_commands("\n".join(body_lines))

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


# ---------------------------------------------------------------------------
# Command building
# ---------------------------------------------------------------------------


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


def _train_command_count(spec):
  return sum(1 for c in spec.commands if c and c[0] == "genml-kit-train")


def build_argv(spec, command, run_root, cmd_index=0):
  """Apply overrides/extra, inject --cache_dir, rewrite head, expand {tmp}.

  ``extra`` (the test-only limiter, e.g. ``--epochs 1``) is appended only to
  ``genml-kit-train`` commands: a fence may mix training and inference/LLM
  commands (block [31]), and the latter reject training-only flags.

  When a fence holds more than one training command (blocks [02], [29],
  [30], [31]) each gets a private ``--checkpoint`` prefix and ``--no-resume``:
  without it every command writes ``<cwd>/genml_kit_latest.pt`` and the next
  one auto-resumes its sibling's (incompatible) weights.
  """
  cmd = list(command)
  # Gated datasets are remapped to a local fixture before user overrides so an
  # explicit override still wins (there are none in practice for gated fences).
  overrides = (_gated_overrides([cmd]) if spec.gated else []) + list(spec.override)
  for entry in overrides:
    _apply_override(cmd, entry)
  is_train = cmd[0] == "genml-kit-train"
  if is_train:
    cmd.extend(spec.extra)
    if _train_command_count(spec) > 1:
      if "--checkpoint" not in cmd:
        cmd += ["--checkpoint", str(run_root / f"ckpt_{cmd_index}")]
      if not any(t in cmd for t in ("--no-resume", "--resume")):
        cmd.append("--no-resume")
  if cmd[0] in _ENTRY_MODULES and "--cache_dir" not in cmd:
    cmd += ["--cache_dir", str(run_root / "hf_cache")]
  module = _ENTRY_MODULES[cmd[0]]
  cmd = [sys.executable, "-m", module] + cmd[1:]
  return [str(token).replace("{tmp}", str(run_root)) for token in cmd]


# ---------------------------------------------------------------------------
# Fixtures (synthesized under run_root)
# ---------------------------------------------------------------------------


def _rgba(value, size):
  import numpy as np
  return np.full((size, size, 3), value, dtype=np.uint8)


def _make_imagefolder(root, n_classes=2, n_per=4, size=32):
  from PIL import Image
  root = Path(root)
  root.mkdir(parents=True, exist_ok=True)
  for cls in range(n_classes):
    cls_dir = root / f"class{cls}"
    cls_dir.mkdir(exist_ok=True)
    for i in range(n_per):
      Image.fromarray(_rgba((cls * 40 + i * 7) % 256,
                            size)).save(cls_dir / f"img{i}.png")
  return root


def _make_segfolder(root, n=4, size=32):
  import numpy as np
  from PIL import Image
  root = Path(root)
  (root / "images").mkdir(parents=True, exist_ok=True)
  (root / "masks").mkdir(parents=True, exist_ok=True)
  for i in range(n):
    Image.fromarray(_rgba(i * 30 % 256, size)).save(root / "images" / f"{i}.png")
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[:, :size // 2] = 1
    Image.fromarray(mask).save(root / "masks" / f"{i}.png")
  return root


def _make_detfolder(root, n=4, size=32):
  import json
  from PIL import Image
  root = Path(root)
  (root / "images").mkdir(parents=True, exist_ok=True)
  images = []
  annotations = []
  categories = [{"id": 1, "name": "lesion"}]
  for i in range(n):
    name = f"{i}.jpg"
    Image.fromarray(_rgba(i * 30 % 256, size)).save(root / "images" / name)
    images.append({"id": i + 1, "file_name": name, "width": size, "height": size})
    annotations.append({
        "id": i + 1,
        "image_id": i + 1,
        "category_id": 1,
        "bbox": [0, 0, size // 2, size // 2],
        "area": (size // 2)**2,
        "iscrowd": 0,
    })
  (root / "annotations.json").write_text(json.dumps({
      "images": images,
      "annotations": annotations,
      "categories": categories
  }),
                                         encoding="utf-8")
  return root


def _make_hf_seg_dataset(root, n=4, size=32):
  import datasets
  from PIL import Image
  import numpy as np
  images = [Image.fromarray(_rgba(i * 30 % 256, size)) for i in range(n)]
  masks = []
  for _ in range(n):
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[:, :size // 2] = 1
    masks.append(Image.fromarray(mask))
  hf = datasets.Dataset.from_dict({"image": images, "annotation": masks})
  # ``SegmentationPipeline._load_hf`` reads local sources through
  # ``datasets.load_dataset`` (not ``load_from_disk``), so materialize the tiny
  # corpus as a directory of parquet shards that ``load_dataset`` can discover.
  root.mkdir(parents=True, exist_ok=True)
  hf.to_parquet(str(root / "train-00000-of-00001.parquet"))
  return root


def _make_text_corpus(root):
  import datasets
  hf = datasets.Dataset.from_dict(
      {"text": ["the quick brown fox", "jumps over the lazy dog"] * 4})
  hf.save_to_disk(str(root))
  return root


def _make_instr_corpus(root):
  import datasets
  hf = datasets.Dataset.from_dict({
      "prompt": ["What is 2+2?", "Name a colour."],
      "response": ["4", "Blue"],
  })
  hf.save_to_disk(str(root))
  return root


def _make_preference(root):
  import datasets
  hf = datasets.Dataset.from_dict({
      "prompt": ["Say hi.", "Count to two."],
      "chosen": ["Hi there!", "One, two."],
      "rejected": ["Nope.", "Two."],
  })
  hf.save_to_disk(str(root))
  return root


def _make_prompts(root):
  root = Path(root)
  root.mkdir(parents=True, exist_ok=True)
  (root / "prompts.txt").write_text("Hello\nWrite a haiku.\n", encoding="utf-8")
  return root


def _make_prompt_dataset(root):
  """HF dataset with a ``question`` prompt column (block [22] cmd 2)."""
  import datasets
  hf = datasets.Dataset.from_dict({"question": ["What is 2+2?", "Name a colour."]})
  hf.save_to_disk(str(root))
  return root


def _make_eval_corpus(root):
  """HF dataset with text/prompt/reference columns (block [23])."""
  import datasets
  hf = datasets.Dataset.from_dict({
      "text": ["the quick brown fox"],
      "prompt": ["What is 2+2?"],
      "reference": ["4"],
  })
  hf.save_to_disk(str(root))
  return root


def _is_causal_lm(model):
  """Whether *model* is a decoder-only LM rather than an image classifier."""
  lowered = str(model).lower()
  return any(tag in lowered for tag in ("gpt", "llama", "mistral", "qwen", "phi"))


def _make_inference_checkpoint(root, model="convvit", num_labels=2):
  import torch
  from genml_kit.models.registry import load_model
  root = Path(root)
  root.mkdir(parents=True, exist_ok=True)
  if _is_causal_lm(model):
    # Causal-LM backbones resolve through the ``task="causal_lm"`` registry path
    # (AutoModelForCausalLM); the image-classification path would raise on a
    # GPT2Config.
    net = load_model(model, task="causal_lm", num_labels=0)
    id2label = None
  else:
    # HF classifiers carry a pretrained 1000-entry ``id2label``; override it so
    # the fresh 2-class head is built consistently (transformers >= 5 errors
    # otherwise). ``convvit`` ignores the mapping.
    id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
    label2id = {name: i for i, name in id2label.items()}
    net = load_model(model, num_labels=num_labels, id2label=id2label, label2id=label2id)
  path = root / "model_best.pt"
  ckpt = {
      "model_state_dict": net.state_dict(),
      "epoch": 0,
      # ``load_model_for_inference`` infers the head size from these.
      "num_labels": num_labels,
  }
  if id2label is not None:
    ckpt["id2label"] = id2label
  torch.save(ckpt, path)
  return path


def _make_teacher_checkpoint(root, model="convvit", num_labels=2):
  return _make_inference_checkpoint(root, model=model, num_labels=num_labels)


def _make_custom_loss(root):
  root = Path(root)
  root.mkdir(parents=True, exist_ok=True)
  (root / "my_loss.py").write_text(
      "import torch\n\n\n"
      "def build_loss(**kwargs):\n"
      "  return torch.nn.CrossEntropyLoss()\n",
      encoding="utf-8")
  return root / "my_loss.py"


def _make_custom_model(root):
  root = Path(root)
  root.mkdir(parents=True, exist_ok=True)
  # ``--model ./my_model.py`` resolves both the model and its processor from the
  # same script: ``load_model`` calls ``create_model`` and ``load_processor``
  # calls ``create_processor`` (both are mandatory for an external script), so
  # the fixture must expose both callables.
  (root / "my_model.py").write_text(
      "import torch\n"
      "from genml_kit.models.registry import ModelOutput\n\n\n"
      "class _Tiny(torch.nn.Module):\n"
      "  def __init__(self, num_labels):\n"
      "    super().__init__()\n"
      "    self.pool = torch.nn.AdaptiveAvgPool2d(1)\n"
      "    self.fc = torch.nn.Linear(3, num_labels)\n\n"
      "  def forward(self, pixel_values, **kwargs):\n"
      "    return ModelOutput(self.fc(self.pool(pixel_values).flatten(1)))\n\n\n"
      "def create_model(num_labels, **kwargs):\n"
      "  return _Tiny(num_labels)\n\n\n"
      "def create_processor(image_size=224, **kwargs):\n"
      "  class _Proc:\n"
      "    image_mean = (0.485, 0.456, 0.406)\n"
      "    image_std = (0.229, 0.224, 0.225)\n\n"
      "  return _Proc()\n",
      encoding="utf-8")
  return root / "my_model.py"


def _make_rl_loss(root):
  root = Path(root)
  root.mkdir(parents=True, exist_ok=True)
  (root / "my_ppo_entropy.py").write_text(
      "import torch\n\n\n"
      "def build_loss(**kwargs):\n"
      "  return {'entropy_coef': 0.01}\n",
      encoding="utf-8")
  return root / "my_ppo_entropy.py"


def _spec_model(spec):
  """Return the *effective* ``--model`` (post-override), or a safe default.

  Using the effective value keeps checkpoint synthesis offline: a marked fence
  may override a network backbone (``facebook/convnextv2-...``) with ``convvit``
  and the tiny checkpoint is then built from the offline model.
  """
  for command in spec.commands:
    effective = list(command)
    for entry in spec.override:
      _apply_override(effective, entry)
    idx = _flag_index(effective, "--model")
    if idx is not None and idx + 1 < len(effective):
      return effective[idx + 1]
  return "convvit"


def _fixture_plan(run_root, model):
  return {
      "img":
          lambda: _make_imagefolder(run_root / "img"),
      "seg":
          lambda: _make_segfolder(run_root / "seg"),
      "det":
          lambda: _make_detfolder(run_root / "det"),
      "hf_seg":
          lambda: _make_hf_seg_dataset(run_root / "hf_seg"),
      "text":
          lambda: _make_text_corpus(run_root / "text"),
      "instr":
          lambda: _make_instr_corpus(run_root / "instr"),
      "pref":
          lambda: _make_preference(run_root / "pref"),
      "prompts":
          lambda: _make_prompts(run_root / "prompts"),
      "prompt_ds":
          lambda: _make_prompt_dataset(run_root / "prompt_ds"),
      "eval":
          lambda: _make_eval_corpus(run_root / "eval"),
      "chk":
          lambda: _make_inference_checkpoint(run_root / "chk", model=model),
      "chk_infer":
          lambda: _make_inference_checkpoint(run_root / "chk_infer", model=model),
      "teacher":
          lambda: _make_teacher_checkpoint(run_root / "teacher", model=model),
      "my_loss.py":
          lambda: _make_custom_loss(run_root),
      "my_model.py":
          lambda: _make_custom_model(run_root),
      "my_ppo_entropy.py":
          lambda: _make_rl_loss(run_root),
  }


def _referenced_fixtures(spec):
  gated = [f"{flag} {_GATED_FIXTURE}" for flag in _GATED_DATASET_FLAGS
          ] if spec.gated else []
  blob = " ".join(spec.extra + spec.override + gated + list(spec.env.values()) +
                  [spec.cwd] + [" ".join(command) for command in spec.commands])
  names = set()
  for match in re.finditer(r"\{tmp\}/([A-Za-z0-9_./-]+)", blob):
    names.add(match.group(1).split("/")[0])
  for match in re.finditer(r"\./([A-Za-z0-9_.-]+)", blob):
    names.add(match.group(1))
  return names


# Bare filename suffixes that denote a fixture to materialize in ``cwd`` (the
# per-test run root): a positional image (block [31]), a literal checkpoint
# such as ``genml_kit_best.pt`` (blocks [13], [31]) or a prompts file.
_FIXTURE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".pt", ".txt", ".jsonl")

# Literal dataset directories referenced directly in README commands (blocks
# [22] cmd 1, [29]).  The value is the fixture builder; ``cwd`` is run_root.
_LITERAL_DIRS = {
    "sft_corpus": _make_instr_corpus,
    "preferences": _make_preference,
    "my_text_corpus": _make_text_corpus,
}


def _make_literal_path(run_root, token, model):
  """Materialize a tiny fixture for a literal local-path command token.

  ``cwd`` defaults to ``run_root``, so a token such as ``path/to/lesion.jpg``
  or ``./sft_corpus`` lands in place.  Only the fixture suffixes and the named
  dataset directories above are handled; anything else (output prefixes,
  ``--loss`` scripts) is left to the CLI.
  """
  cleaned = token.lstrip("./")
  if not cleaned or cleaned.startswith("-"):
    return
  target = run_root / cleaned
  basename = Path(cleaned).name
  if basename in _LITERAL_DIRS:
    _LITERAL_DIRS[basename](target)
    return
  suffix = target.suffix.lower()
  if suffix in (".jpg", ".jpeg", ".png", ".bmp"):
    from PIL import Image
    target.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(_rgba(128, 32)).save(target)
  elif suffix in (".txt", ".jsonl"):
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("Hello\nWrite a haiku.\n", encoding="utf-8")
  elif suffix == ".pt" and target.parent == Path(run_root):
    # Only synthesize a *bare* checkpoint filename; a nested one (e.g.
    # ``./sft/checkpoint_best.pt``) belongs to an earlier command's output.
    shutil.copyfile(_make_inference_checkpoint(run_root / "chk", model=model), target)


def _local_path_tokens(spec):
  """Yield (token, model) for literal fixture tokens, per command.

  Overrides are applied first so the model reflects the *effective* command
  (e.g. scdiag's inference command overrides ``--model`` to ``convvit`` before
  its ``genml_kit_best.pt`` checkpoint is synthesized).
  """
  seen = set()
  for command in spec.commands:
    effective = list(command)
    for entry in spec.override:
      _apply_override(effective, entry)
    model = "convvit"
    idx = _flag_index(effective, "--model")
    if idx is not None and idx + 1 < len(effective):
      model = effective[idx + 1]
    for token in effective:
      if token.startswith("-") or "{" in token:
        continue
      basename = Path(token.lstrip("./")).name
      is_fixture = (token.lower().endswith(_FIXTURE_SUFFIXES) or
                    basename in _LITERAL_DIRS)
      if is_fixture and token not in seen:
        seen.add(token)
        yield token, model


def _prepare_fixtures(run_root, spec):
  plan = _fixture_plan(run_root, _spec_model(spec))
  for name in _referenced_fixtures(spec):
    builder = plan.get(name)
    if builder is not None:
      builder()
  for token, model in _local_path_tokens(spec):
    _make_literal_path(run_root, token, model)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def _failure_message(spec, argv, proc=None, stderr=None):
  lines = [
      f"README CLI command failed: {spec.md}:{spec.line} ({spec.heading})",
      f"argv: {' '.join(argv)}",
  ]
  if proc is not None:
    lines.append(f"return code: {proc.returncode}")
    out = (proc.stdout or "").splitlines()[-40:]
    err = (proc.stderr or "").splitlines()[-40:]
    lines.append("--- stdout (tail) ---")
    lines.extend(out)
    lines.append("--- stderr (tail) ---")
    lines.extend(err)
  if stderr is not None:
    lines.append(stderr)
  return "\n".join(lines)


def run_spec(spec, tmp_path):
  run_root = Path(tmp_path) / "readme_cli"
  run_root.mkdir(parents=True, exist_ok=True)
  try:
    for extra in spec.extras:
      module = _EXTRA_IMPORTS.get(extra)
      if module is None:
        raise ValueError(f"unknown extras key: {extra!r}")
      pytest.importorskip(module, reason=f"needs [{extra}]")
    # A gated fence is remapped to a local fixture, so it needs no service.
    if spec.service and not spec.gated:
      services.require(spec.service, os.environ)

    _prepare_fixtures(run_root, spec)

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT),
                                         env.get("PYTHONPATH", "")]).strip(os.pathsep)
    for key, value in spec.env.items():
      env[key] = str(value).format(tmp=run_root)
    cwd = spec.cwd.format(tmp=run_root)

    for index, command in enumerate(spec.commands):
      argv = build_argv(spec, command, run_root, cmd_index=index)
      try:
        proc = subprocess.run(argv,
                              cwd=cwd,
                              env=env,
                              capture_output=True,
                              text=True,
                              timeout=spec.timeout)
      except subprocess.TimeoutExpired:
        pytest.fail(
            _failure_message(spec, argv, stderr=f"timed out after {spec.timeout}s"))
      if proc.returncode != 0:
        pytest.fail(_failure_message(spec, argv, proc))
  finally:
    shutil.rmtree(run_root, ignore_errors=True)


# ---------------------------------------------------------------------------
# Parametrization
# ---------------------------------------------------------------------------


def _collect_specs():
  specs = []
  for pattern in _README_GLOBS:
    for md in sorted(REPO_ROOT.glob(pattern)):
      specs.extend(read_markers(md))
  return specs


def _param(spec):
  marks = [pytest.mark.slow]
  # A gated fence runs offline against a fixture, so it carries no
  # network/service marks and stays in the default suite.
  if spec.network and not spec.gated:
    marks.append(pytest.mark.network)
  if spec.service and not spec.gated:
    # Umbrella marker selects every service fence; the per-service marker
    # (hf, s3, ...) selects one.  service implies network (both emitted).
    marks.append(pytest.mark.service)
    marks.append(getattr(pytest.mark, spec.service))
  return pytest.param(spec, marks=marks, id=spec.name)


_SPECS = _collect_specs()


@pytest.mark.parametrize("spec", [_param(s) for s in _SPECS])
def test_readme_cli(spec, tmp_path):
  if spec.skip:
    pytest.skip(spec.skip)
  run_spec(spec, tmp_path)


# ---------------------------------------------------------------------------
# Fast unit tests (no subprocess, no network)
# ---------------------------------------------------------------------------


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
    assert argv[:3] == [sys.executable, "-m", "genml_kit.training.train"]
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
