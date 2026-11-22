"""Fixture builders for the README CLI example suite.

``_prepare_fixtures`` lazily builds only the catalogue fixtures whose canonical
names (``img``, ``seg``, ``det``, ``hf_seg``, ``text``, ``instr``, ``pref``,
``prompts``, ``chk``, ``teacher``, ``my_loss.py`` ...) are actually referenced
by a spec's overrides / commands / env / cwd.  Unreferenced fixtures are never
built; unknown ``{tmp}/<name>`` references are left alone so simple overrides
keep working.
"""

import json
import re
import shutil
from pathlib import Path

from tests.readme_cli_harness import (
    _GATED_DATASET_FLAGS,
    _GATED_FIXTURE,
    _apply_override,
    _flag_index,
)


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


def _make_imagefolder_ddp(root):
  """A larger ImageFolder so a multi-rank run has a non-empty shard each.

  The plain ``img`` fixture (2 classes x 4 images) is too small: with two
  ranks and ``drop_last`` one rank could see an empty epoch.  8 per class (16
  total) guarantees each rank gets at least a few batches.
  """
  return _make_imagefolder(root, n_classes=2, n_per=8)


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


def _make_xgboost_model(root, model="convvit", num_labels=2):
  """A real XGBClassifier aligned with the fixture checkpoint's label space.

  ``genml-kit-infer --xgboost_model`` feeds the checkpointed backbone's
  pooled features through ``XGBClassifier.predict_proba``, and refuses to
  load a model whose stored class count disagrees with the backbone's
  ``id2label``.  So the fixture trains the classifier on features drawn
  from the *same* backbone checkpoint the fence loads, keeping the two
  models consistent by construction.

  Unlike the other ``_LITERAL_DIRS`` entries this builds a *file* at the
  exact referenced path: ``root`` is the target path (e.g.
  ``<run_root>/path/to/xgboost_model.json``), so the parent directory is
  created and the model JSON is written there.
  """
  import numpy as np
  import torch
  from genml_kit.models.registry import load_model
  from genml_kit.training.model_utils import extract_backbone_features

  out = Path(root)
  out.parent.mkdir(parents=True, exist_ok=True)
  ckpt_path = _make_inference_checkpoint(out.parent / "xgb_ckpt",
                                         model=model,
                                         num_labels=num_labels)
  id2label = {i: f"LABEL_{i}" for i in range(num_labels)}
  label2id = {name: i for i, name in id2label.items()}
  net = load_model(model,
                   num_labels=num_labels,
                   id2label=id2label,
                   label2id=label2id,
                   checkpoint_path=str(ckpt_path))
  rng = np.random.default_rng(0)
  feats = []
  with torch.no_grad():
    for _ in range(16):
      pixels = torch.tensor(rng.random((1, 3, 32, 32)) * 2 - 1, dtype=torch.float32)
      feats.append(extract_backbone_features(net, pixels)[0].numpy())
  feats = np.stack(feats)
  labels = rng.integers(0, num_labels, size=feats.shape[0])

  from xgboost import XGBClassifier
  xgb = XGBClassifier(n_estimators=4, max_depth=2, eval_metric="mlogloss")
  xgb.fit(feats, labels)
  xgb.save_model(out)
  return out


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
      "img_ddp":
          lambda: _make_imagefolder_ddp(run_root / "img_ddp"),
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
      "xgb_model.json":
          lambda: _make_xgboost_model(run_root, model=model),
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

# Literal dataset directories / files referenced directly in README commands
# (blocks [22] cmd 1, [29]).  The value is the fixture builder; it receives
# the referenced path relative to ``run_root`` and materializes the fixture
# exactly there.  ``xgboost_model.json`` is a *file*: the builder writes a
# real XGBClassifier (trained on features of the fence's own checkpointed
# backbone) at that path.
_LITERAL_DIRS = {
    "sft_corpus": _make_instr_corpus,
    "preferences": _make_preference,
    "my_text_corpus": _make_text_corpus,
    "xgboost_model.json": _make_xgboost_model,
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
