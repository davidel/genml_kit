# REVIEW_2 — genml_kit code & documentation audit

Second-round review of `genml_kit`. Every finding below was **reproduced**
against the working tree (not inferred from a first read). Where my
initial pass over-claimed, I record the correction in §4 so the reasoning
is auditable. This document proposes fixes only; **no production code is
changed yet** and nothing is committed.

- Validation baseline: `ruff check .` → *All checks passed!*;
  `pytest tests -q` → *930 passed*.
- Style for any follow-up change: Google Python, 2-space indent,
  `format_file` with `yapf` + `.style.yapf`, Ruff-clean, no new
  `# noqa`/Ruff-suppression comments without prior agreement.
- Interfaces note: the framework is contract/registry driven, so "this
  method/attribute looks unused" is NOT sufficient evidence of dead code
  (e.g. `Method.evaluate`, `Method.post_train`, `validate`, `build_transform`
  are reached through the driver/registry). Findings below were traced to
  their actual call sites before being called out.

Severity: 🔴 correctness bug affecting training · 🟠 real bug, narrower
reach · 🟡 robustness/latent · ⚪ docs/consistency.

---

## 1. Summary table

| # | Sev | Location | One-line |
|---|-----|----------|----------|
| 1 | 🔴 | `methods/dino.py:143–155` | DINO multi-crop split drops/mis-groups crops → hard `RuntimeError` at default sizes. |
| 2 | 🔴 | `methods/dino.py:171` + `models/dino.py` | Teacher EMA momentum pinned to `1.0`: `_total_steps` is never set, so the teacher **never** tracks the student. |
| 3 | 🔴 | `methods/base.py:58–76` | Default `evaluate()` runs `train_step`, mutating the teacher EMA + DINO center during validation. |
| 4 | 🔴 | `methods/base.py:78–84` + `train.py:436–443` | Loss-keyed methods (dino/byol/supcon/simmim/ijepa) keep maximize-direction → `_best.pt` stores the **worst** epoch. |
| 5 | 🟠 | `models/dino.py:122–123` | After #1 is fixed, teacher→student pairing still self-pairs each global crop (trivial-zero loss term). |
| 6 | 🟠 | `pipelines/images.py:443–448` | `log_validation_images` does `batch[image_column]` on a `DataBlob` → `TypeError`, silently swallowed → recon viz never logged. |
| 7 | 🟠 | `training/train.py:247–264` | Two-pass argparse fires `--help` on the 1st-pass parser → `--help` omits every pipeline/method/optimizer/checkpoint flag. |
| 8 | 🟡 | `datasets/image_folder.py:196–203` | `__getitem__` only bounds-checks the upper index; negatives pass through and get a *random* sample instead of `IndexError`. |
| 9 | 🟡 | `models/encoder_utils.py:78–82` | `raw.pooler_output` read unguarded after `hasattr(raw,"logits")`; a custom `ModelOutput` has no `pooler_output`. |
| 10 | ⚪ | README + `pipelines/images.py:119–123` | `--samples_per_class` / balanced sampler documented (incl. `scdiag/README.md`) but not in the CLI; `--sampler` choices are only `none,weighted`. |
| 11 | ⚪ | README:566–616 | Pre-training CLI defaults table drifted from the real defaults (epochs/lr/state flags). |
| 12 | ⚪ | `methods/base.py:20` vs `:METRIC_KEY` | Base declares `metric_key` (lower); driver reads `METRIC_KEY` (upper). A subclass relying on the base default crashes. |

---

## 2. Correctness bugs (🔴/🟠)

### 1. DINO multi-crop split is wrong → crash at default settings

**Code.** `MultiCropTransform.__call__` (`augmentations/multicrop.py:59–68`)
returns a list `[global1, global2, local1 … localN]`, and the images
collate (`pipelines/images.py:354–356`) stacks each position, so
`blob.data` is a tuple of `2 + local_num` tensors, **each of shape
`(B, C, S, S)`**. `DINOMethod.train_step` does:

```python
if isinstance(images, (tuple, list)):
  global_crops = images[0]                                     # ONE crop, (B,C,H,W)
  local_crops = (torch.cat(images[1:], dim=0) if len(images) > 1
                 else global_crops)                            # global2 + ALL locals
```

**Reproduction** (real transform, real collate, real `DINO`, default
`global=224/local=96/local_num=8`, `B=2`):

```
blob.data is tuple of 10 crops; shapes: [(2,3,224,224),(2,3,224,224),(2,3,96,96), …]
END-TO-END REPRO: RuntimeError -> Sizes of tensors must match except in dimension 0.
                  Expected size 224 but got size 96 …
```

Because `images[1]` (224×224) is concatenated with the 96×96 locals, the
`torch.cat` fails whenever global≠local size — which is the default. Only
the two unit tests that *deliberately* use equal sizes
(`tests/test_dino.py:161` builds a `(2,3,32,32)` "g1" only;
`test_train_step_*`) dodge the crash. When sizes happen to match it trains
but silently: only **one** global crop reaches the teacher, and `global2`
is mislabelled as a local.

**Fix — DECIDED (aligned with D-1 and D-4).** The crop split now lives in
`DINOMethod._run_loss` (the shared helper added for D-1, §2.3), which stacks
the first two positions as the global pair and the rest as locals:

```python
# methods/dino.py — _run_loss (shared by train_step / eval_step)
if isinstance(images, (tuple, list)):
  global_crops = torch.stack(images[:2]).flatten(0, 1)      # (2B, C, H, W)
  if len(images) > 2:
    local_crops = torch.stack(images[2:]).flatten(0, 1)     # (N*B, C, h, w)
  else:
    local_crops = global_crops
else:
  global_crops = local_crops = images
```

- `MultiCropTransform.split_crops` (`multicrop.py:70-74`) remains the
  canonical per-item splitter; the collate keeps stacking each position, so
  `blob.data` stays the tuple-of-stacked-tensors and `_run_loss` owns the
  flatten.
- No collate change (the earlier "...or push into the collate" alternative
  is dropped — the training-path layout `[global1, global2, local1..N]` is
  what D-4's pairing assumes, so keeping it is deliberate).
- Add a regression test using **unequal** global/local sizes end-to-end
  (defaults 224/96), which currently crashes.

**Note on interface (why this isn't caught elsewhere):** the SSL methods
own their batch decoding (`contracts.py:9–13`), so nothing upstream
validates the tuple shape — the crash surfaces only inside `train_step`.

### 2. Teacher EMA is permanently frozen (momentum pinned to 1.0)

**Code.** `methods/dino.py:164–177`:

```python
total = getattr(model, "_total_steps", 0)
if total <= 0:
  return self._momentum_end()          # 1.0 for DINO and BYOL
ratio = min(global_step / total, 1.0)
return end + (start - end) * (1.0 - ratio)
```

**Correction to my first pass:** the *interpolation formula* is correct —
I earlier claimed it was inverted, but `ratio=0→start(0.996)`,
`ratio=1→end(1.0)` (verified numerically). The real bug is the
`total <= 0` early return: `_total_steps` is **never assigned anywhere**
(`grep _total_steps` → only the two reads in dino.py:171 and byol.py:102;
no writer, no model sets it, no CLI arg). So `total` is always `0`, and
`_current_momentum` always returns the **final** momentum.

**Reproduction** (`_dino_momentum=0.996`, `_dino_final_momentum=1.0`):

```
model has _total_steps? False
_current_momentum(step=0/100/10000) -> 1.0000 1.0000 1.0000
teacher params changed after 50 EMA updates: 0 of 14
```

Identical for BYOL (`_current_momentum` at `byol.py:100–108`, same
`_total_steps` read; `byol_final_momentum` default is `1.0`). Consequence:
`update_momentum(m=1.0)` is `θ_t ← 1.0·θ_t + 0·θ_s` — a no-op. The teacher
stays a frozen copy of the **randomly-initialised** student forever, so
DINO/BYOL distil against a fixed random projector. This is the single most
damaging finding: it silently defeats the core mechanism of two methods.

**Fix — DECIDED (per review discussion).** Compute the total-step budget in
`Method.wire_data()` — the lifecycle hook that runs *after* both loaders are
built (`train.py:393-395`) and *before* `build_model` (`:396`):
`args.epochs` is available there, and `len(pipeline.train_loader)` is valid
for `Dataset`-backed loaders. The registry contract (`methods/registry.py:6,
34-38`) mandates `cls()` with **no constructor args**, so the budget cannot
be passed to `__init__` — `wire_data` is the correct seam and keeps the
registry contract intact.

```python
# methods/base.py — wire_data (existing hook, now also computes step budget)
def wire_data(self, args, pipeline):
  """Lifecycle position 2: called after both loaders are built."""
  total = None
  try:
    total = len(pipeline.train_loader)     # DataLoader.__len__ needs a sized dataset
  except TypeError:
    pass                                    # iterable-only datasets have no len()
  self._total_steps = total

# methods/dino.py — ramp driven by the stored budget
def _current_momentum(self, global_step, model):
  total = getattr(self, "_total_steps", None)
  if not total:                            # unknown → teacher STILL MOVES
    return self._momentum_start()          # fall back to start (0.996), NEVER end (1.0)
  ratio = min(global_step / total, 1.0)
  start = self._momentum_start()
  end = self._momentum_end()
  return end + (start - end) * (1.0 - ratio)
```

Notes:
- Fold `epochs` in at `wire_data`: `self._total_steps = args.epochs * len(train_loader)`
  (the ramp runs over optimizer steps if we also divide by
  `grad_accum_steps` — optional, per the run's actual step semantics).
- Unknown-total fallback is **start (0.996)**, never **end (1.0)**: a missing
  budget can no longer silently freeze the teacher.
- BYOL mirrors the same change (`byol.py:100-108`).
- The interpolation formula itself is correct (verified §2.2) — only the
  `total` source and the fallback change.
- Add a test asserting `_current_momentum(0, model) < 1.0` when
  `_total_steps` is set, and `== _momentum_start()` when it is missing; and a
  test that the teacher diverges from its init within a few EMA steps.

### 3. Default `evaluate()` performs training side-effects

**Code.** `methods/base.py:58–76` — the shared `evaluate()` calls
`self.train_step(...)` under `torch.no_grad()`. But `train_step` for DINO
and BYOL calls `model.update_momentum(...)` (`dino.py:157`,
`byol.py:…`) and `DINO.forward` calls `self.loss.update_center(...)`
(`models/dino.py:120`). Both mutate state; `no_grad()` does **not** stop
in-place `.mul_()/.add_()` updates.

**Reproduction** (base `evaluate()` over one DINO blob):

```
evaluate() metrics: {'loss': 2.07}
teacher params MUTATED during evaluate(): 6 of 14
center mutated during evaluate(): True
```

So an epoch boundary does *extra* EMA/center updates during validation,
contaminating the teacher and the run's reproducibility. (classification
and vo_pair override `evaluate`, so they're unaffected — this is a
default-implementation bug.)

**Fix — DECIDED: refined Option A (per review discussion).** Add a
`Method.eval_step()` that computes loss/metrics only (no EMA, no center);
the default `evaluate()` calls `eval_step` instead of `train_step`, and the
base default `eval_step` is `train_step` so non-SSL methods are untouched.
To avoid duplicating `train_step`, factor the shared body into a private
`_run_loss(model, images, …)` helper used by both steps. DINO additionally
needs an explicit `update_center` knob on `DINO.forward` because the center
is mutated *inside* the model forward, and `evaluate()` cannot rely on
`model.training` to gate it (it runs under `torch.no_grad()` without
switching to eval mode).

Refined shape:

```python
# methods/base.py — one-line change:
#   evaluate(): out = self.eval_step(model, blob, 0)   # was self.train_step
# base default keeps current behavior for non-SSL methods:
def eval_step(self, model, blob, global_step=0, *, labels=None):
  return self.train_step(model, blob, global_step, labels=labels)
```

```python
# methods/dino.py — shared helper + two thin steps:
def _run_loss(self, model, images, *, update_center=True):
  """Crop split + forward + LossOutput wrapping (shared by train/eval)."""
  set_train_mode(model, "train")
  if isinstance(images, (tuple, list)):
    global_crops = torch.stack(images[:2]).flatten(0, 1)      # (2B, C, H, W)
    if len(images) > 2:
      local_crops = torch.stack(images[2:]).flatten(0, 1)     # (N*B, C, h, w)
    else:
      local_crops = global_crops
  else:
    global_crops = local_crops = images
  loss, info = model(global_crops, local_crops, update_center=update_center)
  return LossOutput(
      loss=loss,
      metrics={k: (v.detach() if hasattr(v, "detach") else v)
               for k, v in info.items()})

def train_step(self, model, blob, global_step, *, labels=None):
  out = self._run_loss(model, blob.data)
  model.update_momentum(self._current_momentum(global_step, model))   # train-only
  return out

def eval_step(self, model, blob, global_step=0, *, labels=None):
  return self._run_loss(model, blob.data, update_center=False)        # side-effect-free
```

```python
# methods/byol.py — same pattern, no update_center knob:
def _run_loss(self, model, images):
  set_train_mode(model, "train")
  loss, info = model(images)
  return LossOutput(loss=loss,
                    metrics={k: (v.detach() if hasattr(v, "detach") else v)
                             for k, v in info.items()})

def train_step(self, model, blob, global_step, *, labels=None):
  out = self._run_loss(model, blob.data)
  model.update_momentum(self._current_momentum(global_step, model))
  return out

def eval_step(self, model, blob, global_step=0, *, labels=None):
  return self._run_loss(model, blob.data)
```

```python
# models/dino.py — explicit center-update knob (default keeps train behavior):
def forward(self, global_crops, local_crops, update_center=True):
  ...
  with torch.no_grad():
    t_global = self.teacher(global_crops)
    if update_center:
      self.loss.update_center(t_global)
```

Notes:
- The center gate is intentionally an explicit kwarg, **not** a
  `self.training` check, so eval semantics are visible and unit-testable.
- BYOL's momentum update already lives only in `train_step`, so its
  `_run_loss` is naturally side-effect-free.
- IJEPA updates its teacher on `on_epoch_end` (`ijepa.py:237-240`), not per
  step — unaffected by `evaluate()`; no change needed there.

Regression tests (all currently fail before the fix):
- `eval_step` over one DINO blob changes **0** teacher params and leaves
  `loss.center` untouched (mirrors the 6/14-param repro above).
- Same for BYOL.
- Existing `test_dino.py::test_train_step_returns_lossoutput` still passes
  (train path unchanged).

### 4. Loss-keyed methods save the worst checkpoint as "best"

**Code.** Base `has_metric_improved` is maximize (`base.py:78–84`) and only
VO/classification override it. The five SSL methods set
`METRIC_KEY = "loss"` (`dino.py:47`, `byol.py:21`, `supcon.py:20`,
`simmim.py:68`, `ijepa.py:131`) but inherit maximize. `_default_metric`
(`train.py:436–443`) picks the sentinel by *probing* the direction:

```python
return float("inf") if method.has_metric_improved(0.0, 1.0) else float("-inf")
```

For a loss method `has_metric_improved(0.0, 1.0)` is `0.0 > 1.0` → `False`
→ sentinel `-inf`; then `has_metric_improved(worse=1.5, better=0.5)` →
`1.5 > 0.5` → **True**. Verified:

```
DINOMethod   METRIC_KEY=loss sentinel=-inf has_improved(1.5 vs 0.5)=True
SupConMethod METRIC_KEY=loss sentinel=-inf has_improved(1.5 vs 0.5)=True
ClassificationMethod METRIC_KEY=macro_f1 sentinel=-inf has_improved(...)=True  (correct)
VOPairMethod  METRIC_KEY=mce   sentinel=inf   has_improved(1.5 vs 0.5)=False    (correct)
```

So for SSL runs with a val set (`--dataset`, which builds a val_loader),
`_best.pt` keeps the **highest-loss** epoch. Impact is bounded (resume uses
`_latest.pt`, and #2/#3 usually prevent a val_loader on pure pre-training),
but it's wrong whenever SSL + labeled data coexist (e.g. `--method supcon
--dataset …`, a documented flow in README).

**Fix — DECIDED: Option 1 (METRIC_MINIMIZE class flag, per review
discussion).** Bind the direction to the metric declaration with a single
class flag the base consults:

```python
# methods/base.py
class Method(abc.ABC):
  METRIC_KEY = "loss"
  METRIC_MINIMIZE = False      # override True in loss-keyed methods

  def has_metric_improved(self, new_metric, best_metric):
    return new_metric < best_metric if self.METRIC_MINIMIZE else new_metric > best_metric
```

Set `METRIC_MINIMIZE = True` on the five loss-keyed methods
(dino, byol, supcon, simmim, ijepa). VO may migrate from its
`has_metric_improved` override to the flag as well (optional; the override
also works). `_default_metric` (`train.py:436-443`) already probes
`has_metric_improved`, so the resume sentinel follows automatically — no
second change needed.

Rationale: declares the direction next to `METRIC_KEY` so the two cannot
drift (this is the exact class of bug that produced the `metric_key` vs
`METRIC_KEY` mismatch in §3.12); per-method overrides would require each
future loss-keyed method to remember both halves.

Regression tests:
- `has_metric_improved(1.5, 0.5)` is `False` for loss-keyed methods and
  `True` for `ClassificationMethod`.
- `_default_metric` returns `inf` for loss-keyed methods (probed via the
  current function).
- Existing `test_pipeline_method_registry.py::…metric_key_and_direction`
  still passes.

### 5. DINO teacher/student crop pairing self-pairs (unmasked by #1/#2)

`models/dino.py:113–125`:

```python
s_all = torch.cat([s_global, s_local], dim=0)
t_all = t_global.repeat((s_all.shape[0] // n_global) + 1, 1)[:s_all.shape[0]]
```

With `n_global = 2·B`, index `i`'s teacher is `t_global[i % n_global]`.
Verified for `B=2, local_num=8`: student rows 0–3 (the global crops) map to
teacher rows 0–3 — **the same crop**. DINO's cross-entropy must pair global
crop *i* with the *other* global crop *j*, never itself; a self-pair makes
those terms near-trivial. Currently invisible because #1 crashes first and
#2 keeps the teacher static, but it becomes live the moment those are fixed.

**Fix — DECIDED: Option 1 (explicit cross-crop reindex in `forward`, per
review discussion).** Replace the `repeat`+modulo table with an explicit
pairing, exploiting the collate's positional layout
([global1, global2, local1…localN]):

```python
# models/dino.py:forward — after computing s_global/s_local/t_global
# Student rows   [0:B]   = global view 1 of each image
# Student rows   [B:2B]  = global view 2 of each image
# Teacher rows   [0:B]   = teacher output of global view 1
# Teacher rows   [B:2B]  = teacher output of global view 2
#
# DINO pairing: anchor crop i must be teacher-paired with the OTHER view
# of the SAME image. Global anchors pair cross-view:
paired_global = torch.cat([t_global[B:2B], t_global[0:B]], dim=0)
# Local anchors: pair each local student row with the global-teacher row of
# its own image. Locals arrive as (N*B,) in per-crop blocks, so image i of
# a local block is at index i; use view-2 teacher (rows [B:2B]) as target.
n_local = s_local.shape[0]                      # N*B
t_local = t_global[B:2B].repeat((n_local // B) + 1, 1)[:n_local]
t_all = torch.cat([paired_global, t_local], dim=0)

loss = self.loss(torch.cat([s_global, s_local], dim=0), t_all)
```

Assumptions to lock down with a unit test:
- `global_crops` layout is `[view1 of all B images, view2 of all B images]`
  (holds once #1's `torch.stack(images[:2]).flatten(0,1)` is applied).
- `local_crops` layout is `[local crop k of all B images for k=1..N]`
  (holds once #1's `torch.stack(images[2:]).flatten(0,1)` is applied).
- A focused test asserts: no global student row is paired with its own
  teacher row (`t_all[i] != t_global[i % n_global]` for `i < 2B`), and each
  local student row `i` pairs with teacher row `B + (i % B)`.
- `update_center(t_global)` and `DINOLoss` are unchanged; only the target
  table changes.

### 6. Reconstruction/vis logging is dead on the production collate path

`pipelines/images.py:423–455` (`log_validation_images`) assumes the loader
yields a dict: `raw = batch[image_column]`. But both loader paths set
`collate_fn=_images_collate`, which returns a `DataBlob` namedtuple
(`images.py:362`). Verified:

```
collate output type: DataBlob
DataBlob['image'] -> TypeError: tuple indices must be integers or slices, not str
```

The trainer wraps the call in `try/except Exception`
(`training/trainer.py:243–244`) that only logs a warning, so
SimMIM/`--vis_every` image logging **never** appears in TensorBoard — it
just silently warns. The reason CI stays green is that
`tests/test_ensemble_collate.py::…log_validation_images_on_real_loader`
builds a `DataLoader` with the **default** collate (a dict), not
`_images_collate`, so the mismatch is never exercised.

**Fix — DECIDED: Option 7a (fold vis-logging into the method; remove
`validate` from the contract; per review discussion).** Recon
visualisation moves from a pipeline helper into a method-owned hook. This
removes the `DataBlob`/dict mismatch by construction: the method that
consumes a batch also knows its own `data` layout, so no wrapper has to
guess keys. The base provides a no-op hook — "each method decides whether it
supports it" becomes structural, no duck-checking of `validate(...) is None`.

```python
# methods/base.py — optional hook, default no-op:
def log_validation(self, model, loader, to_device, writer, global_step, device):
  """Method-owned recon-visualisation. Default: no-op (method does not
  support image logging). Override in methods that can render samples."""
  return
```

```python
# methods/simmim.py — method owns iteration + decode + writer:
def log_validation(self, model, loader, to_device, writer, global_step, device):
  blob = to_device(next(iter(loader)), device)
  images = blob.data                       # method knows its own batch layout
  with model_mode(model, "eval"), torch.no_grad():
    samples = images[:1].to(device)
    mask = make_mask(samples, model.patch_size, model.mask_ratio)
    output, _ = model(samples, mask)
    recon = unpatchify(output, patch_size=model.patch_size,
                       img_size=samples.shape[2], channels=model.in_channels)
  writer.add_image("recon/original", images[0], global_step)
  writer.add_image("recon/reconstructed", recon[0].clamp(0, 1), global_step)
```

```python
# training/trainer.py:228-244 — the wrapper try/except becomes a one-liner:
if (self.writer is not None and getattr(self.args, "vis_every", 0) > 0
    and self.epoch % self.args.vis_every == 0):
  self.method.log_validation(self.model, self.pipeline.val_loader,
                             self.pipeline.to_device, self.writer,
                             self.global_step, self.device)
```

- `pipelines/images.py:423-455` (`log_validation_images`) is **deleted**;
  its logic moves into `SimMIMMethod.log_validation`.
- `Method.validate(model, images, num_samples)` is **removed** from the
  contract: its only consumers were the helper and the tests.
- `device` again becomes a parameter of `log_validation` (the trainer owns
  it), so no module import of `device` in the method.

Tests to update/add:
- `tests/test_dino.py:187`, `tests/test_byol.py:153`,
  `tests/test_supcon_method.py:90` (`validate(...) is None`) → assert the
  base `log_validation` is a no-op instead.
- `tests/test_pretrain_vis.py` retargeted to `SimMIMMethod.log_validation`
  against a real `_images_collate` loader (this is the regression test that
  currently passes only because it uses the default collate).
- New: `log_validation` on a method without support is a silent no-op; SimMIM
  logs one `recon/original` + one `recon/reconstructed` image.

### 7. `genml-kit-train --help` under-reports the CLI

`parse_args` (`train.py:247–264`) is two-pass: it calls
`parser.parse_known_args(argv)` on the 1st-pass parser *before* adding
pipeline/method/checkpoint/optim/logging args. argparse handles `--help`
inside `parse_known_args`, so it exits there. Verified:

```
genml-kit-train --help  -> 62 lines; contains --dataset: False
                             --lr: False  --weight_decay: False
                             --checkpoint: False  --amp_dtype: False …
```

(Those flags *are* accepted at runtime — proven by a full `--dataset …
--lr …` invocation reaching model download.) The tool's own help contradicts
every README CLI table.

**Fix — DECIDED: single-pass parser registering ALL registered owners
(per review discussion).** Replace the two-pass probe with one parser that
registers every pipeline and method in the registry (not only the selected
one). Each owner already creates its own named argument group inside
`add_args` (`pipelines/images.py:92` "images pipeline", `pipelines/vo_pair.py:60`
"vo_pair pipeline", `methods/classification.py` "classification method",
`methods/dino.py:50` "DINO", `byol.py:24` "BYOL", `ijepa.py:134` "I-JEPA",
`simmim.py`, `supcon.py`); a merged parser of all owners parses `[]` with
**no option-string conflicts and no duplicate dests** (verified).

```python
# training/train.py — no probe parser, no helper, no group-name generation:
def parse_args(argv=None):
  parser = build_parser()
  for name in list_pipelines():
    get_pipeline(name)().add_args(parser)      # owner creates its own group
  for name in list_methods():
    get_method(name)().add_args(parser)        # "classification method", "DINO", ...
  args = parser.parse_args(argv)               # single pass; --help shows ALL groups
  _post_process(parser, args)
  return args
```

Consequences:
- `--help` becomes the **superset** of all registered owners' flags,
  grouped under each owner's own title — a deliberate, documented shift from
  "only my selected method's flags".
- Unselected owners' flags are accepted but inert (their defaults are
  stored; only the selected owner's args drive the run). No conflict because
  all option strings are prefixed per owner (verified).
- **Standard to enforce (add a comment):** every future owner's `add_args`
  MUST create its own named argument group before adding flags; `parse_args`
  will not namespace per-owner anymore. This is the caller-as-knowledge
  design: the owner names its own group.

Regression test: `genml-kit-train --help` output contains `--dataset`,
`--vo_length`, `--dino_local_num`, `--lr`, `--checkpoint` (covers
pipeline + method + shared flags) and the `"DINO"`/`"images pipeline"`
group titles.

### 8. `ImageFolderDataset.__getitem__` mishandles negative / out-of-range-low indices

`datasets/image_folder.py:196–203` guards only `idx >= len(self._paths)`;
negatives slip into `getitem_retry`, which on a bad `self._paths[idx]`
retries with a **random** index. Verified:

```
ds[-1] -> 2                (silently returns python "last", no error)
ds[-7] (out of range) -> 1  (random retry, NO IndexError)
```

For a torch `Dataset` an out-of-range index should raise. The random-fallback
masks genuine bugs (e.g. an off-by-one sampler) by handing back arbitrary
samples. **Fix — DECIDED: Option 1 (normalize then validate, per review
discussion).** `datasets/image_folder.py:196-203`:

```python
def __getitem__(self, idx):
  n = len(self._paths)
  if idx < 0:
    idx += n                     # Python-style negative indexing (-1 == last)
  if idx < 0 or idx >= n:
    fatal(f"Index {idx} out of range for '{self.name}'", IndexError)
  def load(i):
    return Image.open(self._paths[i]).convert("RGB")
  image, gidx = getitem_retry(idx, load, n)
  ...
```

- Keeps Python's `-1 == last` convention (legitimate callers unaffected).
- Truly out-of-range negatives now raise `IndexError` **before**
  `getitem_retry`, so a bad index can never trigger the random-fallback
  retry. `getitem_retry` remains for genuine I/O errors only.
- Account for the normalised `idx`: `load(i)` uses the *original* `idx` for
  the path and the retried index for the label, as today — only the
  pre-check changes.

Regression tests:
- `ds[-1]` returns the last sample (unchanged, Python semantics).
- `ds[-(n+1)]` raises `IndexError` (was: silent random sample).
- `ds[3]` on a 3-item set still raises `IndexError` (upper bound unchanged).
- `getitem_retry` still retries on a real I/O failure (simulate a corrupt
  file) — retry scope unchanged.

### 9. `encode_with_backbone` fallback can `AttributeError` on custom outputs

`models/encoder_utils.py:78–82`:

```python
raw = encoder(images)
if hasattr(raw, "logits"):
  if raw.pooler_output is not None:      # <-- unguarded
    return raw.pooler_output
  return raw.last_hidden_state.mean(dim=1)
```

`genml_kit`'s `ModelOutput` (`registry.py:63–74`) stores only `.logits`; it
has no `pooler_output`/`last_hidden_state`. Reached when a custom model
returns `ModelOutput` and the hook extractor raised (a `ValueError`/
`AttributeError`/`TypeError` is caught at `:74` and falls through).
Verified the crash on such a model:

```
encode_with_backbone -> AttributeError: 'ModelOutput' object has no attribute 'pooler_output'
```

**Fix — DECIDED: Option 1 (guarded getattr chain, fall back to `logits`,
per review discussion).** `models/encoder_utils.py:78-82` becomes:

```python
raw = encoder(images)
if hasattr(raw, "logits"):
  pooler = getattr(raw, "pooler_output", None)
  if pooler is not None:
    return pooler
  last_hidden = getattr(raw, "last_hidden_state", None)
  if last_hidden is not None:
    return last_hidden.mean(dim=1)
  return raw.logits                 # any .logits-only container (registry.ModelOutput)
return raw
```

- Fixes the class of bug (any `.logits`-bearing output without
  pooler/last_hidden) rather than `isinstance(raw, ModelOutput)`
  special-casing — a custom output that structurally matches but isn't the
  registry class is covered too.
- Matches `ClsModelWrapper._extract_hidden_states`'s `getattr` style
  (`cls_model_wrapper/model.py`) — the codebase's own convention.
- `ModelOutput` stays an *interface* type; no registry import added here.

Regression test: a custom `nn.Module` whose `forward` returns
`ModelOutput(logits)` and has **no** `.classifier` (so the hook path
raises) → `encode_with_backbone` returns the `logits` tensor; a custom
fake with `logits` + `pooler_output` returns `pooler_output`; one with
`logits` + `last_hidden_state` returns the mean.

---

## 3. Documentation & consistency (⚪)

### 10. `--samples_per_class` / balanced sampler are documented but not wired

README:398, :499, :511, :553, :616 and `scdiag/README.md`:64,104 tell users
to pass `--samples_per_class 16 --batch_size 64` and cite the balanced
sampler. The CLI rejects it, and the sampler isn't selectable:

```
train.py: error: unrecognized arguments: --samples_per_class 16   (rc=2)
```

`--sampler` (`images.py:119–123`) choices are `["none","weighted"]` only, and
`datasets/balanced_sampler.py` (`BalancedBatchSampler`) has **no production
call site** (grep: definition + `datasets/balanced_sampler.py` logging +
**Fix — DECIDED: wire it up (per review discussion).**

1. `pipelines/images.py:119-123` — extend the sampler flag:
   `choices=["none", "weighted", "balanced"]`, default `"none"`.
2. `pipelines/images.py` (same group) — expose `--samples_per_class`,
   `type=int, default=16`, help notes `batch_size` should be divisible by it
   (mirror README:616).
3. `pipelines/images.py:218-224` — alongside the weighted path, construct
   `BalancedBatchSampler`:

   ```python
   from genml_kit.datasets.balanced_sampler import BalancedBatchSampler

   if args.sampler == "balanced" and train_proxy.label_column:
     if args.batch_size % args.samples_per_class:
       logging.warning(
           "balanced sampler: batch_size %d not divisible by "
           "samples_per_class %d; last group will be partial.",
           args.batch_size, args.samples_per_class)
     sampler = BalancedBatchSampler(
         train_proxy.dataset[train_proxy.label_column],
         batch_size=args.batch_size,
         samples_per_class=args.samples_per_class,
         seed=args.seed,
     )
   ```

   `BalancedBatchSampler.__init__` takes `labels, batch_size,
   samples_per_class, seed` (verified `datasets/balanced_sampler.py:11-63`).
4. The existing `shuffle=sampler is None` (`:233`) already degrades
   correctly — `shuffle=False` with a sampler.
5. Guard: `train_proxy.label_column` must be non-None (same requirement as
   the weighted path); when labels are absent and `--sampler balanced` is
   passed, log a warning and fall back to `shuffle=True`.

Regression tests:
- `--sampler balanced --samples_per_class 2 --batch_size 4` builds a
  `BalancedBatchSampler` whose batches contain exactly 2 samples of each of
  2 classes (extends `tests/test_balanced_sampler.py` style, through the
  pipeline loader).
- `batch_size % samples_per_class != 0` warns but still runs.
- No labels + `--sampler balanced` → warning, `shuffle=True`.

README (`:499, :509-511, :553, :616`) and `scdiag/README.md`
(`:64, :104-105`) already document the flags — after wiring they become
accurate; no doc change needed beyond confirming wording.

### 11. Pre-training CLI Reference defaults table (README:566–616) drifted

Verified actual defaults vs the table:

| Flag | README says | Actual (`parse_args`) |
|------|-------------|-----------------------|
| `--method` | `simmim` | `classification` (default; `simmim` only when you pass it) |
| `--epochs` | `200` | `5` |
| `--lr` | `1e-4` | `3e-5` |
| `--state_save`/`--state_load` | `opt,sched` | `opt,sched,amp` |

`--image_size 448`, `--batch_size 32`, `--vis_every 0`, `--save_every 500`,
`--num_workers 4` all match. Fix the table (and note §7: until the `--help`
bug is fixed, the table is the *only* accurate flag reference).

**Fix — DECIDED: Option 1 (correct the README to actual defaults + framing
note; code stays source of truth; per review discussion).**

- `--method` → keep the row but reframe: generic default is `classification`; the
  pre-training examples explicitly pass `--method simmim`. (No code change —
  the generic default is correct for the unified harness; changing it to
  `simmim` would surprise classification users and contradict v4.2's "train vs
  pretrain are configurations" design.)
- `--epochs` → `5`
- `--lr` → `3e-5`
- `--state_save` / `--state_load` → `opt,sched,amp`
- Add a sentence under the table: "Defaults shown are the generic
  `genml-kit-train` defaults; the pre-training examples override `--method`,
  `--epochs`, `--lr` for their recipe."

README rows to touch: `README.md:566-616` (mainly `:568` method, `:590` epochs/lr,
`:585-586` state flags). No code change. Doc-only.

### 12. `metric_key` vs `METRIC_KEY` interface mismatch

`methods/base.py:20` declares `metric_key = "loss"` (lowercase), but every
consumer reads `self.method.METRIC_KEY` (`trainer.py:152,177`;
`train.py:384,404`). Verified that a subclass relying on the base default
crashes:

```
Sub().METRIC_KEY -> AttributeError: 'Sub' object has no attribute 'METRIC_KEY'
```

This is exactly the "interface" trap: the base advertises a default that the
driver never sees. **Fix — DECIDED: Option 1 (rename base `metric_key` →
`METRIC_KEY`, per review discussion).** The consumers are numerous and
consistent on uppercase (`trainer.py:152,177`, `train.py:384,404`), and all
concrete methods already define `METRIC_KEY = ...`, so the base is the lone
outlier. One-line change in `methods/base.py`:

```python
class Method(abc.ABC):
  NAME = ""
  METRIC_KEY = "loss"          # was: metric_key = "loss"
  METRIC_MINIMIZE = False      # added in D-2 (finding #4)
```

Keep a one-line comment: "the driver reads `METRIC_KEY` (class attribute);
new methods must declare it — there is no base default fallback." This
pairs with D-2's `METRIC_MINIMIZE` so the base reads consistently.
Regression test: a bare subclass that declares only `METRIC_KEY` works;
assert `f"best_{Method.METRIC_KEY}" == "best_loss"`.

### 13. Stale `pretrain.py` / `pretrain/cli.py` provenance references

`grep` finds current-tense references to a module that no longer exists
(`locate **/*pretrain*` → only `tests/`):
`io/checkpointing.py:3–4`, `utils/args.py:3`, `utils/cli.py:3`,
`training/trainer.py:5`, `datasets/factory.py:3–4`,
`datasets/transforms.py:3`, and `pipelines/images.py:142,248`. Many are
harmless historical "extracted from" notes, but `checkpointing.py:3–4`
("Both scripts import these functions") and `images.py:142` read as
present-tense and are now false. **Fix — DECIDED: Option 1 (reword only
the misleading references; keep past-tense provenance; per review
discussion).**

| File | Line(s) | Current (misleading) | Reworded (past tense / current home) |
|------|---------|----------------------|----------------------------------------|
| `io/checkpointing.py` | 3–4 | "Both scripts import these functions rather than maintaining separate copies." | "With v4.2, `genml-kit-train` is the single entry point; there is no separate `pretrain.py`." |
| `utils/cli.py` | 3 | "Used by `train.py`, `pretrain.py`, and `infer.py`" | "Used by the `genml-kit-train` / `genml-kit-infer` CLIs" |
| `training/trainer.py` | 5 | "original (train.py / pretrain/cli.py / train_vo.py)" | mark as historical: "(as they were before the v4.2 merge)" |
| `pipelines/images.py` | 142 | present-tense mixing `pretrain/cli.py` | reword to past tense: "(moved from the v4.1 pretrain CLI)" |
| `pipelines/images.py` | 248 | `# --- Ensemble path (pretrain/cli.py::build_pretrain_*) --` | `# --- Ensemble path (moved from the v4.1 pretrain CLI) --` |

No change needed (already past-tense, accurate provenance):
`utils/args.py:3` ("previously carried near-verbatim copies"), `datasets/factory.py:3-4`
("Extracted verbatim from …"), `datasets/transforms.py:3` ("Promoted from …").

Do this in the same spirit as the `plans/` reference cleanup in `e74827e`.
Doc-only; no behavior change. Optional regression: grep CI step that fails
if `pretrain.py|pretrain/cli` appears in present-tense docstrings (cheap,
future-proof).

---

## 4. Re-checks: items I flagged in my first pass that are **NOT** bugs

Listed explicitly because "looks broken" was wrong, and per your note about
interfaces:

- **DINO/BYOL momentum ramp "inverted."** Wrong — the formula
  `end + (start-end)*(1-ratio)` yields `0.996→1.0` correctly (verified).
  The real issue is §2 (the ramp never runs).
- **`normalize_args` coerces `None`→`False`.** Wrong — `utils/args.py:15`
  only converts `--amp_dtype` string→dtype; I found no None→False rewrite,
  and `load_dataset(cache_dir=…)` behaved fine.
- **`supcon_loss` "assumes 2 views / mask is B×B mismatched."** Wrong —
  SupCon has no `build_transform` override, so it uses the *single-view*
  default (`Compose`, verified) → `features (B,D)` and `labels (B,)` are
  consistent; the loss runs and returns non-zero grad (verified). The
  "2 views" comment in my read was about `has_metric_improved`/other code;
  the mask logic itself is fine.
- **`format_count` prints bytes as "PB".** Wrong — `checkpointing.py:157–158`
  returns `f"{num}"` (no suffix) below 1024; I misread the earlier truncated
  dump.
- **`has_metric_improved` defined twice in `base.py`.** Wrong — single
  definition at `:78`; my earlier "duplicate" impression came from truncated
  reads.
- **`Method.build_transform` ignores `args` → "no-op toggle".** Partly
  intentional: it's the *historic fallback* (`base.py:89–98`) and DINO
  overrides it to compose objective augmentation on top. Not a bug; the real
  DINO problems are §1/§2.
- **`Method.post_train`, `validate`, `build_transform`, `wire_data`,
  `prepare_transforms` "unused."** These are driver/registry-invoked
  contract hooks (`train.py:392–397`, `trainer.py`), not dead code — I traced
  each call site. Flagged here only to confirm they are **not** being
  reported as issues.

---

## 5. Suggested fix ordering

1. **#2 (frozen teacher)** and **#1 (DINO split)** — highest impact, unlock
   DINO/BYOL actually learning. #1 first (it's a hard crash). (#1 fix per
   D-1's `_run_loss`; #2 fix per D-3's `wire_data` budget.)
2. **#3 + #4** — validation side-effects and best-ckpt direction (coupled:
   both are SSL-loop semantics; do together with D-1's `eval_step`/`_run_loss`
   and D-2's `METRIC_MINIMIZE`).
3. **#5 DINO pairing** (do with #1, same PR/test; D-4's cross-crop reindex).
4. **#6 vis logging** (D-7: method-owned `log_validation`), **#7 `--help`**
   (D-5: single-pass registry registration) — cheap, user-facing correctness.
5. **#8, #9** — robustness guards (D-9 normalize-then-validate; D-10 guarded
   getattr chain).
6. **#10, #11, #12, #13** — docs + attribute rename + sampler wiring
   (#10 wired per D-6; #11 README defaults per D-12; #12 `METRIC_KEY` rename
   per D-8; #13 docstring reword per D-11).

Each code fix above should ship with a regression test that currently fails
(the reproductions in §2 are directly liftable into `tests/`).

**Status: plan complete — all 12 findings have a DECIDED fix. No production
code changed yet.**

## 6. Open items / next session

No open decisions remain on the 12 findings. For the implementation session:

- Decide **implementation scope/batching** (one PR for the SSL set #1–#5 vs.
  separate PRs; suggest #1/#5 + #2/#3/#4 together, then #6–#9, then docs).
- Confirm whether to add the optional grep-CI step suggested in D-11
  (present-tense `pretrain.py` reference guard).
- Confirm whether `grad_accum_steps`-aware step budget is wanted for the
  D-3 ramp (divide total by `grad_accum_steps` so the momentum anneals over
  optimizer steps).
- Re-run `ruff check .` + `pytest tests` + `format_file` (yapf, `.style.yapf`)
  after each fix batch; no new Ruff suppressions without approval.
