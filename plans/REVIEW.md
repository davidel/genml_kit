# Generic Pipeline + Method Registry — Design Review (v4)

## Review Status
Reviewed against `plans/GENERIC_PIPELINE.md` (v4). Source code examined for architectural context: `training/train.py`, `training/trainer.py`, `training/vo/train_vo.py`, `pretrain/cli.py`, `pretrain/methods/base.py` (and concretions), `training/classifiers/__init__.py`, `training/optim_factory.py`. No source files modified.

Every factual claim in this review was re-checked line-by-line against the
repository (see "Verification Log" at the end). Claims whose evidence did not
hold up have been corrected in place rather than deleted, so the rationale for
each fix stays auditable.

---

## Verdict

The plan proposes a sound core architecture. The pipeline↔method split cleanly decomposes training into orthogonal concerns. The registries faithfully mirror existing patterns. Two-pass CLI parse elegantly solves dynamic flag registration.

However, **eleven critical gaps** undermine completeness. Six were in the original
review (#1, #3–#7: one arithmetic bug in the generic loop, two affecting
mixed-precision training and gradient monitoring, three affecting SSL
checkpoint/resume and per-epoch state semantics). Five more emerged while
verifying it (#12–#14, #17, #21) — and the three wiring gaps among them mean the
plan's §4 loop cannot execute as written, before any of the deeper design
questions are reached. Plus two design concerns about where the transform boundary
sits and about the shared `vo_pair` name, and four completeness items.

All these fixes are tractable; none require rethinking the overall architecture.

---

## Internal Consistency

### 1. `evaluate()` Default Contradiction 🔴

**§3.1 shows:**
```python
def evaluate(self, model, loader, device):
    raise NotImplementedError
```

**§12, item 5 ("`evaluate()` optional vs mandatory"), says:**
`plans/GENERIC_PIPELINE.md:635-637`
> **RESOLVED**: optional. Default averages `loss_fn` metrics over the loader; only VO (`mce`) and classification (macro-F1/confusion) override.

(The §3.1 code above is at `plans/GENERIC_PIPELINE.md:298-304`.)

Code says "raise." Text says "default exists." Contradiction.

Either implement the averaging default, or keep `NotImplementedError` as the contract (meaning all methods must provide custom validation logic). Both approaches work architecturally — pick one and make the code match the text (or vice versa).

---

### 2. `ModelOutput` Indirection 🟢 Minor

```python
ModelOutput = collections.namedtuple("ModelOutput", ["predictions"])
```

Note: the plan's §4 loop does **not** actually wrap anything — it calls
`out = self.model(data)` and passes `out` straight to `loss_fn`, so
`ModelOutput` is *declared in §2.2 but never constructed anywhere in the plan*.

That is worse than gratuitous indirection: it is dead contract. Either the loop
should wrap (`out = ModelOutput(self.model(data))`) or the namedtuple should be
dropped from `contracts.py`. As written, a reader cannot tell which is intended,
and the §11.1 test "Unit — contracts: namedtuple fields/defaults" would assert
the existence of a type nothing produces.

Adds no semantic value either way — direct pass of `out = self.model(data)`
works identically. If the team anticipates attaching metadata to model outputs
later, fine; otherwise delete it. Low priority, but the plan must pick one.

---

### 3. Loss Accumulation Math Is Wrong 🔴

**§4 loop shows:**
```python
for blob in self.pipeline.train_loader:
    loss = loss_out.loss / self.args.grad_accum_steps  # scaled for gradient
    loss.backward()
    ...
    total += loss.item(); batches += 1                    # ← BUG
```

Reported epoch loss = `total / batches` = sum(`raw_li / G`) / N = mean(`raw_li`) / G, where `G = grad_accum_steps`. Off by factor `G`.

This isn't cosmetic — reporting losses that are off by `grad_accum_steps` breaks log comparison, early-stopping heuristics, and user trust in TensorBoard values.

**Fix:** Separate gradient scaling from loss accumulation:
```python
accumulated_raw = 0.0
micro_count = 0
for blob in self.pipeline.train_loader:
    raw_loss = loss_out.loss                     # unmodified loss
    scaled_grad = raw_loss / self.args.grad_accum_steps
    scaled_grad.backward()                       # scale only for gradient correctness
    ...
    accumulated_raw += raw_loss.item()           # accumulate RAW loss
    micro_count += 1
epoch_loss = accumulated_raw / (micro_count / self.args.grad_accum_steps)
```

Or equivalently: report `accumulated_raw / (micro_count / G)`.

---

## Architectural Soundness Issues

### 4. Gradient Scaler / AMP Absent From Generic Loop 🔴

Current supervised code (`training/train.py:778-810`) handles AMP explicitly.
Verbatim structure (the scaler is `None` unless fp16-on-CUDA, see
`optim_factory.py:75`, so every branch tests for it):

```python
with torch.amp.autocast("cuda", dtype=amp_dtype,
                        enabled=(amp_dtype is not None and device.type == "cuda")):
    outputs = model(pixel_values=images)
    loss = criterion(logits, soft_targets) / args.grad_accum_steps

if amp_dtype == torch.float16 and scaler is not None:
    scaler.scale(loss).backward()
    if (step + 1) % args.grad_accum_steps == 0:
        scaler.unscale_(optimizer)          # ONLY when a scaler exists
        if monitor is not None:
            monitor.step(global_step)       # reads TRUE grads, pre-clip
        if args.grad_clip > 0:
            clip_grad_norm_(...)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
else:
    loss.backward()
    if (step + 1) % args.grad_accum_steps == 0:
        if monitor is not None:
            monitor.step(global_step)
        if args.grad_clip > 0:
            clip_grad_norm_(...)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
```

Plan's §4 loop has none of this:

```python
loss.backward()               # ← no autocast context!
clip_grad_norm_(...)          # ← no unscale before clip!
optimizer.step()              # ← no scaler involvement!
```

Two concrete breakages:

1. **AMP silently degrades to fp32 (and fp16 becomes unsafe).** `self.model(data)` in §4 is called outside any `autocast` context, so the forward pass runs in float32 regardless of `--amp_dtype` — the flag becomes a no-op. Worse, if fp16 AMP is requested, `optim_factory.build_optimization` still constructs a `GradScaler` (`optim_factory.py:75-77`); the plan then calls `optimizer.step()` directly instead of `scaler.step()`, so the scaler never participates and fp16 inf/nan protection is lost.

2. **Gradient monitoring reads wrong values.** Current code carefully sequences `unscale → monitor.step → clip_grad_norm`. Monitor needs true gradient magnitudes (post-unscale, pre-clip). The §4 loop has no monitoring at all. Combined with no `unscale_` call, any gradient monitoring code would read clipped (incorrect) gradients.

The plan's `BaseTrainer.__init__` receives an `optimization` named tuple (`from optim_factory.py`) with fields `(param_groups, optimizer, scheduler, scaler)`. The `.scaler` field exists but is completely unused in §4.

**Fix:** Add the full AMP/scaler sequence to §4, mirroring
`train.py:778-810`. Note the forward pass must be *inside* the autocast
context (the plan's §4 computes `out` and the loss outside it), and use the
non-deprecated `torch.amp.autocast("cuda", ...)` form already used in this
codebase — `torch.cuda.amp.autocast()` is deprecated in the installed torch:
```python
accum = self.args.grad_accum_steps
with torch.amp.autocast("cuda", dtype=self.args.amp_dtype,
                        enabled=(self.args.amp_dtype is not None
                                 and self.device.type == "cuda")):
    out = self.model(data)
    loss_out = self.method.loss_fn(out, blob)
loss = loss_out.loss / accum

scaler = self.optimization.scaler
if scaler is not None:
    scaler.scale(loss).backward()
else:
    loss.backward()

if (step + 1) % accum == 0:
    if scaler is not None:
        scaler.unscale_(self.optimization.optimizer)
    if monitor is not None:                     # true grads: post-unscale,
        monitor.step(step)                      # pre-clip
    if self.args.grad_clip > 0:
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                       max_norm=self.args.grad_clip)
    if scaler is not None:
        scaler.step(self.optimization.optimizer)
        scaler.update()
    else:
        self.optimization.optimizer.step()
    self.optimization.optimizer.zero_grad(set_to_none=True)
```
Two details easy to get wrong here: `optimizer.step()` must be skipped on the
steps where the scaler found inf/nan (that is what `scaler.step` handles), and
`zero_grad` must stay inside the accumulation guard.

---

### 5. Scheduler Placement Missing 🔴

Both existing callers step the LR scheduler once per epoch, guarded against
`None` (`build_optimization` only returns a scheduler when one is configured):

- `training/train.py:1213-1214` — inside `ClassificationTrainer.train_epoch`,
  immediately after `train_one_epoch(...)` returns.
- `pretrain/cli.py:713-714` — inside `PretrainTrainer.train_epoch`, at the end.

Neither lives in `BaseTrainer.run()`. The plan's §4 `train_epoch` body never
invokes it, and §4 shows only `train_epoch` + `validate` plus an
`__init__(self, args, model, method, pipeline, optimization, device, writer, ...)`
signature — `run()` is not shown at all, so there is nowhere in the plan where a
`scheduler.step()` could plausibly occur.

If the scheduler is accepted but never used, learning rate annealing is silently
broken: the LR stays flat at the base value for the whole run for anyone passing
`--scheduler CosineAnnealingLR` / `--scheduler StepLR` (`pretrain/cli.py:512-517`;
default is `None`, so the regression only bites users who *did* configure one —
and nothing warns them). Note this is a *behaviour* regression, not a crash — it
will pass any test that only checks "training runs", which is exactly why it must
be spelled out in the plan.

**Fix:** State the ownership explicitly. `BaseTrainer.run()` already owns the
per-epoch sequence (`trainer.py:184-191`: `train_epoch` → best-checkpoint →
`epoch_end`); add the scheduler step there so it cannot be forgotten by each
subclass:
```python
loss, step = self.train_epoch(epoch, saver, step, monitor)
if self.optimization.scheduler is not None:      # per-epoch, as today
    self.optimization.scheduler.step()
...
self.epoch_end()
```
If instead the step stays in the subclass `train_epoch` (matching today's code),
say so in §4 — otherwise the generic loop has two plausible homes for it and the
likely outcome is zero.

---

### 6. Checkpoint State Hooks Removed From Method 🔴

`PretrainMethod` has `get_checkpoint_state(model, args)` / `load_checkpoint_state(model, state, args)`. These persist SSL-critical internal state:

Verified against each method's `get_checkpoint_state`. Exact payload:

| Method | State preserved | Purpose |
|--------|----------------|---------|
| BYOL | `momentum` (0.996), `final_momentum` (1.0) | Teacher EMA ramp bounds (`byol.py:93-98`) |
| DINO | `momentum`, `final_momentum` **+ `model.loss.center.clone()`** | EMA ramp + the running teacher-output center that prevents collapse (`dino.py:162-171`) |
| SupCon | `proj_dim`, `proj_hidden`, `temperature` | Architecture/hparam record — head itself is in the model state dict (`supcon.py:57-63`, `load` is a `pass`) |
| IJEPA | `teacher_momentum`, `teacher_final_momentum`, `embed_dim`, `predictor_dim`, `predictor_heads` | EMA ramp + architecture record (`load` is a `pass`, `ijepa.py:198-210`) |

Reshaped `Method` (§3.1) has NO checkpoint hooks. Yet `BaseTrainer.__init__` clearly manages checkpoint persistence (saves/restores model, optimizer, etc.). There's nowhere for method-level state to live.

**Impact:** SSL checkpoints save model/optimizer/scheduler but NOT momentum factors or centers. On resume, trained models reset their momentum ramps — losing weeks of incremental training progress. The trained state effectively restarts. DINO's case is the most severe: the `center` is a running average of teacher outputs, so restoring momentum without the center re-introduces the collapse the center exists to prevent.

**Fix:** Do not invent new names — the mechanism already exists and the plan
should just carry it forward. `BaseTrainer` already exposes two save-time hooks
(`training/trainer.py:121-128`):

```python
def saver_extra(self):   # extra state attached to EVERY checkpoint write
  return {}
def ckpt_extra(self, best, step):   # best-checkpoint / exit saves only
  return {}
```

`PretrainTrainer` wires the first one to the method today
(`pretrain/cli.py:720-721`):
```python
def saver_extra(self):
  return {"method_state": self.method.get_checkpoint_state(self.model, self.args)}
```
and resume restores it via `method.load_checkpoint_state(model, method_state, args)`
(`pretrain/cli.py:777-778`). The generic `BaseTrainer` in plan §4 must keep that
pair verbatim, so `Method` needs `get_checkpoint_state`/`load_checkpoint_state`
(abstract, as in `methods/base.py:59-70`) or at minimum documented optional hooks.

Also note `SimMIMMethod.load_checkpoint_state` carries a backward-compat branch
for legacy `_mask_ratio` checkpoints (`methods/simmim.py:113-124`) — deleting the
hook silently drops that migration path.

---

### 7. Epoch-End Hook Removed 🔴

The reshaped `Method` ABC (§3.1) drops **every** lifecycle hook that
`PretrainMethod` has. `plans/GENERIC_PIPELINE.md` contains zero occurrences of
`epoch_end` or `on_epoch_end`, and §3.1 lists only
`add_args / build_model / loss_fn / evaluate / metric_key / has_metric_improved`.
The hooks being lost are:

| Hook on `PretrainMethod` | Who implements it | What it does |
|---|---|---|
| `on_epoch_end(model, epoch, writer)` | IJEPA (`ijepa.py:212-215`) | Ramps teacher EMA: `model.update_momentum(epoch)` |
| `on_epoch_end(model, epoch, writer)` | BYOL, DINO, SupCon | **Literal `pass`** — no-op today (`byol.py:90`, `dino.py:159`, `supcon.py:71`) |
| driven by | `pretrain/cli.py:717-718` → `BaseTrainer.epoch_end()` (`trainer.py:117-119`, called at `trainer.py:191`) | |
| `build_transform(...)` | BYOL, DINO, vo_pair | Objective-specific augmentation (see #8) |
| `get/load_checkpoint_state` | BYOL, DINO, SimMIM, SupCon, IJEPA | See #6 |

**Correction to the original version of this review:** BYOL and DINO do *not*
ramp momentum in `on_epoch_end` — their bodies are `pass`. Their ramp is computed
**per optimizer step** inside `train_step` via
`_current_momentum(global_step, model)` → `model.update_momentum(momentum)`
(`byol.py:68-72`, `dino.py:125-135`), interpolating over `global_step /
model._total_steps`. So the per-epoch hook that actually matters is **IJEPA's**,
not BYOL's.

That makes the loss larger, not smaller: the §3.3 reshape of `train_step` into
`loss_fn(model_output, blob)` has **no `global_step` parameter and no place for
the `set_train_mode` / `update_momentum` side effects** BYOL and DINO perform.
See new item #15 — a momentum EMA that is frozen at its initial value means the
teacher never tracks the student, which is a correctness failure for those
methods, not merely "likely inferior results".

**Fix:** Restore the lifecycle hooks explicitly in §3.1 and show where §4/§5.1
call them. `BaseTrainer.run()` already has the right slot — it calls
`self.epoch_end()` at `trainer.py:191` after `validate()` and the best-checkpoint
save; the generic version should route it to the method:
```python
def epoch_end(self):                       # BaseTrainer, generic
  self.method.on_epoch_end(self.model, self.epoch, self.writer)
```
Keep it non-abstract with a documented no-op default, exactly as
`methods/base.py:93-99` does today (that file marks it `# noqa: B027` to signal
"intentionally empty, not forgotten").

---

### 8. Transform Ownership Blurs Pipeline↔Method Boundary ⚠️

The plan states in §6.1 (`plans/GENERIC_PIPELINE.md:479-481`):

> "**Transform**: the pipeline owns the transform (`build_transforms`/
> `DictFieldTransform` with `fields=("image",)`), NOT the method.
> `Method.build_transform` (used by SimMIM etc.) moves into the pipeline's
> data-transform responsibility."

(The earlier version of this review also cited "item 11 in Open Items" — §12 has
only 8 items and none concern transforms, so that citation was wrong; §6.1 is the
only place this decision is recorded.)

But some transforms are objectively objective-defining:

| Transform | Classification | BYOL | DINO | SimMIM |
|-----------|---------------|------|------|--------|
| Random crop+flip | Preprocessing ✅ | Preprocessing ✅ | Preprocessing ✅ | Preprocessing ✅ |
| Normalization | Preprocessing ✅ | Preprocessing ✅ | Preprocessing ✅ | Preprocessing ✅ |
| Dual-view augmentation | N/A | Objective-defining ❌ | Objective-defining ❌ | N/A |
| Multi-crop strategy | N/A | N/A | Objective-defining ❌ | N/A |
| Mask generation | N/A | N/A | N/A | Objective-defining ❌ |

These are real classes today, and they are all method-owned via
`PretrainMethod.build_transform` (`methods/base.py:72-81`, whose own docstring
says "Override in subclasses that need custom augmentations (e.g. dual-view for
BYOL, multi-crop for DINO)"):

| Object | Lives in | Used by |
|---|---|---|
| `DualViewTransform` | `pretrain/augmentations/dual_view.py:4` | `byol.py:58` (`build_transform`) |
| `MultiCropTransform` | `pretrain/augmentations/multicrop.py:18` | `dino.py:94` (`build_transform`) |
| `make_mask()` | `pretrain/methods/simmim.py:11` | `simmim.py:97` inside **`train_step`**, not a transform |

So the sentence "`Method.build_transform` (used by SimMIM etc.)" in §6.1 is
doubly wrong: SimMIM does **not** implement `build_transform` (it inherits the
base default), and its masking is not a transform at all — `make_mask` runs per
step on the batch tensor inside `train_step`.

Moving dual-view composition and multi-crop into `ImagesPipeline` means the
pipeline emits fundamentally different data structures depending on which method
consumes it:
- Regular images `(B,C,H,W)` for classification
- `(view_a, view_b)` for BYOL, `(global_crops, local_crops)` for DINO (today these
  come back from the transform and are unpacked by the *model's* forward)
- `(B,C,H,W)` + a per-step mask for SimMIM (unchanged by §6.1 — masking is not a
  transform)

This undermines the reason for separating pipeline from method. The boundary
should guarantee that pipelines produce consistent data shapes regardless of
method, while methods decide how that data is consumed.

A concrete counter-example from the plan itself: §2.5's `ImagesPipeline.add_args`
registers **no** augmentation flags at all, so there is no mechanism by which the
pipeline could specialise per method — the review's earlier phrasing ("varies by
method type even within `ImagesPipeline`") overstated what §6.1 actually
mandates. The real risk is the opposite: because `Method.build_transform` is
declared to move to the pipeline while no pipeline API replaces it, **BYOL's and
DINO's objective-specific augmentation has nowhere to live** and would silently
degrade to the generic `build_transforms` path.

**Better approach:** Keep objective-specific augmentation (views, crops) in the
method layer; move only generic preprocessing (resize, normalize, basic
crop/flip/jitter) into pipelines. Alternatively, make it a two-phase contract and
say so in §2.3/§3.1:
1. **Pipeline transform**: generic preprocessing, uniform across methods.
2. **Method transform**: objective-specific augmentation, composed *on top* by the
   pipeline (`pipeline.build_loader` calls `method.transform()` when the method
   declares one).

Either way, §6.1 needs an explicit answer for where `DualViewTransform` and
`MultiCropTransform` end up; right now it accidentally deletes them.

---

### 9. Supervised vs Self-Supervised VO Share `vo_pair` Name ⚠️

Two paths (supervised and self-supervised VO) register under the *same* name
`vo_pair` (pipeline `name = "vo_pair"`, `VOPairMethod.NAME = "vo_pair"`),
distinguished only by `--vo_stage`. Today that flag is defined inconsistently in
the two places that touch it, and the plan makes it worse rather than resolving
it:

| Location | Declaration | Values |
|---|---|---|
| `pretrain/methods/vo_pair.py:26-29` | `type=int, default=1` | `0` / `1` |
| `training/vo/train_vo.py:19,184` | consumed as `getattr(args, "vo_stage", STAGES["supervised"])` | `STAGES = {"supervised": 0, "photometric": 1}` |
| plan §5.2 / §8 (`:450`, `:531`) | `--vo_stage photometric` | string |

So the plan's own examples pass a **string** to a flag that is currently parsed as
`int` (`--vo_stage 1`). One of the two must change, and the plan never says which.
Worse, `VOPairMethod.add_args` uses a bare `parser.add_argument` (not a group) and
`train_vo.py`'s supervised path reads the same name — merging both into one binary
with one flag namespace makes an accidental re-registration (`argparse` `conflicting
option string` error) likely unless §5.1 explicitly deduplicates it.

Beyond the mechanics: name→objective mapping is indirect. A user typing
`--method vo_pair` cannot tell whether they get a supervised mce-trained model or a
photometric-SSL encoder, and §3.4's table lists both under one row-set. Consider
distinct names (`vo_supervised` / `vo_ssl`), keeping `--vo_stage` only for the
in-method loss-weight schedule, or at minimum document the branching in §3.1 and
fix the string-vs-int type in §5.2.

---

## Completeness Gaps

### 10. No Pipeline↔Method Compatibility Validation 🟡

Resolved in §12-Q3 as "none. Any combination allowed." But several combinations produce deterministic runtime failures:

| Pairing | Failure Mode |
|---------|-------------|
| `images` × `vo_pair` | `VOPairMethod.loss_fn` expects `blob.data` as `(image_a, image_b)` tuple; gets `(B,C,H,W)` tensor |
| `vo_pair` × `simmim` | SimMIM forward passes single images through masked encoder; receives paired geometry batches |
| `vo_pair` × `dino` | DINO expects individual images; receives structured pairs |

Note first: this is a **deliberate user decision**, recorded at
`plans/GENERIC_PIPELINE.md:630-631` ("**RESOLVED**: none. Any combination allowed;
no `fatal` on mismatch (per user decision)"). It is listed here as residual risk,
not as something to silently "fix" against that decision — implementers must not
re-add a `fatal` pairing check without re-asking.

Given that constraint, the cheap mitigation that does *not* violate the decision is
to let failures surface as *informative* errors instead of raw shape mismatches —
e.g. `loss_fn` raising `KeyError: 'gt'` inside step 1 of epoch 0 is technically a
failure but reads as a bug. Two options consistent with "no validation at startup":

1. Have `DataPipeline` declare a `data_spec` (shape/type description) and have
   `Method.loss_fn` raise a message naming both sides
   (`"vo_pair method needs pipeline 'vo_pair'; got pipeline 'images' yielding
   (B,C,H,W) tensor"`). Still no startup `fatal`.
2. Print the resolved `pipeline × method` pair with its blob contract at startup
   (§2.4 already logs `Pipeline kwargs`), so a mismatch is obvious before training
   begins.

Either keeps the user's decision while removing the "debug a tensor error 40 minutes
into a job" cost.

---

### 11. Monitoring Integration Lost 🟡

Current code carefully sequences: `unscale → monitor.step → clip_grad_norm`. Monitor needs true gradient magnitudes (post-unscale, pre-clip). The §4 generic loop omits monitoring entirely. Combined with missing scaler (#4 above), gradient monitoring is broken in practice.

Reinsert the `unscale → monitor → clip` sequence alongside the scaler fix —
the corrected snippet in #4 now includes it. Both call sites do agree on passing
the cumulative `global_step` (`train.py:804,811` and `pretrain/cli.py:236,242`),
so the plan should keep that convention rather than the per-epoch `step` §4's
`monitor` parameter name invites.

Two further discrepancies the merge must resolve, both currently invisible in §4:

- **Epoch-end flush.** *Both* existing loops step on
  `(idx + 1) % grad_accum_steps == 0 or (idx + 1) == total_batches`
  (`train.py:798`, `pretrain/cli.py:230`). Plan §4 dropped the second clause and
  has only the modulo test, so for a loader whose length is not a multiple of
  `grad_accum_steps` the trailing micro-batches never step **and their gradients
  leak into the next epoch** (no `zero_grad` fires either). Restore the
  `or (step + 1) == total_batches` clause — which means `train_epoch` needs
  `len(self.pipeline.train_loader)`, not just the iterator §4 shows.
- **Clip magnitude.** `train.py:806` uses `max_norm=args.grad_clip` guarded by
  `if args.grad_clip > 0`; `pretrain/cli.py:238` hardcodes `max_norm=1.0`
  unconditionally. One generic loop cannot have both — §4 should state that
  `args.grad_clip` wins (and note that this changes pretrain's effective clipping
  when `--grad_clip` differs from 1.0).

---

## Additional Gaps Found During Verification

The following were **missed by the review as originally written** and were added
while verifying each claim against the source. Numbering continues from #11.

### 12. Pipeline Loaders Are Never Wired 🔴

§4 iterates `self.pipeline.train_loader` and `validate()` reads
`self.pipeline.val_loader`, but §2.3's `DataPipeline` ABC defines only
`build_loader(args, mode="train")` — there is no `train_loader`/`val_loader`
property or attribute anywhere in the plan. Meanwhile §5.1 creates them as
*locals* and never passes them to the trainer:

```python
train_loader = pipeline.build_loader(args, mode="train")
val_loader   = pipeline.build_loader(args, mode="val")
...
BaseTrainer(args, model, method, pipeline, optimization, device, writer).run()
```

Both loaders are dropped on the floor. As written, `BaseTrainer.run()` raises
`AttributeError` on the first line of the first epoch. This is the single most
load-bearing omission in §4/§5.1 — every other fix in this review sits on top of
it.

**Fix:** pick one and state it in §2.3: either (a) `build_pipeline`/§5.1 calls
`pipeline.build_loader(...)` and the pipeline caches them as
`self.train_loader`/`self.val_loader` (matching what §4 already assumes), or
(b) `BaseTrainer` owns construction —
`self.train_loader = pipeline.build_loader(args, "train")` in `__init__` — and
§4's body reads `self.train_loader`. (b) is closer to today's code
(`ClassificationTrainer` receives a `DataBundle`, `train.py:1189+`) and keeps the
pipeline stateless, which matters because `build_loader` also has to be callable
for `len(loader)` (see #11's epoch-end flush note).

---

### 13. `blob.meta` Stays on CPU 🔴

§4 moves only the data half of the blob:

```python
data = self.pipeline.to_device(blob.data, self.device)
out = self.model(data)
loss_out = self.method.loss_fn(out, blob)     # <-- receives the ORIGINAL blob
```

`loss_fn` is handed the *untouched* `blob`, whose `meta["labels"]` is a CPU tensor.
Supervised today moves labels explicitly (`train.py:770`:
`targets = targets.to(device, non_blocking=True)`), and VO's `meta` carries GT
similarity/terrain/residual tensors (`VOPairDataset.__getitem__` returns a dict of
tensors). Any method comparing `out` against `blob.meta` gets a device mismatch
`RuntimeError`.

Also note the plan moves only `blob.data` but hands `loss_fn` the whole `blob`, so
the moved copy is invisible to the loss. The contract needs to be
`blob = pipeline.to_device(blob, device)` returning a blob with both halves moved
(namedtuple `_replace`), not `to_device(blob.data)`.

**Fix:** change §2.3's signature to `to_device(self, blob, device) -> DataBlob`
and §4 to `blob = self.pipeline.to_device(blob, self.device)`. §6's blob table
already lists `meta` contents per pipeline, so the information is there — only the
movement rule is missing.

---

### 14. Objective-Specific Model Lifecycle Has No Home in `loss_fn` 🔴

§3.3 reshapes `train_step(self, model, images, global_step, *, labels=None)` into
`loss_fn(self, model_output, blob)`. That signature drops **both** `model` and
`global_step`, but four methods do per-step work that needs them:

| Method | Per-step side effect in `train_step` | Needs |
|---|---|---|
| BYOL | `set_train_mode(model, "train")`; `model.update_momentum(self._current_momentum(global_step, model))` (`byol.py:68-72`) | `model`, `global_step` |
| DINO | same, incl. global/local crop unpacking (`dino.py:125-135`) | `model`, `global_step` |
| IJEPA | `set_train_mode(model, "train")`; block masking runs inside `IJEPA.forward` | `model` |
| SimMIM | `make_mask(images, model.patch_size, model.mask_ratio)` then `model(images, mask)` (`simmim.py:96-98`) | `model`, and a **two-arg** forward |

The plan's own §3.3 "before/after" example is `VOPairMethod`, which happens to need
neither — that is precisely why the gap is invisible in the plan. Note also that
`model_output` is produced by §4 *before* `loss_fn` is called, so a method cannot
even re-run its own forward with masking applied (SimMIM's `model(images, mask)`
takes two args and §4 calls `self.model(data)` with one).

**Fix:** give the loop a hook that owns the step, e.g.
`method.train_step(self, model, blob, global_step) -> LossOutput` (the existing
`PretrainMethod.train_step` signature, `methods/base.py:43`) with §4 calling it
*inside* the autocast context, and keep `loss_fn` as the pure-computation helper
that `train_step` defaults to. This also resolves the `model(images, mask)`
multi-arg problem, since the method controls its own forward.

---

### 15. `Model.train()` / Eval-Mode Handling Missing 🟠

§4 never puts the model in train mode. Existing loops do it themselves:
`train_vo.py:181` `self.model.train()`; BYOL/DINO call
`set_train_mode(model, "train")` per step (`byol.py:69`); SimMIM/IJEPA use
`model_mode(...)` context managers. Dropping it means dropout/BN behaviour depends
on whatever mode `build_model` left the model in — `nn.Module`s default to
train mode, so this "works" by accident and then breaks the first time a method
constructs a sub-model in eval mode (the momentum teacher).

**Fix:** state whether the loop or the method owns mode. If the loop:
`self.model.train()` at the top of `train_epoch`, `model.eval()` around
`evaluate()`. If the method: add `Method.train_mode(model)` mirroring today's
`set_train_mode`, and say so in §3.1.

---

### 16. Two-Pass Parse Ordering Is Underspecified 🟠

§1.3 quotes `pretrain/cli.py:540-543` and §5.1 shows
`pipeline_cls().add_args(parser)` then `method_cls().add_args(parser)`. But:

- `add_args` is declared `@classmethod` in §2.3/§3.1 while §4/§5.1 instantiate
  `pipeline_cls()` / `method_cls()` first — inconsistent (today's methods mix both:
  `byol.py:21` / `dino.py:50` are `@classmethod`, `simmim.py:65`, `supcon.py:20`,
  `ijepa.py:129`, `vo_pair.py:26` are instance methods). A `@classmethod`
  declaration called on an instance works, but the reverse (instance method called
  on the class) does not.
- `build_method(name, **kwargs)` in §3.2 takes kwargs, but §5.1 calls
  `method_cls()` with no args, and §2.4's `build_pipeline(name, **kwargs)` is
  never shown being called with the parsed flags either. Where do pipeline
  `__init__` args (`VOPairPipeline.__init__(self, length, val_length, size,
  pitch_deg, seed)`) come from? Today's VO CLI constructs them from `args`.
- Duplicate flag registration: both `ImagesPipeline.add_args` and the
  `classification` method may declare `--image_size`; §6.1 assigns
  `--image_size`/`--batch_size`/`--num_workers` to the **pipeline**, while the
  methods being ported already declare `--image_size` (`simmim.py:65-74`). The
  merge needs an explicit "method must not re-declare pipeline flags" rule or an
  `parse_known_args` conflict guard.

**Fix:** make `add_args` uniformly `@classmethod`, show the kwargs plumbing in
§5.1, and add a flag-ownership rule to §6.1.

---

### 17. Plan's §3.4 Still Invokes the Deleted Binary 🔴

§5.2/§9/§12 delete `genml-kit-pretrain` entirely, but §3.4 lists it as live usage:

```
plans/GENERIC_PIPELINE.md:352: - `genml-kit-pretrain --method vo_pair --pipeline vo_pair ...`  (self-supervised)
plans/GENERIC_PIPELINE.md:353: - `genml-kit-pretrain --method simmim  --pipeline images ...`   (existing UX)
```

(§5.2's and §8's uses at `:445`, `:449`, `:540` are fine — they are explicitly
prefixed "formerly".) As an internal-consistency review item this belongs in the
"Internal Consistency" section alongside #1; it is the only place in the plan where
a resolved decision is contradicted by unedited prose, and an implementer copying
§3.4 into a test or doc string would produce a command that cannot run.

**Fix:** rewrite `:352-353` as `genml-kit-train --method vo_pair --pipeline vo_pair --vo_stage photometric`
and `genml-kit-train --method simmim --pipeline images`, per §8.

---

### 18. §10 Risks Table Is Stale 🟡

`plans/GENERIC_PIPELINE.md:573` still reads:

> Circular imports | `pipelines` imports `train_vo` **lazily inside
> `loss_fn`/methods**

but `loss_fn` is a **Method** concept in v4 and does not exist in `train_vo.py`
today (which has `train_step`/`validate`) — this is v3 wording that survived the
rewrite, and it also contradicts §9's "`VOTrainer` deleted, loss moves to
`vo_pair` method". The same table's row "Existing tests construct trainers |
update to `BaseTrainer(model, method, pipeline, ...)`" gives a constructor
signature that contradicts §4's
`__init__(self, args, model, method, pipeline, optimization, device, writer, ...)`
(`args` missing).

**Fix:** refresh §10 and make the two signatures agree.

---

### 19. Documentation Inventory Understated 🟡

§12 item 7 cites `README.md:492,518,632` and "`scdiag/scripts/*.py` usage
prints". Verified against the tree, the full set is larger:

- `README.md`: **492, 518, 591, 632, 1061** (the plan misses `:591` — "genml-kit-pretrain stitches multiple datasets" — and `:1061` — "Both `genml-kit-train` and `genml-kit-pretrain` accept the flag").
- `scdiag/README.md`: **27, 48, 57, 72, 78** (lines 57 and 72 are live
  `genml-kit-pretrain` invocations; the plan cites only `scdiag/scripts/*.py`, not
  `scdiag/README.md`).
- `scdiag/scripts/prepare_isic.py`: 6, 201; `prepare_derm1m.py`: 7, 14, 216.
- `vo/README.md`: **no matches** — the plan's "`vo/README.md` references" is
  unfounded (`grep -rn pretrain vo/` is empty), so that clause should be dropped
  or replaced with whatever `vo/README.md` actually documents.

**Fix:** re-derive the list with `grep -rn "genml-kit-pretrain"` at implementation
time rather than trusting the cited line numbers; note `--pipeline`/`--method`
also need adding to the docs' flag tables, not just the invocation lines.

---

### 20. §9 File Inventory Misses Two Things 🟡

1. **`pretrain/losses/` and `pretrain/augmentations/` survival is never decided.**
   §9 deletes `genml_kit/pretrain/methods/` and `pretrain/cli.py`, and §12 item 2
   says delete `genml_kit/pretrain/methods/` wholesale — but the plan never
   mentions `genml_kit/pretrain/{losses,augmentations}`, which are imported by
   **production** code that stays: `training/train.py:26`
   (`from genml_kit.pretrain.losses.focal import CombinedFocalLoss`),
   `models/dino.py:28`, `models/byol.py:63`,
   `pretrain/methods/dino.py:13`. After the move, `genml_kit/pretrain/` survives
   containing only `losses/` + `augmentations/` — i.e. the "pretrain" package
   still exists even though §5.2 declares the concept gone. Decide: keep as-is, or
   move to `genml_kit/losses/` + `genml_kit/augmentations/` (or under
   `methods/`), and update the four importers.
2. **`build_pretrain_transform` has no destination.** §12 item 2 deletes
   `pretrain/cli.py`, which defines it (`cli.py:60`) and which
   `methods/base.py:79-80` imports *inside* `build_transform` (so the moved
   `Method.build_transform` default would import from a deleted module). §6.1 says
   the transform "moves into the pipeline" but names no function. Same problem for
   `log_validation_images` and `build_pretrain_dataset`, both imported by tests
   from `pretrain.cli` (see #21).

---

### 21. §11.2 Test Inventory Is Incomplete 🔴

§11.2 lists 4 named test files plus "pretrain tests
(`simmim`/`supcon`/`dino`/`byol`/`ijepa`)". The real blast radius, from
`grep -rl "genml_kit\.pretrain" tests/`:

```
test_byol.py                 test_ensemble_collate.py  test_seed_utils.py
test_dino.py                 test_headless_models.py   test_signal_utils.py
test_supcon_loss.py          test_pretrain_methods.py  test_supcon_method.py
test_transformer_encoder_init.py      test_pretrain_smoke.py
                                      test_pretrain_vis.py
```

**12 files.** §11.2's catch-all row ("pretrain tests
(`simmim`/`supcon`/`dino`/`byol`/`ijepa`)") plausibly covers six of them
(`test_byol`, `test_dino`, `test_supcon_loss`, `test_supcon_method`,
`test_pretrain_methods`, `test_pretrain_smoke`). **Six are not accounted for at
all:** `test_ensemble_collate`, `test_headless_models`, `test_pretrain_vis`,
`test_seed_utils`, `test_signal_utils`, `test_transformer_encoder_init`. Two of
those omissions are not "rename the class" edits:

- `test_headless_models.py:111` imports `_PatchEmbedder` from
  `pretrain.methods.ijepa`, and `test_transformer_encoder_init.py:43` imports
  `_Predictor` from the same module. These are **private helper** imports of a
  module §9 renames — they break on the move regardless of the Method reshape.
- `test_seed_utils.py` (5 sites) and `test_signal_utils.py` import
  `pretrain.cli.parse_args`, which §12 item 2 deletes;
  `test_ensemble_collate.py:20` imports `build_pretrain_dataset` and
  `log_validation_images` from `pretrain.cli`; `test_pretrain_vis.py:11` imports
  `log_validation_images`. These functions need a documented new home (see #20.2)
  before these tests can be updated at all — this is an API decision, not a
  mechanical test edit.
- `test_supcon_loss.py:6` / `test_dino.py:9` / `test_byol.py:9` import from
  `pretrain.losses`, and `test_dino.py:8` from `pretrain.augmentations` —
  whether these change depends on the undecided item in #20.1.

§11.2 also omits `tests/test_vo_training.py`'s dependency details and never lists
`test_pipeline.py`'s neighbours; the row "tests/test_pipeline.py | keep" is
probably right but should be verified, since §6.1 changes the label-stripping path
(`FieldSectorDataset`) that file exercises.

**Fix:** regenerate the inventory from the grep above and add a column for "why it
changes" (registry rename vs deleted helper vs deleted CLI module), because three
of the four categories need source changes in non-deleted files first.

---

### 22. Minor Consistency Nits 🟢

- §2.1's `__all__` lists `"DataBlob", "DataPipeline", "build_pipeline",
  "get_pipeline", "list_pipelines", "register_pipeline"` — fine — but §9's row for
  `genml_kit/methods/__init__.py` says it exports
  `register_method/build_method/get_method/list_methods` while §2.1's example
  `from genml_kit.pipelines import images, vo_pair` has no counterpart for methods
  (§3.2 shows no `_register_builtins()` equivalent for the new
  `genml_kit/methods/`). Without it, `list_methods()` returns empty unless the
  concrete method modules happen to be imported as a side effect — and §5.1's
  `get_method("classification")` would `fatal` at startup.
- Registry attribute mismatch: §1.2/§3.2 register by `cls.NAME` (uppercase, as
  today, `registry.py:11`), but §2.3/§3.1 declare `name: str = ""` (lowercase)
  and §2.5/§2.6 use `name = "images"`. The plan must pick one; if `Method` uses
  `name`, `register_method` needs editing and §1.2's "keep API" claim breaks.
- §7's checkpoint rule says `BaseTrainer` reads/writes
  `best_{method.metric_key}`; today's `BaseTrainer` uses a class attribute
  `BEST_METRIC_KEY = "best_metric"` (`trainer.py:57`, used at `:185,188,198`),
  overridden per subclass: `"best_macro_f1"` (`train.py:1160`,
  `pretrain/cli.py:682`) and `"best_mce"` (`train_vo.py:160`). §11.2's current-state
  column says tests "check `best_macro_f1`/`best_mce_negated`", but
  `tests/test_trainer.py:262` asserts `"best_mce_negated" not in latest` — the
  negated key was already removed. §7 and §11.2 should describe today's schema
  accurately (`best_mce`, stored positive) before proposing the new one.
- §2.4's `build_pipeline` logs `"Pipeline kwargs: %s"` at `info` on every build;
  trivial, but two such logs per run (train+val loader under #12 option (a)) is
  noise — use `debug`.

---

## Summary of Required Fixes Before Implementation

Rows are in the same order as the body sections above (#1–#11 were in the
original review; #12–#22 were found while verifying it). The numbering in earlier
drafts of this table did not match the body, which made the
"fix #1–#6 before implementation" instruction ambiguous.

| # | Severity | Issue | Plan § | Fix |
|---|----------|-------|--------|-----|
| 1 | 🔴 Critical | `evaluate()` default contradicts prose (§3.1 `raise` vs §12 item 5 "optional/averages") | §3.1 | Pick one: implement the averaging default OR make it mandatory |
| 2 | 🟢 Low | `ModelOutput` declared but never constructed | §2.2 | Wrap in §4 or delete the namedtuple |
| 3 | 🔴 Critical | Reported epoch loss off by `grad_accum_steps` | §4 | Separate gradient scaling from loss reporting |
| 4 | 🔴 Critical | AMP/`autocast`/`scaler` absent from generic loop | §4 | Move forward+loss into `autocast`; full `scale/unscale/step/update` |
| 5 | 🔴 Critical | `scheduler.step()` has no home | §4 | Step in `BaseTrainer.run()` (or state subclass ownership) |
| 6 | 🔴 Critical | SSL checkpoint state hooks dropped | §3.1 | Keep `get/load_checkpoint_state` via `BaseTrainer.saver_extra` |
| 7 | 🔴 Critical | Method lifecycle hooks dropped (`on_epoch_end`, `build_transform`) | §3.1 | Restore optional hooks; route via `BaseTrainer.epoch_end()` |
| 8 | ⚠️ Design | Transform ownership blurs pipeline↔method boundary | §6.1 | Two-phase pipeline/method transform contract |
| 9 | ⚠️ Design | Supervised + SSL VO share `vo_pair`; `--vo_stage` int-vs-string conflict | §3.4/§5.2 | Distinct names or document branching; fix flag type |
| 10 | 🟡 Medium | No pipeline×method compatibility signal (deliberate per §12 item 3) | §2/§3 | Improve error text/log the resolved pair — no startup `fatal` |
| 11 | 🟡 Medium | Grad-monitor sequencing lost; clip magnitude + `global_step` convention diverge | §4 | Reinsert `unscale → monitor → clip`; pick `args.grad_clip` |
| 12 | 🔴 Critical | Pipeline loaders never wired (`self.pipeline.train_loader` undefined) | §2.3/§4/§5.1 | Decide loader ownership and show it |
| 13 | 🔴 Critical | `blob.meta` never moved to device; `loss_fn` gets original CPU blob | §2.3/§4 | `to_device(blob) -> DataBlob`, move both halves |
| 14 | 🔴 Critical | `loss_fn(out, blob)` drops `model`/`global_step` needed by BYOL/DINO/IJEPA/SimMIM | §3.3 | Keep `train_step(model, blob, global_step)` as the loop hook |
| 15 | 🟠 High | No `model.train()`/eval-mode ownership | §4 | State loop-owned or method-owned |
| 16 | 🟠 High | Two-pass parse: `add_args` class-vs-instance, kwargs never passed, flag collisions | §1.3/§5.1 | Unify signatures; add flag-ownership rule |
| 17 | 🔴 Critical | §3.4 still documents the deleted `genml-kit-pretrain` binary | §3.4 | Rewrite `:352-353` to `genml-kit-train` |
| 18 | 🟡 Medium | §10 risks table stale (`loss_fn` in `train_vo`, wrong `BaseTrainer` signature) | §10 | Refresh table; align signatures |
| 19 | 🟡 Medium | Doc-reference inventory incomplete/wrong (`vo/README.md` has none) | §12 item 7 | Re-grep at implementation time |
| 20 | 🟡 Medium | `pretrain/{losses,augmentations}` fate undecided; `build_pretrain_transform` homeless | §9/§12 | Decide package layout; name new home for CLI helpers |
| 21 | 🔴 Critical | §11.2 test inventory misses 6 files, incl. private-symbol and deleted-module imports | §11.2 | Regenerate from grep; add "why it changes" column |
| 22 | 🟢 Low | `name` vs `cls.NAME`; no `_register_builtins()` for methods; §7 key-naming prose | §2.1/§3.2 | Pick one attribute; register builtins; align §7 |

---

## Conclusion

The plan's core architecture — splitting data from objective via orthogonal pipeline/method registries — is solid. Registries mirror established patterns. CLI design is elegant.

The plan **as written cannot run at all** before any of the design questions are
settled, because the data never reaches the loop:

1. **#12** — `self.pipeline.train_loader` / `.val_loader` do not exist, and the
   loaders §5.1 builds are discarded → `AttributeError` at step 1.
2. **#13** — `blob.meta` (labels, GT similarity) is never moved to the device, and
   `loss_fn` receives the un-moved blob.
3. **#14** — `loss_fn(out, blob)` has no `model`/`global_step`, so BYOL, DINO,
   IJEPA and SimMIM cannot perform their per-step objective work (teacher EMA
   update, mask generation, crop unpacking).

After those, the correctness-critical set is: **#4** AMP/autocast, **#5**
scheduler, **#6** checkpoint state hooks, **#7** lifecycle hooks, **#3** loss
reporting math, **#17** and **#21** (deleted-binary prose and the incomplete test
inventory — both cost real debugging time during implementation), and **#1**
(the `evaluate()` contradiction).

Then the design/consistency layer: **#8** transform ownership, **#9** the
`vo_pair` name + `--vo_stage` type conflict, **#15**–**#16** model mode and CLI
plumbing, **#18**–**#20**, **#22**.

These are functional gaps in the plan's mechanics, not disagreements about
direction. Addressing them preserves the intended architecture while making it
actually work.

**Recommendation:**
- **Before implementation (blocking):** #12, #13, #14 — nothing runs without these.
- **Before implementation (correctness):** #1, #3, #4, #5, #6, #7, #17, #21.
- **During refactoring:** #8, #9, #10, #11, #15, #16, #18, #19, #20.
- **Nice-to-have / polish:** #2, #22.

Note that #10 is a **deliberate user decision** (§12 item 3) — do not "fix" it by
adding a startup `fatal` without re-confirming.

---

## Verification Log

Every claim in this review was re-checked against the working tree. Outcomes:

### Confirmed correct (evidence located)

| Claim | Evidence |
|---|---|
| #1 `evaluate()` contradiction | `plans/GENERIC_PIPELINE.md:298-305` (`raise NotImplementedError`) vs `:635-637` ("RESOLVED: optional. Default averages...") |
| #3 off-by-`grad_accum` reporting | `plans/GENERIC_PIPELINE.md:366,375`; today's compensation at `training/train.py:841` (`loss.item() * batch_size * grad_accum_steps`) |
| #4 no AMP in §4 | §4 (`:357-377`) has no `autocast`/`scaler`; `optim_factory.py:21-22` provides `.scaler` |
| #5 no `scheduler.step()` in plan | zero occurrences of `scheduler.step` in `plans/GENERIC_PIPELINE.md`; present at `train.py:1214`, `cli.py:714` |
| #6 hooks dropped | zero occurrences of `checkpoint_state` in the plan |
| #7 lifecycle hooks dropped | zero occurrences of `epoch_end` in the plan |
| #9 shared `vo_pair` name | `methods/vo_pair.py:24`, `train_vo.py:19,184`; `vo_supervised`/`vo_ssl` absent from plan |
| #11 monitor dropped from §4 | `train.py:804,811`, `cli.py:236,242` vs §4 (no `monitor`) |

### Corrected during verification

- **#2** — claimed "the loop wraps `self.model(data)` through `ModelOutput`". The
  loop does not wrap anything; the real defect is that `ModelOutput` is never
  constructed at all. Rewritten.
- **#4** — the "current supervised code" snippet was a paraphrase that misplaced
  `scaler.unscale_()` in the non-AMP branch and dropped `enabled=(...)`; replaced
  with the real `train.py:778-810` structure. The proposed fix also used the
  deprecated `torch.cuda.amp.autocast()`; the codebase (and torch 2.x) use
  `torch.amp.autocast("cuda", dtype=..., enabled=...)`.
- **#6** — table said SupCon persists "projection head params (may not be in model
  dict)" and IJEPA persists "encoder weights / prediction head state"; both
  wrong. Their `get_checkpoint_state` returns hyperparameter/architecture records
  and their `load_checkpoint_state` is `pass` (`supcon.py:65-66`,
  `ijepa.py:208-210`). Only DINO persists a tensor (`center`, `dino.py:169-170`).
  The fix also now names the existing `saver_extra` mechanism instead of inventing
  `get_checkpoint_meta`.
- **#7** — attributed BYOL/DINO momentum ramping to `on_epoch_end`; their bodies
  are literal `pass` (`byol.py:90-91`, `dino.py:159-160`). The ramp is per-step
  (`byol.py:68-72`, `dino.py:125-135`) and only IJEPA genuinely ramps per epoch
  (`ijepa.py:212-215`). Reframed, and the per-step breakage split out into #14.
- **#8** — "item 11 in Open Items" does not exist (§12 has 8 items, none about
  transforms); only §6.1 (`:479-481`) states the decision. SimMIM does not
  implement `build_transform` (it inherits the base default) and masks per step in
  `train_step`, so "masking moves into the pipeline" mis-describes §6.1. Note
  `DualViewTransform`/`MultiCropTransform` **do** exist
  (`augmentations/dual_view.py:4`, `augmentations/multicrop.py:18`) — an earlier
  draft of this verification wrongly claimed they did not; only `MaskGenerator`
  is a non-name (the code uses the free function `make_mask`,
  `methods/simmim.py:11`).
- **#9** — severity was set from "consider distinct names" alone; the `--vo_stage`
  `int` (`methods/vo_pair.py:27-29`) vs string (`plan:450,531`) conflict is a
  concrete breakage, so the item was expanded.
- **#10** — reworded: "no pairing validation" is an explicit user decision
  (`plan:630-631`), so the review must not read as an instruction to add it.
- **#11** — an intermediate draft of this verification claimed the two call sites
  pass different step indices; both pass `global_step`. Replaced with the real
  divergences (clip magnitude `args.grad_clip` vs hardcoded `1.0`).
- **Verdict / Summary table / Recommendation** — "five critical gaps" undercounted
  the review's own 🔴 labels (six), and the summary table's numbering contradicted
  the body headers (e.g. table #4 = scheduler, body #4 = AMP; table #10 =
  `ModelOutput`, body #10 = compatibility), making "Fix #1–#6" unactionable. Both
  rebuilt against body order.

### Also checked and found accurate (no change needed)

Plan §1.1's registry quote matches
`training/classifiers/__init__.py:23,67-77` (`_CLASSIFIERS`,
`register_classifier`, duplicate-name `fatal`); §1.3's
two-pass parse matches `cli.py:540-543`; `needs_labels` exists
(`methods/base.py:17`); `DataBlob`/`LossOutput`/`ModelOutput` field lists match
`plan:121-129`; "Reviewed ... No source files modified" is true (`git status`
clean for `genml_kit/`); §9's claim that `pyproject.toml` holds a
`genml-kit-pretrain` script is true (`pyproject.toml:35`).

---

*Review conducted 2026-09-14; claims verified against source and revised
2026-09-15. Evaluates INTERNAL CONSISTENCY, ARCHITECTURAL SOUNDNESS, COMPLETENESS,
and TRADEOFF QUALITY. Source code examined for verification of claims; no source
files modified.*
