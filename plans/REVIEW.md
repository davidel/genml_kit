# Generic Pipeline + Method Registry — Design Review (v4)

## Review Status
Reviewed against `plans/GENERIC_PIPELINE.md` (v4). Source code examined for architectural context: `training/train.py`, `training/trainer.py`, `training/vo/train_vo.py`, `pretrain/cli.py`, `pretrain/methods/base.py` (and concretions), `training/classifiers/__init__.py`, `training/optim_factory.py`. No source files modified.

---

## Verdict

The plan proposes a sound core architecture. The pipeline↔method split cleanly decomposes training into orthogonal concerns. The registries faithfully mirror existing patterns. Two-pass CLI parse elegantly solves dynamic flag registration.

However, five **critical functional gaps** undermine completeness: three affect mixed-precision training and SSL checkpoint/resume semantics, two involve arithmetic bugs in the generic loop. Plus one concern about where the transform boundary sits — if poorly drawn, the pipeline↔method abstraction becomes blurry.

All these fixes are tractable; none require rethinking the overall architecture.

---

## Internal Consistency

### 1. `evaluate()` Default Contradiction 🔴

**§3.1 shows:**
```python
def evaluate(self, model, loader, device):
    raise NotImplementedError
```

**§12 Q5 resolution says:**
> resolved: optional. Default averages `loss_fn` metrics over the loader; only VO (mce) and classification (macro-F1/confusion) override.

Code says "raise." Text says "default exists." Contradiction.

Either implement the averaging default, or keep `NotImplementedError` as the contract (meaning all methods must provide custom validation logic). Both approaches work architecturally — pick one and make the code match the text (or vice versa).

---

### 2. `ModelOutput` Indirection 🟢 Minor

```python
ModelOutput = collections.namedtuple("ModelOutput", ["predictions"])
```

Loop wraps `self.model(data)` result through this before passing to `loss_fn`. Adds no semantic value — direct pass of `out = self.model(data)` would work identically. If the team anticipates needing to attach metadata to model outputs later, fine; otherwise it's gratuitous indirection. Low priority.

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

Current supervised code (`training/train.py`) handles AMP explicitly:

```python
with autocast(amp_dtype):
    outputs = model(images)
    loss = criterion(outputs, labels)
if amp_dtype:
    scaler.scale(loss).backward()
else:
    scaler.unscale_(optimizer)
    monitor.step(...)  # reads TRUE gradients BEFORE clipping
    clip_grad_norm_(...)
    scaler.step(optimizer)
    scaler.update()
```

Plan's §4 loop has none of this:

```python
loss.backward()               # ← no autocast context!
clip_grad_norm_(...)          # ← no unscale before clip!
optimizer.step()              # ← no scaler involvement!
```

Two concrete breakages:

1. **AMP mode crashes.** `loss.backward()` without `autocast` means float32 computation throughout — defeats the purpose of enabling AMP, or may crash if gradients are unexpectedly large without scaler protection.

2. **Gradient monitoring reads wrong values.** Current code carefully sequences `unscale → monitor.step → clip_grad_norm`. Monitor needs true gradient magnitudes (post-unscale, pre-clip). The §4 loop has no monitoring at all. Combined with no `unscale_` call, any gradient monitoring code would read clipped (incorrect) gradients.

The plan's `BaseTrainer.__init__` receives an `optimization` named tuple (`from optim_factory.py`) with fields `(param_groups, optimizer, scheduler, scaler)`. The `.scaler` field exists but is completely unused in §4.

**Fix:** Add the full AMP/scaler sequence to §4:
```python
loss = loss_out.loss / self.args.grad_accum_steps
with torch.cuda.amp.autocast():                           # if using AMP
    scaled = self.optimization.scaler.scale(loss)
    scaled.backward()
else:
    loss.backward()
    
# At step time:
if self.optimization.scaler is not None:
    self.optimization.scaler.unscale_(self.optimization.optimizer)
# monitor.step(...) -- reads true gradients after unscale, before clip
if self.args.grad_clip > 0:
    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.args.grad_clip)
if self.optimization.scaler is not None:
    self.optimization.scaler.step(self.optimization.optimizer)
    self.optimization.scaler.update()
else:
    self.optimization.optimizer.step()
```

---

### 5. Scheduler Placement Missing 🔴

Current code calls `scheduler.step()` after `train_one_epoch` returns and before the next epoch begins. The plan's `BaseTrainer.__init__` accepts `optimization` (containing `.scheduler`) but §4's `train_epoch` body never invokes it. `BaseTrainer.run()` is not fully shown — we can see `__init__` signature but not the loop body that would call `scheduler.step()` between epochs.

If the scheduler is accepted but never used, learning rate annealing is silently broken. This is a clear omission.

**Fix:** Show `scheduler.step()` call location in `BaseTrainer.run()`, e.g., at the end of each epoch iteration.

---

### 6. Checkpoint State Hooks Removed From Method 🔴

`PretrainMethod` has `get_checkpoint_state(model, args)` / `load_checkpoint_state(model, state, args)`. These persist SSL-critical internal state:

| Method | State preserved | Purpose |
|--------|----------------|---------|
| BYOL | `_byol_momentum`, `_byol_final_momentum` | Teacher momentum decay over training |
| DINO | momentum factors + moving-average `center` vectors | Self-distillation convergence |
| SupCon | Projection head params (may not be in model dict) | Contrastive encoder config |
| IJEPA | Encoder weights, prediction head state | Joint-embedding model internals |

Reshaped `Method` (§3.1) has NO checkpoint hooks. Yet `BaseTrainer.__init__` clearly manages checkpoint persistence (saves/restores model, optimizer, etc.). There's nowhere for method-level state to live.

**Impact:** SSL checkpoints save model/optimizer/scheduler but NOT momentum factors or centers. On resume, trained models reset their momentum ramps — losing weeks of incremental training progress. The trained state effectively restarts.

**Fix:** Add optional `get_checkpoint_meta(args)` / `load_checkpoint_meta(state, args)` to `Method`. Merge returned dicts into checkpoint alongside model/optimizer/scheduler state, matching the existing extension mechanism.

---

### 7. Epoch-End Hook Removed 🔴

`PretrainMethod.on_epoch_end(model, epoch, writer)` drives BYOL's teacher momentum ramping (step up toward final over epochs) and potentially DINO's center schedule. Reshaped `Method` ABC drops this entirely.

Without a per-epoch callback invoked by `BaseTrainer`, BYOL loses its core time-dependent learning mechanism. It still trains, but with constant momentum instead of ramping — likely inferior results.

**Fix:** Restore `on_epoch_end(epoch)` as a non-abstract optional hook on `Method`. Have `BaseTrainer.run()` call `self.method.on_epoch_end(epoch)` after `train_epoch` completes, before `validate`.

---

### 8. Transform Ownership Blurs Pipeline↔Method Boundary ⚠️

§6.1 states: *"the pipeline owns the transform... Method.build_transform moves into the pipeline's data-transform responsibility."*

But some transforms are objectively objective-defining:

| Transform | Classification | BYOL | DINO | SimMIM |
|-----------|---------------|------|------|--------|
| Random crop+flip | Preprocessing ✅ | Preprocessing ✅ | Preprocessing ✅ | Preprocessing ✅ |
| Normalization | Preprocessing ✅ | Preprocessing ✅ | Preprocessing ✅ | Preprocessing ✅ |
| Dual-view augmentation | N/A | Objective-defining ❌ | Objective-defining ❌ | N/A |
| Multi-crop strategy | N/A | N/A | Objective-defining ❌ | N/A |
| Mask generation | N/A | N/A | N/A | Objective-defining ❌ |

Moving masking, dual-view composition, and multi-crop into `ImagesPipeline` means the pipeline generates fundamentally different data structures depending on which method consumes it:
- Regular images `(B,C,H,W)` for classification
- `(image_a, image_b)` tuples for BYOL/DINO
- Masked image stacks for SimMIM
- Varies by method type even within `ImagesPipeline`

This undermines the whole reason for separating pipeline from method. The abstraction boundary should enforce that pipelines produce consistent data shapes regardless of method, while methods handle how that data is consumed.

**Better approach:** Keep objective-specific augmentation (masks, views, crops) in the method layer. Only move generic preprocessing (resize, normalize, basic aug like color jitter/flip) into pipelines. Alternatively, introduce a two-phase transform system:
1. **Pipeline transform**: generic preprocessing (uniform across methods)
2. **Method transform**: objective-specific augmentation (method-defined)

---

### 9. Supervised vs Self-Supervised VO Share `vo_pair` Name ⚠️

Both variants register under name `vo_pair`, distinguished by `--vo_stage`. The `VOPairMethod` class internally branches on stage to switch objectives. This works but makes the name→objective mapping indirect. Consider distinct names (`vo_supervised` / `vo_ssl`) for clarity, or document the branching clearly.

---

## Completeness Gaps

### 10. No Pipeline↔Method Compatibility Validation 🟡

Resolved in §12-Q3 as "none. Any combination allowed." But several combinations produce deterministic runtime failures:

| Pairing | Failure Mode |
|---------|-------------|
| `images` × `vo_pair` | `VOPairMethod.loss_fn` expects `blob.data` as `(image_a, image_b)` tuple; gets `(B,C,H,W)` tensor |
| `vo_pair` × `simmim` | SimMIM forward passes single images through masked encoder; receives paired geometry batches |
| `vo_pair` × `dino` | DINO expects individual images; receives structured pairs |

Users waste cycles debugging tensor-shape errors mid-training instead of getting immediate startup failures. Minimal validation (e.g., list valid pipeline×method pairs or check data shape compatibility) would prevent obvious mismatches.

---

### 11. Monitoring Integration Lost 🟡

Current code carefully sequences: `unscale → monitor.step → clip_grad_norm`. Monitor needs true gradient magnitudes (post-unscale, pre-clip). The §4 generic loop omits monitoring entirely. Combined with missing scaler (#4 above), gradient monitoring is broken in practice.

Reinsert the `unscale → monitor → clip` sequence alongside the scaler fix.

---

## Summary of Required Fixes Before Implementation

| # | Severity | Issue | Fix |
|---|----------|-------|-----|
| 1 | 🔴 Critical | `evaluate()` default contradicts code (§3.1 vs §12-Q5) | Pick one: implement default OR make mandatory |
| 2 | 🔴 Critical | Mixed precision/scaler absent from loop (§4) | Add `autocast`, `scaler.scale/unscale/step/update` |
| 3 | 🔴 Critical | Loss accumulation math off-by-N (§4) | Separate gradient scaling from loss reporting |
| 4 | 🔴 Critical | Scheduler placement undefined | Show `scheduler.step()` in `BaseTrainer.run()` |
| 5 | 🔴 Critical | SSL state lost on checkpoint (no hooks) | Add `get_checkpoint_meta/load_checkpoint_meta` |
| 6 | 🔴 Critical | Per-epoch callback removed (BYOL/DINO break) | Restore `on_epoch_end` optional hook |
| 7 | 🟡 High | Monitor timing lost (reads wrong grads) | Reinsert `unscale → monitor → clip` sequence |
| 8 | 🟡 High | Transform ownership blurs boundary (§6.1) | Split preprocessing↔objective-augmentation |
| 9 | 🟡 Medium | No pipeline×method compatibility checks | Add basic mismatch detection at startup |
| 10 | 🟢 Low | `ModelOutput` indirection unnecessary (§2.2) | Remove wrapper unless future-proofing needed |
| 11 | 🟢 Low | Supervised/SSL VO naming confusion (§3.3) | Consider distinct names for clarity |

---

## Conclusion

The plan's core architecture — splitting data from objective via orthogonal pipeline/method registries — is solid. Registries mirror established patterns. CLI design is elegant.

Five critical gaps need fixing before implementation:
1. **Mixed precision/scaler** absent from the generic loop — breaks AMP mode
2. **Loss accumulation** arithmetic is incorrect — reports wrong values
3. **Scheduler** placement undefined — LR annealing silently broken
4. **Checkpoint hooks** removed — SSL methods lose internal state on resume
5. **Per-epoch callback** removed — BYOL/DINO momentum ramping breaks

Plus two medium-priority items: monitoring timing and transform boundary clarity.

These are functional gaps in the plan's mechanics, not disagreements about direction. Addressing them preserves the intended architecture while making it actually work.

**Recommendation:** Fix #1–#6 before implementation. Resolve #7–#8 during refactoring. #9–#11 are nice-to-haves.

---

*Review conducted 2026-09-14. Evaluates INTERNAL CONSISTENCY, ARCHITECTURAL SOUNDNESS, COMPLETENESS, and TRADEOFF QUALITY. Source code examined for verification of claims.*
