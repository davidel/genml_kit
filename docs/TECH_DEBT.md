# Technical Debt

A running list of known issues, workarounds and design gaps that are understood
but not yet fixed. Each entry states the problem, the impact, how to reproduce
it, the root cause with code references, and one or more candidate fixes.

**This file is a working note. It is intentionally untracked (not added to git).**

---

## 1. Nested `StreamingDataset` wrappers double-apply the worker split (data loss)

**Severity: high — silent data loss whenever `--num_workers > 1`.**

### Symptom

With more than one DataLoader worker, an ensemble or a split built on top of a
streaming source silently drops most items. The loss grows with the worker
count, so a run looks "fast" while training on a fraction of the corpus.

Measured on the current tree (10 samples, 2 tar shards):

```
images pipeline, --num_workers 0  ->  10 images seen  (correct)
images pipeline, --num_workers 2  ->   2 images seen  (8 lost)
```

The same happens on the text path when `--val_split` is used on a stream:

```
text pipeline, --num_workers 0  ->  15 train items  (correct)
text pipeline, --num_workers 2  ->   8 train items  (7 lost)
text pipeline, --num_workers 4  ->   4 train items  (11 lost)
```

### How to reproduce

Images (ensemble) path:

```python
# Two shards of 5 samples each, built with webdataset.TarWriter.
from torch.utils.data import DataLoader
from genml_kit.datasets.field_selector import IterableFieldSelectorDataset
from genml_kit.datasets.webdataset_source import WebDatasetSource

raw = WebDatasetSource("shard-{0000..0001}.tar")          # StreamingDataset
sel = IterableFieldSelectorDataset(raw, {"png": "image"})  # StreamingDataset
loader = DataLoader(sel, batch_size=None, num_workers=2,
                    collate_fn=lambda x: x)
print(sum(1 for _ in loader))   # 2, not 10
```

Text (`--val_split`) path: build the pipeline with
`dataset_arg={"streaming": True}` and `val_split=0.25`, then iterate
`pipeline.train_dataset` through a `DataLoader(..., num_workers=2)`.

### Root cause

`StreamingDataset.__iter__` applies a **strided worker split** so that N workers
do not each emit the whole corpus:

```python
# genml_kit/datasets/iterable.py:85
def __iter__(self):
  for index, item in enumerate(self.source_iter()):
    if self.worker_slice(index):
      yield item

# genml_kit/datasets/iterable.py:68
def worker_slice(self, index):
  info = get_worker_info()
  if info is None or info.num_workers <= 1:
    return True
  return index % info.num_workers == info.id
```

`worker_slice` is applied to the index of the item **as yielded by
`source_iter`**. When a `StreamingDataset` wraps another `StreamingDataset`, both
run the split. The inner one already returned only worker `w`'s items; the outer
one then re-splits *those* by position, keeping roughly `1/num_workers` of an
already-`1/num_workers` slice — `1/num_workers²` in total.

Traced with 2 workers over 10 items (`Inner` = source, `Outer` = wrapper):

```
worker 0/2: Inner.__iter__   -> [0, 2, 4, 6, 8]     # correct 1/2 slice
worker 0/2: Outer(Inner)     -> [0, 4, 8]           # 1/2 of a 1/2 slice
worker 1/2: Inner.__iter__   -> [1, 3, 5, 7, 9]
worker 1/2: Outer(Inner)     -> [3, 7]
```

The wrappers that subclass `StreamingDataset` are:

| Class | File | Role |
| --- | --- | --- |
| `WebDatasetSource` | `genml_kit/datasets/webdataset_source.py:85` | leaf source over tar shards |
| `IterableFieldSelectorDataset` | `genml_kit/datasets/field_selector.py:77` | selects/renames fields |
| `_FilteredStream` | `genml_kit/datasets/split.py:28` | one bucket of a positional split |

So the two live chains are both two deep:

- **Images ensemble:** `IterableFieldSelectorDataset(WebDatasetSource)` — both
  split. (Also, further down, the HuggingFace streaming dataset used by the
  `hf` ensemble source shards *itself* via `num_shards`, see below.)
- **Text `--val_split`:** `_FilteredStream(HF streaming dataset)` —
  `_FilteredStream` splits, and the HF stream splits itself.

### A second, related face of the same bug: sources that self-shard

A HuggingFace streaming `IterableDataset` shards itself inside
`IterableDataset._iter_pytorch` (it calls `shard_data_sources` with
`worker_info.num_workers`). It therefore needs **no** help from
`StreamingDataset.worker_slice` — but it gets it anyway if it is wrapped by a
`StreamingDataset` subclass. This is the text `--val_split` case above: the HF
source shards, then `_FilteredStream` shards again.

Note the HF self-shard also prints `Too many dataloader workers: N (max is
dataset.num_shards=1)` and silently caps the worker count, which makes the
symptom harder to attribute.

### Why the existing tests did not catch it

- `tests/test_iterable_workers.py` runs a real `DataLoader` with real workers,
  but over a **single** `CountingStream` (one split, no nesting).
- `tests/test_iterable_split.py` tests `IterableDatasetFilter` in **process**
  (no worker processes), so only one split runs.
- No test exercised a **nested** streaming wrapper under `num_workers > 1`.

### Candidate fixes

**Option A — split ownership flag (smallest, targeted).**
Add a `worker_split: bool` constructor argument to `StreamingDataset`, defaulting
to `True`. `__iter__` applies `worker_slice` only when it is `True`. A wrapper
that wraps another `StreamingDataset` passes `worker_split=False` (it forwards
`__iter__` and lets its source split). Constructors to update:
`IterableFieldSelectorDataset`, `_FilteredStream`, and the ensemble streaming
path.

- Pro: local, no global state, easy to reason about; keeps a single split at the
  leaf.
- Con: every future wrapper must remember to pass `worker_split=False`, which is
  the same class of mistake that caused the bug.

**Option B — split only at the leaf (structural).**
Introduce a distinct base class for *transforming* streams (a "pipe" that
forwards `__iter__` unchanged and never splits), separate from *leaf sources*
that own the split. `StreamingDataset` keeps `worker_slice`; a new
`StreamingTransform` does not. `IterableFieldSelectorDataset` and
`_FilteredStream` become `StreamingTransform`.

- Pro: makes the invariant impossible to violate by construction; composes
  cleanly with the batch pipes added in `genml_kit/datasets/stream_batching.py`
  (which already assume the source owns the split).
- Con: touches more classes; needs a decision on where the split lives when a
  transform wraps a non-`StreamingDataset` source (e.g. a raw HF stream).

**Option C — detect self-sharding sources.**
When the wrapped source already splits (a `StreamingDataset`, or a HuggingFace
`IterableDataset`), skip the outer split automatically.

- Pro: no caller discipline required; fixes the HF case without special-casing.
- Con: detection is heuristic (`isinstance(source, StreamingDataset)` plus a
  HuggingFace-specific check) and couples `StreamingDataset` to HF internals.

**Recommendation.** Option B is the durable fix (the invariant becomes
structural, matching the "pipe" model already used for batch batching). Option A
is an acceptable stopgap if the change must stay minimal. Whichever is chosen, a
regression test must run a **nested** chain under `num_workers >= 2` and assert
the exact item multiset — that is the gap that let this through.

### Interim mitigation

Until fixed, run streaming training with `--num_workers 0`, or use a single
streaming wrapper depth (e.g. a `webdataset:` spec without a further field
selector). Both are workarounds, not fixes.

---

## 2. Image classification (`--pipeline images --dataset`) cannot stream

**Severity: medium — a missing capability, not a correctness bug. It fails
loudly rather than silently.**

### Symptom

`--dataset_arg streaming=True` has no effect on the image **classification**
path: `load_and_split_dataset` does not accept `dataset_arg`, so the flag is
dropped and the dataset is materialised (or the load fails if the dataset is too
large to materialise).

### Root cause

`load_and_split_dataset` (`genml_kit/training/train.py:33`) forwards nothing to
`datasets.load_dataset` beyond `cache_dir`, and it splits with
`train_test_split` / `.select`, which a stream does not support. The text
pipeline's counterpart (`genml_kit/datasets/text_proxy.py`) was given
`dataset_arg` support, which is why only text streams today.

Streaming here is not merely "plumb the kwarg": the classification path needs
whole-corpus facts that a one-pass stream cannot provide without scanning it:

- the **label vocabulary** and `num_labels`
  (`HFDatasetProxy._detect_labels`, `genml_kit/datasets/hf_proxy.py`),
- the **class weights** (`compute_class_weights`, `genml_kit/training/labels.py:24`),
- the **balanced sampler** label array
  (`train_proxy.dataset[train_proxy.label_column]`,
  `genml_kit/pipelines/images.py:290`),
- the train/val split (`train_test_split` / `.select`).

Each of these reads a column across all rows.

### Candidate fixes

**Option A — refuse clearly.** If `--dataset` is used with `streaming=True` on
the images pipeline, fail fast with a message pointing at `--datasets` (the
ensemble path, which does stream). Cheapest, and makes the limitation explicit
rather than silent.

**Option B — support streaming classification with declared metadata.** Accept
`streaming=True` plus explicit `--num_labels` / a label-vocabulary argument, and
route the train/val split through `IterableDatasetFilter` (already used by the
text path). Disable the label-scanning features (`--sampler balanced`, class
weights) with a clear error when a stream is detected.

- Pro: unlocks large image-classification corpora.
- Con: shifts responsibility to the user to declare what used to be inferred.

**Option C — bounded-probe vocabulary.** Estimate the label vocabulary and class
weights from a bounded prefix of the stream, warn that they are approximate.

- Pro: no new required flags.
- Con: an incomplete vocabulary would mislabel unseen classes; a wrong class
  vocabulary is a silent correctness bug, which is worse than the current loud
  failure.

**Recommendation.** Option A now (explicit, cheap, prevents confusion), with
Option B as the real feature if streaming image classification is wanted. Avoid
Option C unless a partial vocabulary is explicitly acceptable.

### Related

The README's streaming section documents this limitation. If Option B is
implemented, update that section and the `--dataset_arg` help text.

---

## 3. (Resolved) The `genml-cli-test` README harness and new markdown files

Not debt, recorded for context: the README CLI harness globs `README.md` and
`*/README.md` only (`tests/test_readme_cli.py:109`), so a file such as
`docs/TECH_DEBT.md` is not scanned for runnable command fences. Any `bash` fence
containing a `genml-kit-*` command in a scanned README must be marked
(`<!-- genml-cli-test ... -->`) or excluded
(`<!-- genml-cli-test-exclude: reason -->`).
