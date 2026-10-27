# genml_kit — Local LLM Support Plan

This document is a **planning artifact**: it describes how to extend `genml_kit`
to cover decoder-only large language models (tokenization, causal-LM training,
generation, evaluation, and preference-based alignment), and *how each piece
slots into the existing v4.2 architecture* without new control flow in the
driver. It is a companion to [`NEXT_STEPS.md`](NEXT_STEPS.md), which lists
"Vision-language / multimodal" (#8) and "Alignment / post-training" (#9) as the
two Tier-3 items that this plan starts to unblock.

> Status: proposal. Nothing here is implemented yet. Scope of this first cut is
> **the full path through Phase 4** (text data path, causal-LM training, LoRA,
> generation/eval, and DPO/reward-model alignment). Phases are ordered so each
> ships a working, reviewable increment.

---

## 1. Does it make sense? — Yes

The v4.2 refactor turned the harness into a *branchless* driver over exactly
two abstractions:

- **`Method`** (`genml_kit/methods/base.py`) — the objective side. It owns
  `build_model`, `train_step(model, blob, step) -> LossOutput(loss, metrics)`,
  `evaluate`, `prepare_transforms`, `wire_data`, `post_train`,
  `get_checkpoint_state`, `build_optimization`, and `add_args`.
- **`DataPipeline`** (`genml_kit/pipelines/base.py`) — the data side. It owns
  `build_loader(args, mode, method=...)`, `to_device(blob, device)`,
  `get_objective_transform`, `init_env`, and `add_args`.

`BaseTrainer.run()` (`genml_kit/training/trainer.py`) only knows three things:
the `DataBlob` / `LossOutput` contracts (`pipelines/contracts.py`) and the
`optimizer.zero_grad / backward / step` protocol. It never inspects the method
or pipeline name.

A causal LM is therefore *"another pipeline (`text`) plus another method
(`causal_lm`)"*. The loop, AMP/autocast, gradient accumulation, LLRD, `--lr_group`,
`--freeze` patterns, `--grad_checkpoint`, `--source_checkpoint`/`--resume`,
signal handling, atomic checkpointing, and the already-generic `--lora` support
(`training/model_utils.py`, which wraps any model with `peft`) all carry over
**unchanged**.

### What the current design does *not* have

These are the genuine gaps, and they define the work:

1. **The "processor" abstraction is image-only.** `load_processor`
   (`models/registry.py`) returns an object exposing `image_mean` / `image_std`
   and wraps `AutoImageProcessor`. There is no tokenizer concept anywhere.
2. **The data path is image-centric.** `pipelines/images.py` coerces items to
   `{"image", "label"}`; `datasets/hf_proxy.py` even lists `text` / `caption`
   in `IGNORE_LABEL_NAMES`. There is no text collator, no label masking, no
   sequence packing.
3. **Eval is classification-centric.** `training/eval.py` computes top-1 /
   macro-F1 / confusion matrices. There is no perplexity and no decoding.
4. **No generation surface.** `genml-kit-infer` (`training/infer.py`) is
   image-only: no sampler (temperature / top-k / top-p / repetition penalty),
   no chat template, no batched generation.

So: extension is natural for **training**, and requires genuinely new surface
for **tokenization, collation, generation, and LLM eval**.

---

## 2. Do small LLM experiments matter without huge compute? — Yes

Small-scale LLM research is standard methodology, not a consolation prize:

- **Reproducible small suites.** The Pythia suite (EleutherAI) and OLMo were
  built as small, reproducible models (70M–1B) explicitly so phenomena can be
  studied on a single GPU.
- **Mechanistic findings were made on tiny models.** Induction heads (2-layer
  attention-only), grokking, in-context-learning emergence, and task-vector
  arithmetic are all reproducible at 1–100M parameters.
- **Scaling laws are fit on small sweeps.** Chinchilla/Kaplan curves and the
  ~20-tokens-per-parameter heuristic are derived from many small runs; a laptop
  can generate a meaningful loss-vs-params/-tokens sweep.
- **Alignment and efficiency research thrives small.** LoRA vs full fine-tuning,
  **DPO vs PPO** (the DPO paper uses GPT-2-XL-class models), QLoRA on a single
  consumer GPU, data-mixture ablations, reward-model overfitting, and
  quantization / KV-cache / speculative-decoding studies are all routinely done
  at or below 1B parameters.

**Where scale genuinely matters (honest non-goals).** Emergent capabilities that
only appear above ~10B, instruction-following *quality*, long-context behavior,
and multimodal VLMs. This plan targets the *mechanism, dynamics, alignment-
algorithm, and efficiency* axes, not frontier capability.

**Experiment menu this unlocks (no big compute):** perplexity / Chinchilla
sweeps on TinyStories; induction-head probing; LoRA rank/alpha ablations; DPO vs
SFT on a preference subset; data-ordering / curriculum effects; packing vs
padding throughput; quantized-vs-fp16 eval deltas.

---

## 3. Design overview

The guiding rule (from `NEXT_STEPS.md` §"Cross-cutting concerns") is that every
new capability registers through the existing registries so the CLI picks it up
with **no new branches in `train.py`**. Concretely, the new components are:

| Registry | New entry | File |
|---|---|---|
| `PIPELINES` | `text` | `genml_kit/pipelines/text.py` |
| `PIPELINES` | `preference` | `genml_kit/pipelines/preference.py` (Phase 4) |
| `METHODS` | `causal_lm` | `genml_kit/methods/causal_lm.py` |
| `METHODS` | `dpo` | `genml_kit/methods/dpo.py` (Phase 4) |
| `METHODS` | `reward_model` | `genml_kit/methods/reward_model.py` (Phase 4) |
| `TOKENIZERS` (new) | custom tokenizers | `genml_kit/models/tokenizer.py` |
| `LOSSES` | `dpo`, `bradley_terry` | `genml_kit/losses/alignment.py` (Phase 4) |

New non-registry modules:

- `genml_kit/datasets/text_proxy.py` — HF text dataset → dict items.
- `genml_kit/datasets/preference_proxy.py` — chosen/rejected pairs (Phase 4).
- `genml_kit/training/text_collator.py` — tokenize / pad / mask / pack.
- `genml_kit/utils/generation.py` — samplers + batched `generate`.
- `genml_kit/training/generate.py` — `genml-kit-generate` entry point.
- `genml_kit/training/llm_eval.py` — perplexity, exact-match, token-F1, pass@k.

### 3.1 Tokenizer abstraction

Mirror the model/processor split: `MODELS` / `PROCESSORS` already dispatch HF vs
custom via the `parse_model_name` `name:backbone` protocol. Add a parallel
`TOKENIZERS` registry and a `load_tokenizer(name, ...)` entry point:

- `load_tokenizer` returns an object exposing `tokenizer_path`, `vocab_size`,
  `pad_token_id`, `eos_token_id`, `apply_chat_template`, and `__call__`
  (so it is a drop-in for the collator).
- **Default backend:** `transformers.AutoTokenizer.from_pretrained`. This is the
  path every normal model ID takes (`sshleifer/tiny-gpt2`, `gpt2`,
  `meta-llama/*`, `mistralai/*`, `EleutherAI/pythia-*`, `allenai/OLMo-*`).
  Crucially, as of `transformers >= 4.40`, `AutoTokenizer` *already*
  auto-detects and converts **both** SentencePiece `.model` files and
  tiktoken-format `.model` files into the Rust `tokenizers` backend, so no
  per-family registration is needed for Hub models.
- **`TOKENIZERS` is an extension point, not a populated catalogue** — exactly
  like `PROCESSORS`, which only registers custom wrappers (`"timm"`,
  `"convvit"`, `"uvito"`, ...) and falls back to `AutoImageProcessor` otherwise.
  We register an entry only when `AutoTokenizer` is insufficient:
  - `byte` — a dependency-free byte/char tokenizer used by **offline unit
    tests** so the collator and pipeline tests need no network download.
  - `spm` — wraps `sentencepiece.SentencePieceProcessor` to load a **bare
    `tokenizer.model`** (no `tokenizer_config.json` / `special_tokens_map.json`),
    which `AutoTokenizer` cannot resolve on its own. Requires the `sentencepiece`
    dependency (see §5).
- **tiktoken** is deliberately *not* registered in Phase 1 (it would add a
  second tokenizer dependency for no Phase-1 use). It stays a documented
  on-demand entry: a thin wrapper over
  `transformers.integrations.tiktoken.convert_tiktoken_to_fast` for vocabularies
  not on the Hub.
- `load_processor` is left **untouched** for vision — text and image paths stay
  separate, matching the existing "pipeline owns its data side" philosophy.

### 3.2 Text pipeline (`PIPELINES` name `text`)

`pipelines/text.py` mirrors `pipelines/images.py` but produces token batches:

- `TextDatasetProxy` (`datasets/text_proxy.py`) reads an HF dataset by
  `--text_column` (plain LM corpus) or `--instruction_column` +
  `--response_column` (instruction tuning).
- The **collator** (`training/text_collator.py`) tokenizes, pads to
  `--max_seq_len`, builds `labels` with `-100` masking for prompt tokens when
  `--mask_prompt` is set, and optionally packs sequences when `--packing` is set.
- `DataBlob(data={"input_ids", "attention_mask"}, meta={"labels": ...})`;
  `to_device` moves the whole dict, so the generic loop and the trainer's
  `blob = pipeline.to_device(blob, device)` line are untouched.
- `add_args`: `--max_seq_len`, `--text_column`, `--instruction_column`,
  `--response_column`, `--packing`, `--mask_prompt`, `--pad_to_multiple_of`.
- Reuses `parse_dataset_specs` (`datasets/factory.py`) for `hf` vs local paths.

### 3.3 Causal-LM method (`METHODS` name `causal_lm`)

`methods/causal_lm.py`:

- `build_model` → `AutoModelForCausalLM.from_pretrained(...)` and then the same
  `_apply_model_extras` path every other method uses, so `--lora`,
  `--source_checkpoint`, `--freeze`, `--grad_checkpoint` all work for LMs.
- `train_step` → forward, **shifted** cross-entropy over `labels` (ignore
  index `-100`), returning `LossOutput(loss, {"loss": ..., "ppl": ...})`.
- `METRIC_KEY = "loss"`, `METRIC_MINIMIZE = True`, `NEEDS_LABELS = True`.
- `prepare_transforms` resolves the tokenizer (lifecycle position 1) and keeps
  it on the method object, exactly as `ClassificationMethod` keeps its
  transforms.
- `evaluate` computes mean loss / perplexity.

### 3.4 Generation & evaluation

- `utils/generation.py`: a small `GenerationConfig`-like wrapper with
  temperature / top-k / top-p / repetition penalty / max-new-tokens, batch
  `generate`, and chat-template application.
- `training/generate.py` + console script `genml-kit-generate`
  (`pyproject.toml [project.scripts]`): prompts (file or HF dataset) in,
  completions out (JSONL).
- `training/llm_eval.py`: perplexity, exact-match, token-F1, optional pass@k,
  structured so a future `lm-eval-harness` adapter can slot in.

### 3.5 Alignment (Phase 4)

Follows the same pattern as the RL methods (which already satisfy `Method`):

- `pipelines/preference.py` (`preference`): chosen/rejected pairs via
  `datasets/preference_proxy.py`.
- `methods/dpo.py` + `losses/alignment.py::dpo_loss`: sigmoid DPO objective over
  policy vs reference log-probs.
- `methods/reward_model.py` + `losses/alignment.py::bradley_terry_loss`: a
  scalar head (registered in `DECODERS`/`CLASSIFIERS`) with a Bradley–Terry
  ranking loss.

---

## 4. Phased rollout

Each phase is a self-contained, reviewable increment.

### Phase 1 — Foundation
**Goal:** `genml-kit-train --pipeline text --method causal_lm --model sshleifer/tiny-gpt2 --dataset wikitext` runs end-to-end.

Deliverables:
- `models/tokenizer.py` (`TOKENIZERS`, `load_tokenizer`).
- `datasets/text_proxy.py`, `training/text_collator.py`.
- `pipelines/text.py` (`NAME = "text"`).
- `methods/causal_lm.py` (`NAME = "causal_lm"`).
- Register the new modules in `models/__init__.py`, `pipelines/__init__.py`,
  `methods/__init__.py`.

### Phase 2 — Efficient sequence fine-tuning
**Goal:** `--lora`, `--grad_checkpoint`, `--amp_dtype bf16` verified on LMs.

Deliverables:
- Default `--lora_target_modules` for attention projections
  (`q_proj,k_proj,v_proj,o_proj`) and a documented LM LoRA recipe.
- Audit `io/checkpointing.py` LoRA save/load against a `PeftModel`-wrapped LM.
- Confirm `--source_checkpoint` / `--resume` round-trip for causal LMs.

### Phase 3 — Generation & evaluation
**Goal:** `genml-kit-generate` produces completions; `llm_eval.py` reports
perplexity and quality metrics.

Deliverables:
- `utils/generation.py`, `training/generate.py`, `training/llm_eval.py`.
- `pyproject.toml`: add `genml-kit-generate` to `[project.scripts]`.

### Phase 4 — Alignment
**Goal:** SFT (`--method causal_lm --mask_prompt`) → DPO → reward model.

Deliverables:
- `pipelines/preference.py`, `datasets/preference_proxy.py`.
- `methods/dpo.py`, `methods/reward_model.py`, `losses/alignment.py`.
- Register in the respective registries.

### Phase 5 — Tests, docs, packaging (runs alongside each phase)
- Tests (offline-friendly, tiny random models): `test_pipelines_text.py`,
  `test_causal_lm.py`, `test_text_collator.py`, `test_dpo.py`. Follow
  `pytest.ini` conventions (`-m 'not slow'`, warnings-as-errors).
- `README.md` LLM section; `NEXT_STEPS.md` status update for #8/#9.

---

## 5. Dependencies

The common path needs **no new dependency**: `transformers` (already a hard
dependency) provides `AutoTokenizer` and `AutoModelForCausalLM`, and
`AutoTokenizer` handles both SPM and tiktoken Hub models. One tokenizer
dependency is authorised for the bare-`.model` case, plus optional extras:

| Extra | Packages | Needed for |
|---|---|---|
| `llm` | `peft`, `accelerate` | LoRA + device-map training (Phase 2+) |
| `llm-spm` | `sentencepiece` | the `spm` tokenizer entry (bare `tokenizer.model`) |
| `llm-quant` | `bitsandbytes` | 4/8-bit QLoRA experiments (optional) |

`peft` already exists as the `lora` extra, so `llm` will reuse it. `sentencepiece`
is added behind the `llm-spm` extra and imported lazily inside the `spm` entry,
so it never becomes a hard requirement. All extras are opt-in; no new *hard*
dependencies are introduced.

---

## 6. Testing strategy

- All new tests run on **tiny** models (`sshleifer/tiny-gpt2`,
  `hf-internal-testing/tiny-random-gpt2`) and small synthetic corpora so they
  stay offline-friendly and fast under `-m 'not slow'`.
- Structural tests assert the registry contracts: `PIPELINES.contains("text")`,
  `METHODS.contains("causal_lm")`, metric direction consistency (mirroring
  `test_pipeline_method_registry.py::TestMetricDirection`).
- One `slow`-marked convergence test: overfit a 20-example corpus to near-zero
  loss (mirrors existing learning tests).

---

## 7. Risks & non-goals

- **Padding vs packing correctness.** Packed batches need correct attention
  masking if `--packing` is used with document boundaries; start with
  padding-only, add packing behind a flag.
- **Label shifting.** Off-by-one in the shifted cross-entropy is the classic
  bug; covered by a dedicated unit test with a hand-computed target.
- **Warnings-as-errors.** `pytest` turns warnings into errors, and `transformers`
  is chatty; any noise must be handled by fixing causes (not silencing).
- **Non-goals:** frontier-scale training, multimodal VLM heads (that is item #8,
  a separate milestone), and instruction-quality claims.

---

## 8. Task checklist

- [ ] `models/tokenizer.py` — `TOKENIZERS` + `load_tokenizer`, with the `byte`
      (test-only) and `spm` (bare `.model`) entries (+ tests)
- [ ] `datasets/text_proxy.py` — HF text → dict items
- [ ] `training/text_collator.py` — tokenize/pad/mask/pack (+ tests)
- [ ] `pipelines/text.py` — `PIPELINES` name `text` (+ tests)
- [ ] `methods/causal_lm.py` — `METHODS` name `causal_lm` (+ tests)
- [ ] Register new modules in the three `__init__.py` files
- [ ] Phase 2: LM LoRA blend + checkpoint audit
- [ ] Phase 3: `utils/generation.py`, `training/generate.py`,
      `training/llm_eval.py`, `genml-kit-generate` script
- [ ] Phase 4: `preference` pipeline, `dpo` + `reward_model` methods,
      `losses/alignment.py`
- [ ] Docs: `README.md` section + `NEXT_STEPS.md` status update
