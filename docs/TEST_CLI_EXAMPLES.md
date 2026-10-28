# Testing the README CLI Examples

Status: **PLAN — not yet implemented.**
Owner: TBD.
Related: `scripts/md_latex.py` / `tests/test_md_latex.py` (precedent for
"logic in `scripts/`, thin test imports it"), `pyproject.toml`
(`[tool.pytest.ini_options]`).

Contents:

- 1. Motivation
- 2. Scope
- 3. Design
- 4. New / changed files
- 5. Per-block decisions
- 6. Validation workflow
- 7. CI integration
- 8. Implementation reference
- 9. Fixture catalogue
- 10. README edits (exact specifications)
- 11. Acceptance matrix
- 12. Open items / risks
- Appendix A — Verbatim in-scope blocks
- Appendix B — Marker quick reference

---

## 1. Motivation

Every README in this repository hands the reader copy-pasteable command
lines as "quick start" recipes.  These commands are almost never executed by
CI, so they silently drift as the CLI, the model registry, and the dataset
pipeline evolve.  A command that worked six months ago may today reference a
renamed flag, a removed dataset spec, or a shell typo (`genml_kit-train`
instead of `genml-kit-train` — this bug exists in the tree *right now*, see
s 3.6).

We want a test suite that:

1. **Parses** the READMEs,
2. **Selects only** the command blocks a maintainer has explicitly opted in
   via an invisible marker, and
3. **Executes** each such block as a bounded smoke test that writes every
   artifact into a temporary directory which is removed afterwards.

Published command lines (the text the human reads) keep their "real" values
(e.g. `--epochs 5`, a full recipe).  Test-only limiting arguments are supplied
*through the marker*, never by mutating the rendered README command.

---

## 2. Scope

### 2.1 In scope

* Fenced code blocks whose info string is exactly `bash` or `sh` (case
  insensitive).  Unlabeled fences (` ``` `) are **not** candidates — the RL
  examples that currently use unlabeled fences will be relabeled ` ```bash `
  as part of this work.
* A candidate block is *tested* only if its command form begins with one of
  the **core package entry points**:
  * `genml-kit-train`
  * `genml-kit-infer`
  * `genml-kit-llm`
  (plus the underscore typo variant, see s 3.6.)
* Any README under the repo root: `README.md`, `rl/README.md`,
  `vo/README.md`, `knowledge_distillation/README.md`, `scdiag/README.md`.

### 2.2 Out of scope (never executed)

* `pip install …`, `git clone …`, `pytest …`, `cd …` lines.
* IPython magics (`%env AWS_ACCESS_KEY_ID=…`).
* Any `python scripts/…` or `python scdiag/scripts/…` invocation.  **Only
  the core package CLI is under test.**  Helper/preparation scripts are not.
* Non-`bash`/`sh` fences (JSON output, directory trees, Python snippets,
  config fragments).

A candidate fence that contains **no** core-entry-point command is skipped by
the parser entirely (no marker needed).

---

## 3. Design

### 3.1 The marker

HTML comments are not rendered by GitHub Markdown, so a comment placed
immediately above a fence is invisible to the reader but trivially parseable.
The marker body is **YAML**.

````markdown
<!-- genml-cli-test
name: quick start cifar10
network: true
timeout: 600
extra: --epochs 1 --max_items 16
override:
  - --dataset {tmp}/img
  - -.--source_checkpoint
extras: lora
env: {HF_HOME: "{tmp}/hf"}
-->
```bash
genml-kit-train --model google/vit-base-patch16-224 \
                --dataset cifar10 \
                --label_column label \
                --epochs 5 \
                --batch_size 32 \
                --lr 3e-5 \
                --image_size 224
```
````

#### Marker keys

| Key | Type | Default | Meaning |
|---|---|---|---|
| `name` | str | `"<md>:<line>"` | Human label; used as the pytest param id. |
| `skip` | str | — | Recognised but not executed; value is the skip reason. Keeps the block "claimed" so accidental changes surface. |
| `network` | bool | `false` | Needs internet, **no** credentials. Emits `@pytest.mark.network`. |
| `hf_token` | bool | `false` | Needs an `HF_TOKEN` credential. **Implies** `network`; emits both `@pytest.mark.hf` and `@pytest.mark.network`. |
| `timeout` | int | `300` | Per-command wall-clock seconds. |
| `extra` | str (argv) | — | Tokens **appended** after the command (test-only limiters). Parsed with `shlex.split`. |
| `override` | list of str | — | Per-flag substitutions/deletions applied to the command (see below). |
| `extras` | str or list | — | Required pip extras → drive `pytest.importorskip` guards (e.g. `lora`, `rl`). |
| `env` | mapping | — | Environment variables to set for the run. `{tmp}` is expanded. |
| `cwd` | str | `{tmp}` | Working directory for the run. |

Unknown keys are a **hard error** (guards against typos like `netwrok`).

#### `{tmp}` placeholder

`{tmp}` expands to the per-test temporary run directory.  It is valid in
`override`, `env`, and `cwd`.

#### `override` semantics

Each entry is one of:

* `--flag value…` → **replace** the value of `--flag` (all existing
  occurrences are dropped, this one is inserted at the original position).
* `--flag` (bare) → replace a bare flag's value set (rare).
* `-.--flag` (leading `-.`) → **delete** `--flag` and its value from the
  command.  (`-.` is the marker's "delete" sigil; a real CLI never starts a
  flag with `-.`.)

Overrides are matched against each command in the fence by flag name.  An
`override` entry that matches no command is a **hard error** (guards against
a stale override after the README command changed).

`override` entries may themselves contain `{tmp}`.  Where two commands in the
same fence share a flag name (e.g. both use `--dataset`), the entry applies to
**both** — use distinct flags or split the fence if per-command targeting is
ever needed (no current block requires it; see s 10.3).

#### Automatic `--cache_dir` injection

For every core command that does not already specify `--cache_dir`, the
harness appends `--cache_dir {tmp}/hf_cache`.  This guarantees all HuggingFace
downloads land inside the removable temp dir (s 3.4).

### 3.2 Multi-command fences

A single fence may contain several commands (e.g. "Step 1 … Step 2 …").

**One fence = one test.**  All commands in the fence run **sequentially, in
order, in the same working directory**, so a later command can consume an
earlier command's output (e.g. a checkpoint written by Step 1).  The marker
applies to the whole fence; `override` entries are matched per-command.

### 3.3 Execution without installing the package

The suite must pass on a **clean checkout with no `pip install`**.  Therefore
the harness never relies on the console-script shims
(`genml-kit-train`, …), which only exist after installation.  Instead, at
execution time it rewrites the command:

| README command | Executed as |
|---|---|
| `genml-kit-train …` | `[sys.executable, "-m", "genml_kit.training.train", …]` |
| `genml-kit-infer …` | `[sys.executable, "-m", "genml_kit.training.infer", …]` |
| `genml-kit-llm …` | `[sys.executable, "-m", "genml_kit.llm_cli", …]` |

(All three modules have an `if __name__ == "__main__"` guard — verified.)

The subprocess environment is:

```
env["PYTHONPATH"] = <repo_root> + os.pathsep + env.get("PYTHONPATH", "")
```

so `import genml_kit` resolves from the checkout.  The README keeps the
pretty `genml-kit-*` spelling; the rewrite exists only in the harness.

Any command in a candidate fence whose form is **not** a core entry point is a
**hard error** — this is what pins the s 2.2 scope.

### 3.4 Temporary-directory discipline

* pytest's `tmp_path` is used only as a *parent*.
* The harness creates `run_root = tmp_path / "readme_cli"`.
* A `try/finally` always does `shutil.rmtree(run_root, ignore_errors=True)`,
  so the directory is gone even on failure — we do **not** rely on pytest's
  automatic `tmp_path` retention.
* All `{tmp}` expansions, `--cache_dir`, checkpoints, synthesized inputs, and
  command outputs live under `run_root`.

### 3.5 Markers: `network` vs `hf`

Two **orthogonal** axes:

| Marker key | pytest marker(s) | Means | Examples today |
|---|---|---|---|
| `network: true` | `network` | internet, **no** credentials | `cifar10`, `google/vit-*`, `sshleifer/tiny-gpt2`, `timm:*`, `marmal88/...`, `scene_parse_150`, gym envs |
| `hf_token: true` | `hf` **and** `network` | needs `HF_TOKEN` | none today; future gated/private repos |

Notes:

* **timm counts as network.**  `timm.create_model(..., cache_dir=...)` fetches
  the public `timm/*` repos from the HuggingFace Hub.  timm does **not** need
  a token (the `timm/*` repos are public/non-gated), so those blocks are
  `network`, not `hf_token`.
* The public HF IDs used in the READMEs (`cifar10`, `marmal88/skin_cancer`,
  `scene_parse_150`, `sshleifer/tiny-gpt2`, `google/vit-*`,
  `facebook/convnextv2-*`) are un-gated: network only, no token.  Hence **no
  block gets `hf_token: true` today.**
* The harness **inherits** `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` from the
  environment if present (harmless, helps rate limits) but never sets or
  requires them for `network` tests.

`pyproject.toml`:

```toml
markers = [
  "slow: long-running learning/convergence tests (deselected by default; run with -m slow)",
  "network: command downloads models/data over the internet (deselected by default)",
  "hf: command additionally requires HF_TOKEN (implies network; deselected by default)",
]
addopts = "-ra --strict-markers --strict-config -m 'not slow and not network and not hf'"
```

Opt-in runs:

* tokenless internet: `pytest -m network`
* credentialed:       `pytest -m hf`
* everything online:  `pytest -m 'network or hf'`

### 3.6 README defects surfaced during review (to fix in this change)

1. **Shell typo.** The segmentation blocks (root README "Segmentation … Quick
   start" and the `--mask_column` example) print `genml_kit-train …` with an
   **underscore**.  That binary does not exist; it must be `genml-kit-train`.
   (Blocks [18], [19].)
2. **Unlabeled fences.** The three RL example fences use ` ``` ` with no info
   string; relabel to ` ```bash ` so they are candidates.  (Blocks [24]–[26].)
3. **Ellipsis-terminated blocks.** Several blocks end in `...` and lack a
   dataset (gradient checkpointing, param-rename, LoRA+custom-classifier,
   custom loss/model, RL loss).  Where the block is marked, complete it into a
   real command and drop the `...`.  (Blocks [10]–[15], [16], [28], [32].)
4. **Possible checkpoint-path bug.** The SFT → DPO → reward-model recipe has
   DPO read `./sft/checkpoint_best.pt` while Step 1 writes `--output_dir ./sft`
   without a matching `--checkpoint` prefix.  Verification will reveal whether
   the path resolves; fix the README if not.  (Block [29].)

### 3.7 Data synthesis (offline substitutes)

Where a command references a placeholder dataset/path, the marker `override`
substitutes a value pointing at a **minimal synthesized** fixture created under
`{tmp}` (full catalogue in s 9):

| Needs | Synthesized fixture |
|---|---|
| image classification / simmim / supcon / distillation | tiny `imagefolder/` (2 classes × ~4 images) |
| segmentation (paired tree) | tiny `segfolder/` with `images/` + `masks/` |
| segmentation (HF `--mask_column`) | tiny **HF-style local dataset with a mask feature** |
| causal LM / SFT | tiny text corpus (`text_column`) |
| preference / DPO / reward model | tiny chosen/rejected `preference` pair set |
| `genml-kit-llm generate` | `prompts.txt` (one prompt per line) |
| `genml-kit-llm eval` | tiny eval corpus (`text`/`prompt`/`reference` columns) |
| inference | `_best.pt` checkpoint + a tiny image |
| external teacher | tiny teacher checkpoint |
| custom `--loss` / `--model` / RL loss | minimal `my_loss.py` / `my_model.py` / `my_ppo_entropy.py` |

**Limiting strategy** (README keeps the real value; the test bounds it):

* `--epochs 1` (append via `extra`).
* Image pipelines do **not** honour `--max_items` (only `text`, `preference`,
  `llm` do) → for image blocks the limiter is a **self-sized tiny fixture**
  plus `--epochs 1`.
* VO: `--vo_length` / `--vo_val_length` trim, plus `--epochs 1`.
* `network` blocks: also capped by `timeout`.

---

## 4. New / changed files

### 4.1 `tests/test_readme_cli.py` (new, self-contained)

Per the agreement, the parsing + execution logic lives **inside the test
module** (not in `scripts/`, not in the installed package).  Full sketch in
s 8.

### 4.2 `pyproject.toml`

* register `network` and `hf` markers (s 3.5);
* `addopts` default deselection of `slow`, `network`, `hf`;
* add `pyyaml` to the `[dev]` extra.

### 4.3 READMEs

* insert `<!-- genml-cli-test … -->` markers per s 5 / s 10;
* apply the s 3.6 defects (typo fix, RL relabel, complete `...` blocks);
* a short "Verifying README commands" paragraph in the Development section of
  the root README explaining the marker and the `pytest -m network` /
  `pytest -m hf` invocations.

---

## 5. Per-block decisions

Complete inventory of in-scope fences and their treatment.  "ovr" = `override`,
"extra" = appended limiter.  All synthesized paths are under `{tmp}`.  Verbatim
source text for every block is in Appendix A.

| # | Location / heading | Core cmds | Treatment |
|---|---|---|---|
| 01 | Quick Start (cifar10 + `google/vit-base-patch16-224`) | 1 | `network`; extra `--epochs 1 --max_items 16`; keep ID as written |
| 02 | Pre-Training: Full Pipeline (SimMIM Step 1 + FT Step 2) | 2 | one test; synth `imagefolder`; ovr Step1 `--datasets`, Step2 `--dataset`; delete Step2 `--source_checkpoint`; extra `--epochs 1` |
| 03 | Pre-Training: Supervised Contrastive | 1 | synth `imagefolder`; ovr `--datasets`; extra `--epochs 1` |
| 04 | Fine-Tuning: Basic | 1 | `network`; ovr `--dataset`; extra `--epochs 1` |
| 05 | Fine-Tuning: Custom Classifier Head | 1 | `network`; ovr `--dataset`; extra `--epochs 1` |
| 06 | Fine-Tuning: LoRA | 1 | `network`; `extras: lora`; ovr `--dataset`; extra `--epochs 1` |
| 07 | Fine-Tuning: From A Pre-Trained Checkpoint | 1 | ovr `--dataset`; **delete `--source_checkpoint`**; extra `--epochs 1` |
| 08 | `--seed 42 ...` | 1 | **unmarked** (skip: illustrative ellipsis, no dataset) |
| 09 | Cross-Dataset Resume | 1 | `network`; ovr `--dataset`; extra `--epochs 1` |
| 10 | Gradient Checkpointing (`...`) | 1 | **fix README** (add dataset, drop `...`); `network`; extra `--epochs 1` |
| 11 | LoRA + Custom Classifier (`...`) | 1 | **fix README**; `network`; `extras: lora` |
| 12 | Checkpoint prefix + `--param_rename` (`...`) | 1 | **fix README**; `network`; ovr `--dataset`; delete `--source_checkpoint` |
| 13 | Custom Loss (`--loss ./my_loss.py ...`) | 1 | **fix README** (add dataset, drop `...`); synth `my_loss.py`; `network` |
| 14 | Custom Models (`--model ./my_model.py ...`) | 1 | **fix README**; synth `my_model.py`; `network` |
| 15 | RL Loss (`--method ppo --loss ./my_ppo_entropy.py ...`) | 1 | **fix README**; synth script; add CartPole env; `extras: rl` |
| 16 | Detection: tiny-coco note | 2 | synth `detfolder`; drop `scripts/` line + `...`; extra `--epochs 1` |
| 17 | Detection: Example | 1 | synth `detfolder`; ovr `--dataset`; keep `timm/resnet50` (`network`); extra `--epochs 1` |
| 18 | Segmentation: Quick start | 1 | **fix typo**; synth `segfolder`; ovr `--dataset`; keep `timm/resnet34` (`network`); extra `--epochs 1` |
| 19 | Segmentation: HF `--mask_column` | 1 | **fix typo**; synth HF-style mask dataset; ovr `--dataset`; `network`; extra `--epochs 1` |
| 20 | causal LM: Quick start | 1 | synth corpus; ovr `--dataset`; `extras: lora`; `network`; extra `--epochs 1` |
| 21 | causal LM: LoRA + grad-ckpt + bf16 | 1 | synth corpus; ovr `--dataset`; ovr `--model`→`sshleifer/tiny-gpt2`; `extras: lora`; `network` |
| 22 | `genml-kit-llm generate` (2 cmds) | 2 | synth `prompts.txt` (cmd 1); **edit README** so cmd 2 is self-contained; synth checkpoint/adapter; `network` |
| 23 | `genml-kit-llm eval` | 1 | synth eval corpus; ovr `--dataset`; `network` |
| 24 | RL Quick start (CartPole/Pendulum/LunarLander) | 3 | **relabel `bash`**; `extras: rl`; extra `--epochs 1`; keep envs (skip if Box2D absent) |
| 25 | RL model registry | 4 | **relabel `bash`**; `extras: rl`; extra `--epochs 1` |
| 26 | RL serious PPO | 1 | **relabel `bash`**; `extras: rl`; extra `--epochs 1` |
| 27 | VO: Quick start (2 cmds) | 2 | **offline**; extra `--epochs 1 --vo_length N --vo_val_length M` |
| 28 | prepared_images SimMIM (`...`) | 1 | **fix README** (drop `...`); synth imagefolders; ovr `--datasets`; extra `--epochs 1` |
| 29 | Preference: SFT → DPO → reward chain | 3 | one test; synth `sft_corpus` + `preferences`; `extras: lora`; `network`; fix README if checkpoint path wrong |
| 30 | Distillation: Quick start (ema + external, 2 cmds) | 2 | `network`; test both; synth tiny teacher checkpoint for cmd 2; extra `--epochs 1` |
| 31 | `scdiag/README.md` Canonical Commands (5 cmds) | 5 | one test; `network`; synth imagefolder for cmds 2–4, tiny HF IF for `marmal88` cmds 1–3+5; delete `--source_checkpoint`; synth lesion image + checkpoint for inference; extra `--epochs 1` |
| 32 | Inference Guide | 1 | synth image + checkpoint; ovr `--checkpoint` + image args; `network` |

Overall: **32 in-scope fences**, all marked except #08 (`--seed 42 ...`).

---

## 6. Validation workflow

1. `format_file` every touched `.py` with **yapf** + `.style.yapf`
   (Google style, 2-space indent).
2. `ruff check` — no annotation comments added (will ask first if ever needed).
3. Run the fast unit tests (always).
4. Run the reachable execution tests locally where feasible:
   * offline blocks (`pytest -m 'not network and not hf'`),
   * `network` blocks (`pytest -m network`) where connectivity exists,
   * report which pass / skip (e.g. token, Box2D, missing data).
5. Show the full `git diff` for review.  **No commit** until approval.

---

## 7. CI integration

The README CLI suite is deliberately **not wired into GitHub Actions yet**.
`.github/workflows/ci.yml` keeps its original shape: the slow job still runs
plain `pytest -q -m slow` and `pyproject.toml` does not register the `network`
/ `hf` markers.  Wiring the job in before `tests/test_readme_cli.py` exists
makes CI fail outright, because `-m 'slow and not network and not hf'` aborts
with `ERROR: Unknown marker(s) in '-m' expression: hf, network` (a marker is
only known to `-m` once it is registered in `[tool.pytest.ini_options]`).

Enable the job **as part of the commit that adds the test module** — s 7.3 has
the exact steps and YAML.

### 7.1 Token-free by design

Every resource referenced by the in-scope README commands is **public and
un-gated** (`cifar10`, `google/vit-*`, `sshleifer/tiny-gpt2`, `timm/*`,
`marmal88/skin_cancer`, `scene_parse_150`, `facebook/convnextv2-*`).  They
require the internet but **no** HuggingFace credentials.  The CI job therefore
runs:

```bash
pytest -q -m 'network and not hf'
```

Note that `network` (internet) and `hf` (credentials) are distinct markers
(s 3.5); `network and not hf` selects exactly the tokenless internet tests.

Two practical CI details:

* `pytest` exits **5** ("no tests collected") when no network tests exist yet.
  While the suite is being populated, the job treats exit 5 as success.
* When the job is enabled, the slow job must be narrowed to
  `pytest -q -m 'slow and not network and not hf'`, because the execution
  tests are `slow` **and** `network`; this keeps the slow job offline.  Do not
  narrow it before the markers are registered (see the note in s 7).

### 7.2 If a gated resource is ever added

The only case that needs credentials is a gated/private model or dataset
(marked `hf_token: true`).  **Do not hardcode a token.**  HuggingFace supports
keyless auth to the Hub via **Trusted Publishers** (OIDC), analogous to PyPI's
— available since `huggingface_hub >= 1.19.0`.

Mechanism: the CI provider issues a short-lived OIDC ID token describing the
job; the `hf` CLI exchanges it at `https://huggingface.co/oauth/token`
(RFC 8693) for a **60-minute**, tightly-scoped Hub token, validated against a
trusted publisher configured on the Hub.

Two flavors:

| Flavor | Configured on the Hub | Token grants | Use |
|---|---|---|---|
| Repo trusted publisher | repo → Settings → Trusted Publishers | write + read, **that one repo** | publishing models/datasets from CI |
| Account CI/CD identity | Auth settings → CI/CD Access | **read-only** (`gated-repos`) | reading **gated** repos you can already access |

CI/CD identities **cannot read private repos** — only gated ones granted to
the account.  For the test suite we would use the **CI/CD identity** flavor
(read-only, gated), not a repo publisher.

Sketch (only if/when a gated block is added):

```yaml
    permissions:
      id-token: write     # allow the job to request an OIDC token
      contents: read
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev]"
      - name: Exchange OIDC for a scoped Hub token (no secret)
        env:
          HF_OIDC_RESOURCE: <hf-username>   # user-scoped -> gated-repos read
        run: echo "HF_TOKEN=$(hf auth token)" >> "$GITHUB_ENV"
      - run: pytest -q -m 'network and not hf'
      - run: pytest -q -m hf                 # uses the exchanged HF_TOKEN
```

The harness already **inherits** `HF_TOKEN` / `HUGGING_FACE_HUB_TOKEN` from the
environment (s 3.5), so nothing else changes: `hf`-marked tests simply skip
when the variable is absent and run when it is present.

Caveats: tokens live 60 minutes with no refresh for pre-minted flows (GitHub
Actions flows are cached/refreshed by `huggingface_hub`); trusted-publisher
claims match **exactly** (no regex) so pin `repository`/`branch`/`workflow`;
managing repo publishers needs the Hub **Write** role; only GitHub Actions is
auto-minted (other providers must pre-mint via `HF_OIDC_ID_TOKEN`).

### 7.3 Turning the CI job on (do this with the test module commit)

The workflow is currently **clean** — no `readme-cli` job and no
`network`/`hf`/`slow and not ...` expression.  Flip it on in the *same* commit
that lands `tests/test_readme_cli.py` and the marker registration, in this
order, otherwise CI goes red ("Unknown marker(s) in '-m' expression").

1. **Register the markers first** (`pyproject.toml`, exact diff in s 8.7).
   Without `network` and `hf` in `[tool.pytest.ini_options].markers`, any
   `-m` expression naming them is a hard error under `--strict-markers`.

2. **Narrow the slow job** in `.github/workflows/ci.yml`:

   ```yaml
         - name: Run slow tests
           run: pytest -q -m 'slow and not network and not hf'
   ```

   (was `pytest -q -m slow`).  The execution tests are `slow` **and**
   `network`, so this keeps the slow job offline.

3. **Add the `readme-cli` job** as a sibling of `test:` under `jobs:`:

   ```yaml
     readme-cli:
       runs-on: ubuntu-latest

       steps:
         - name: Check out repository
           uses: actions/checkout@v4

         - name: Set up Python
           uses: actions/setup-python@v5
           with:
             python-version: "3.11"

         - name: Install package and development dependencies
           run: python -m pip install -e ".[dev,llm,llm-spm,timm,uvito,lora,s3,rl]"

         - name: Run README CLI examples (tokenless network)
           run: |
             set +e
             pytest -q -m 'network and not hf'
             code=$?
             # Exit code 5 == "no tests collected": not an error while the
             # marker suite is still being populated.
             if [ "$code" -eq 5 ]; then
               echo "No network README CLI tests collected yet."
               exit 0
             fi
             exit "$code"
   ```

   Token-free (no secrets); s 7.2 covers the OIDC path if a gated resource is
   ever added.

**Sanity check before pushing** (the failure mode that produced the CI red):

```bash
pytest -q -m 'slow and not network and not hf'   # must NOT say "Unknown marker(s)"
pytest -q -m 'network and not hf'                # exit 5 (none yet) is fine post-registration
```

If `-m` still reports unknown markers, step 1 did not land — the marker
registration is what makes the expressions legal, not the workflow.

---

## 8. Implementation reference

This section is the "full clear picture" for whoever implements the plan.
It is intentionally concrete: exact module constants, the parser contract,
the runner contract, and a code sketch.

### 8.1 `tests/test_readme_cli.py` — module header and constants

```python
"""Verify the copy-pasteable CLI examples in the READMEs.

A markdown HTML comment (invisible on GitHub) immediately above a ```bash
fence opts that fence in.  Each marked fence is executed as a bounded smoke
test that writes everything under a removable temp dir.  The package is
imported via PYTHONPATH (no install); the ``genml-kit-*`` console scripts are
rewritten to ``python -m genml_kit...`` so the suite needs no editable install.
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

REPO_ROOT = Path(__file__).resolve().parents[1]

# Fence opener: leading whitespace, >=3 backticks/tildes, an info string.
_FENCE_RE = re.compile(r"^(?P<indent>\s*)(?P<ticks>`{3,}|~{3,})\s*"
                       r"(?P<info>[A-Za-z0-9_+.\-]*)\s*$")

# HTML comment carrying the marker (may span multiple lines).
_MARKER_RE = re.compile(r"<!--\s*genml-cli-test\b(?P<body>.*?)-->", re.DOTALL)

# README command -> python -m module.  Both hyphen (correct) and underscore
# (typo still present in the tree until s 3.6.1 lands) spellings map.
_ENTRY_MODULES = {
    "genml-kit-train": "genml_kit.training.train",
    "genml-kit-infer": "genml_kit.training.infer",
    "genml-kit-llm": "genml_kit.llm_cli",
    "genml_kit-train": "genml_kit.training.train",
    "genml_kit-infer": "genml_kit.training.infer",
    "genml_kit-llm": "genml_kit.llm_cli",
}

_README_GLOBS = ["README.md", "*/README.md"]

_ALLOWED_KEYS = {"name", "skip", "network", "hf_token", "timeout", "extra",
                 "override", "extras", "env", "cwd"}

_EXTRA_IMPORTS = {"lora": "peft", "llm": "transformers", "rl": "gymnasium",
                  "timm": "timm"}

_DEFAULT_TIMEOUT = 300
```

### 8.2 Data model

```python
@dataclass
class CommandSpec:
  md: str                 # repo-relative path, e.g. "README.md"
  line: int               # 1-based line of the opening fence
  heading: str            # nearest preceding "##/### ..." heading
  name: str               # marker name, or f"{md}:{line}"
  commands: list          # list[list[str]]: each inner list is one argv
  network: bool = False
  hf_token: bool = False
  timeout: int = _DEFAULT_TIMEOUT
  extra: list = field(default_factory=list)      # argv appended to each cmd
  override: list = field(default_factory=list)   # override spec strings
  extras: list = field(default_factory=list)     # pip extras -> importorskip
  env: dict = field(default_factory=dict)
  cwd: str = "{tmp}"
  skip: str | None = None
```

### 8.3 Parser contract

`read_markers(md_path) -> list[CommandSpec]`

1. Read the file; split into lines.
2. Track the current heading (lines starting with `#`).
3. Walk lines; on a line matching `_FENCE_RE`:
   * determine whether a `_MARKER_RE` comment appears immediately above
     (allowing only blank lines between the comment and the fence);
   * if the info string is in `{"bash", "sh"}` **and** a marker precedes it,
     collect the body up to the closing fence and parse it;
   * if the info string is `""` (unlabeled) **and** a marker precedes it →
     **hard error** ("relabel this fence to ```bash"); this is how the
     relabel requirement in s 3.6.2 is enforced;
   * otherwise (no marker) skip the fence.
4. `yaml.safe_load(marker_body)`; confirm it is a mapping; unknown keys →
   `ValueError`.
5. Validate: `hf_token` implies `network`; `network is False and hf_token` is
   an error; `timeout` int > 0; `extra` parses under `shlex.split`;
   `override` entries are strings.
6. Split the body into commands (see below); confirm at least one command
   whose first token is in `_ENTRY_MODULES`; otherwise the fence is not a
   candidate (marker on a command-less fence → error, to catch mistakes).

**Command splitting** (`split_commands(body)`):

* join backslash-continued lines (`\\` at end-of-line) into one logical line;
* drop full-line `#` comments;
* start a new command whenever a logical line's first token is in
  `_ENTRY_MODULES`;
* any other non-empty logical line that appears *between* commands (e.g. the
  trailing "# Then fine-tune …" note) is dropped if it is a comment; a
  non-comment, non-core line is a **hard error** (scope guard, s 2.2);
* each command is tokenized with `shlex.split(posix=True)`.

`build_argv(spec, command, run_root) -> list[str]`

1. `cmd = list(command)`.
2. Apply `override` entries by flag name (replace = drop existing
   flag+value(s), insert at the first occurrence's position; delete = remove
   flag+value).  Unmatched entry → **hard error**.
3. Append `spec.extra`.
4. If `cmd[0]` is a core entry point and `--cache_dir` is absent, append
   `["--cache_dir", str(run_root / "hf_cache")]`.
5. Rewrite the head: `cmd = [sys.executable, "-m",
   _ENTRY_MODULES[cmd[0]]] + cmd[1:]`.
6. Expand `{tmp}` in every remaining token.

### 8.4 Runner contract

```python
def run_spec(spec, tmp_path):
  run_root = tmp_path / "readme_cli"
  run_root.mkdir(parents=True, exist_ok=True)
  try:
    for extra in spec.extras:
      pytest.importorskip(_EXTRA_IMPORTS[extra], reason=f"needs [{extra}]")
    if spec.hf_token and not _hf_token():
      pytest.skip("requires HF_TOKEN")
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO_ROOT), env.get("PYTHONPATH", "")]).strip(os.pathsep)
    env.update({k: v.format(tmp=run_root) for k, v in spec.env.items()})
    cwd = spec.cwd.format(tmp=run_root)
    for command in spec.commands:
      argv = build_argv(spec, command, run_root)
      proc = subprocess.run(argv, cwd=cwd, env=env, capture_output=True,
                            text=True, timeout=spec.timeout)
      if proc.returncode != 0:
        pytest.fail(_failure_message(spec, argv, proc))
  finally:
    shutil.rmtree(run_root, ignore_errors=True)
```

* `_hf_token()` → `os.environ.get("HF_TOKEN") or
  os.environ.get("HUGGING_FACE_HUB_TOKEN")`.
* `_failure_message` includes `md:line`, heading, argv, return code, and the
  last ~40 lines of stdout + stderr.
* `timeout` expiry raises `subprocess.TimeoutExpired`; wrap it and
  `pytest.fail("timed out after Ns")` with the same context.
* `env` values are `{tmp}`-formatted (e.g. `HF_HOME`); the *keys* are used
  literally.

### 8.5 Test parametrization

```python
def _collect_specs():
  specs = []
  for pattern in _README_GLOBS:
    for md in sorted(REPO_ROOT.glob(pattern)):
      specs.extend(read_markers(md))
  return specs

def _param(spec):
  marks = [pytest.mark.slow]
  if spec.network:
    marks.append(pytest.mark.network)
  if spec.hf_token:
    marks.append(pytest.mark.hf)       # hf implies network (both emitted)
  return pytest.param(spec, marks=marks, id=spec.name)

_SPECS = _collect_specs()

@pytest.mark.parametrize("spec", [_param(s) for s in _SPECS])
def test_readme_cli(spec, tmp_path):
  if spec.skip:
    pytest.skip(spec.skip)
  run_spec(spec, tmp_path)
```

`read_markers` is imported once at module load; collection errors (bad
markers) therefore fail fast and loudly.

### 8.6 Unit tests (fast, unmarked)

All run with **no** subprocess and **no** network (mark them `not slow`, i.e.
they are ordinary tests).  Cases:

1. marker above a `bash` fence is detected; identical fence without a marker
   is ignored;
2. multi-line comment body parsed as YAML;
3. unknown key raises;
4. malformed YAML raises;
5. marker above an *unlabeled* fence raises (relabel needed);
6. `hf_token: true` yields both `network` and `hf`; `network: false` +
   `hf_token: true` raises;
7. `{tmp}` expands in `override` / `env` / `cwd`;
8. `--cache_dir` injected only when absent;
9. `override` replace keeps position; delete removes flag + value; unmatched
   override raises;
10. multi-command fence: ordering preserved; `\`-continuations and `#`
    comments handled; a non-core chunk raises;
11. `_ENTRY_MODULES` rewrite maps both hyphen and underscore forms;
12. **count assertion**: `len(_SPECS) == <expected>` so a README edit that
    drops a marker is caught.

### 8.7 `pyproject.toml` diff (exact)

```toml
[tool.pytest.ini_options]
minversion = "8.0"
testpaths = ["tests"]
addopts = "-ra --strict-markers --strict-config -m 'not slow and not network and not hf'"
markers = [
  "slow: long-running learning/convergence tests (deselected by default; run with -m slow)",
  "network: command downloads models/data over the internet (deselected by default)",
  "hf: command additionally requires HF_TOKEN (implies network; deselected by default)",
]
filterwarnings = [
  "error",
  "ignore:\\[Errno 13\\] Permission denied.*joblib:UserWarning",
]
```

In `[project.optional-dependencies]`:

```toml
dev = ["pytest", "ruff", "yapf", "pyyaml"]
```

---

## 9. Fixture catalogue (synthesized under `{tmp}`)

Every fixture is built by a small helper in the test module.  Sizes are
deliberately tiny (smoke, not convergence).

| Fixture | Helper | Structure | Used by blocks |
|---|---|---|---|
| ImageFolder | `_make_imagefolder(root, n_classes=2, n_per=4)` | `<root>/<class>/*.png` (PIL, 32x32, deterministic) | 02, 03, 04, 05, 06, 07, 09, 10, 11, 12, 16, 17, 31 |
| segfolder | `_make_segfolder(root, n=4)` | `<root>/images/*.png` + `<root>/masks/*.png` (stem-matched) | 18 |
| HF-style mask dataset | `_make_hf_seg_dataset(root)` | `DatasetDict` saved with `datasets` (`image` + `annotation` mask) | 19 |
| text corpus | `_make_text_corpus(root)` | HF `Dataset` from dict `{"text": [...]}` | 20, 21 |
| instruction corpus | `_make_instr_corpus(root)` | `{"prompt": [...], "response": [...]}` | 29 cmd 1 |
| preference pairs | `_make_preference(root)` | `{"prompt": [...], "chosen": [...], "rejected": [...]}` | 29 cmds 2–3 |
| prompts file | `_make_prompts(root)` | `prompts.txt`, one prompt per line | 22 cmd 1 |
| eval corpus | `_make_eval_corpus(root)` | `{"text": [...], "prompt": [...], "reference": [...]}` | 23 |
| detection tree | `_make_detfolder(root)` | `images/*.jpg` + `annotations.json` (COCO `{"images","annotations","categories"}`) | 16, 17 |
| inference checkpoint | `_make_inference_checkpoint(root)` | `<name>_best.pt` + one `lesion.jpg` | 32, 31 cmd 5 |
| teacher checkpoint | `_make_teacher_checkpoint(root)` | tiny `convvit` `..._best.pt` | 30 cmd 2 |
| custom loss script | `_make_custom_loss(root)` | `my_loss.py` defining `build_loss(**kw)` | 13 |
| custom model script | `_make_custom_model(root)` | `my_model.py` defining `create_model(num_labels, **kw)` | 14 |
| RL loss script | `_make_rl_loss(root)` | `my_ppo_entropy.py` defining `build_loss(**kw) -> dict` | 15 |

**Dataset spec syntax** (verified against `genml_kit/datasets/factory.py`):

* `imagefolder/<path>` → local `ImageFolder` tree.
* `segfolder/<path>` → paired `images/` + `masks/`.
* `detfolder/<path>` → `images/` + `annotations.json`.
* an **existing directory path** → treated as `ImageFolder`.

Because the synthesized temp path is absolute and exists, overrides use the
**bare absolute path** (`{tmp}/img`) wherever possible — it is unambiguous and
avoids the `imagefolder/` prefix-and-slash question entirely.  The
`imagefolder/` / `segfolder/` / `detfolder/` prefixes are only needed when the
README command itself already uses them (blocks [02], [16], [18] override the
value, so bare paths are fine).

**HuggingFace-style fixtures:** `_make_hf_seg_dataset`, `_make_text_corpus`,
etc. build an in-memory `datasets.Dataset` and `save_to_disk(root)`, then pass
`{tmp}/<name>` as `--dataset`.  `load_dataset` accepts a saved directory path
— confirm the exact reader during implementation (`load_from_disk` vs
`load_dataset(path)`); adjust the override if the pipeline branches on
`os.path.isdir`.

---

## 10. README edits (exact specifications)

### 10.1 Typo fix (blocks [18], [19])

Replace `genml_kit-train` with `genml-kit-train` in the two segmentation
command blocks.  Nothing else on those lines changes.

### 10.2 Marker insertion pattern

For each block, insert the comment immediately above the opening fence, at
column 0.  Example for block [01]:

````markdown
<!-- genml-cli-test
name: quick-start-cifar10
network: true
extra: --epochs 1 --max_items 16
-->
```bash
# Fine-tune a ViT on an image-classification dataset (5 epochs, ~2 minutes on GPU)
genml-kit-train --model google/vit-base-patch16-224 \
                --dataset cifar10 \
                --label_column label \
                --epochs 5 \
                --batch_size 32 \
                --lr 3e-5 \
                --image_size 224
```
````

### 10.3 Blocks needing the README command completed

These get a **content** edit (drop `...`, add a dataset) in addition to the
marker.  The dataset placeholder used is `my-org/my-labeled-images`, matching
the style already used elsewhere in the README; the marker overrides it to a
`{tmp}` fixture.

| Block | Edit |
|---|---|
| [10] Gradient checkpointing | add `--dataset my-org/my-labeled-images`, drop `...` |
| [11] LoRA + custom classifier | add `--dataset my-org/my-labeled-images`, drop `...` |
| [12] param rename | add `--dataset my-org/my-labeled-images`, drop `...`; marker deletes `--source_checkpoint` |
| [13] custom loss | add `--dataset my-org/my-labeled-images`, drop `...` |
| [14] custom model | add `--dataset my-org/my-labeled-images`, drop `...` |
| [15] RL loss | add `--pipeline rl --env_id CartPole-v1`, drop `...` |
| [16] detection note | drop the `python scripts/make_tiny_coco.py …` line (out of scope); keep the pointer to it as prose; drop `...` |
| [28] prepared_images | drop `...` |
| [32] `scdiag` cmds 3 | drop `...` (keep `./derm1m_images ./ham10000_grouped`, overridden by the marker) |
| [22] llm generate cmd 2 | make self-contained: keep `--dataset` + `--checkpoint`; the marker supplies a synthesized corpus/checkpoint and drops `--lora_adapter` if the adapter fixture is not built |

> Rationale for `my-org/my-labeled-images`: it reads naturally to a user,
> matches existing placeholder style, and is trivially overridden to a
> `{tmp}` fixture.  Leaving `...` would mean the block is not a real command
> and cannot be tested.

### 10.4 RL fences (blocks [24], [25], [26])

Change the opening fence from ` ``` ` to ` ```bash `; the closing fence is
unchanged.  No command-text changes.

### 10.5 "Verifying README commands" paragraph

Add to the root README "Development" section, after the yapf/ruff paragraph:

> ### Verifying README commands
>
> Command blocks preceded by a `<!-- genml-cli-test … -->` HTML comment are
> executed by `tests/test_readme_cli.py`.  The comment is invisible on
> GitHub; its YAML body controls limiting args, network/token markers, and
> substitutions.  Run them with `pytest -m network` (needs internet, no
> credentials) — the offline suite excludes them by default.

---

## 11. Acceptance matrix

| Check | Command | Expected |
|---|---|---|
| Unit tests | `pytest tests/test_readme_cli.py -m 'not slow'` | pass, no network, no subprocess |
| Offline blocks | `pytest tests/test_readme_cli.py -m 'slow and not network and not hf'` | VO block [27] passes; no network |
| Network blocks (sandbox) | `pytest -m network` | run or skip gracefully; failures reported with md:line + argv |
| Default run | `pytest` | slow/network/hf deselected; existing suite unaffected |
| Slow job parity | `pytest -m 'slow and not network and not hf'` | the 5 pre-existing slow tests, unchanged |
| Marker coverage | count assertion (s 8.6#12) | every in-scope fence marked except [08] |
| Markdown math | `python scripts/md_latex.py` | 0 findings (this doc + READMEs) |
| Lint/format | `ruff check .` + `format_file` (yapf) | clean, 2-space indent |

---

## 12. Open items / risks

* **Network blocks can't be fully validated in an offline sandbox.** They
  will be authored, rate-limited by `timeout`, and marked `network`; once the
  `readme-cli` job is enabled (s 7.3), CI confirms them on a connected runner.
* **`--max_items` is ignored by image/segmentation pipelines** (only
  `text`/`preference`/`llm` honour it).  Image blocks therefore rely on tiny
  self-sized fixtures + `--epochs 1`.
* **Tiny-checkpoint synthesis** for inference (blocks [32], [31] cmd 5):
  training a 1-epoch model is slow.  Preferred: instantiate the model and
  `torch.save` its state dict in the registry's expected format — confirm the
  exact dict shape against `genml_kit/io/checkpointing.py` during
  implementation.
* **Box2D env** (`LunarLanderContinuous-v3`) may be unavailable → run marked
  `extras: rl` but skipped if the env can't be created (per decision).
* **HF-style local fixtures**: confirm `load_dataset(path)` vs
  `load_from_disk(path)` for the saved-directory form (s 9).
* **Doc token**: `hf_token` is unused by any block today; keep the marker
  supported and unit-tested even though s 5 sets it nowhere.
* **Marker drift**: the count-assertion unit test (s 8.6#12) is the guard.

---

## Appendix A — Verbatim in-scope blocks

Exact current text of each in-scope fence, in document order.  `→ [NN]` links
to the s 5 numbering.  Backslash continuations are shown as in the source.
If a README changes, update this appendix too.

### A.1 `README.md`

**A.1.1 Quick Start → [01]**

````markdown
```bash
# Fine-tune a ViT on an image-classification dataset (5 epochs, ~2 minutes on GPU)
genml-kit-train --model google/vit-base-patch16-224 \
                --dataset cifar10 \
                --label_column label \
                --epochs 5 \
                --batch_size 32 \
                --lr 3e-5 \
                --image_size 224
```
````

**A.1.2 Pre-Training: Full Pipeline (two commands) → [02]**

````markdown
```bash
# Step 1: Pre-train with SimMIM on two large image collections.
# v4.2: pre-training is a *configuration* of the unified genml-kit-train
# CLI (--pipeline images + --method simmim), not a separate binary.
genml-kit-train --pipeline images --method simmim \
                --model convvit \
                --datasets "imagefolder/raw-photos" "imagefolder/more-photos" \
                --image_size 448 \
                --batch_size 32 \
                --epochs 100 \
                --lr 1.5e-4 \
                --amp_dtype bfloat16 \
                --checkpoint ./checkpoints/convvit_simmim

# Step 2: Fine-tune on your labeled dataset
genml-kit-train --model convvit \
                --dataset my-org/labeled-photos \
                --label_column category \
                --source_checkpoint ./checkpoints/convvit_simmim_latest.pt \
                --epochs 100 \
                --lr 3e-5 \
                --batch_size 32 \
                --amp_dtype bfloat16
```
````

**A.1.3 Pre-Training: Supervised Contrastive → [03]**

````markdown
```bash
genml-kit-train --pipeline images --method supcon \
                --model convvit \
                --datasets my-org/labeled-photos \
                --label_column category \
                --image_size 448 \
                --batch_size 64 \
                --samples_per_class 16 \
                --proj_dim 128 \
                --temperature 0.07 \
                --epochs 100 \
                --lr 1e-4 \
                --amp_dtype bfloat16 \
                --checkpoint ./checkpoints/convvit_supcon

# Then fine-tune as above with --source_checkpoint ./checkpoints/convvit_supcon_latest.pt
```
````

**A.1.4 Fine-Tuning: Basic → [04]**

````markdown
```bash
genml-kit-train --model google/vit-base-patch16-224 \
                --dataset my-org/my-labeled-images \
                --label_column category \
                --epochs 5 \
                --batch_size 32 \
                --lr 3e-5 \
                --image_size 224
```
````

**A.1.5 Fine-Tuning: Custom Classifier Head → [05]**

````markdown
```bash
# Freeze backbone, train only the custom head
genml-kit-train --model cls_model_wrapper:google/vit-base-patch16-224 \
                --dataset my-org/my-labeled-images \
                --label_column category \
                --model_arg classifier=mlp:hidden=512,dropout=0.3 \
                --freeze ".*\.(head|pool)"
```
````

**A.1.6 Fine-Tuning: LoRA → [06]**

````markdown
```bash
genml-kit-train \
    --model cls_model_wrapper:facebook/dinov2-with-registers-large \
    --lora --lora_r 16 --lora_alpha 32 \
    --lora_target_modules "query,key,value" \
    --freeze "classifier\.(head|pool|encoder)" \
    --lr 3e-5 \
    --dataset my-org/my-labeled-images \
    --label_column category \
    --epochs 20
```
````

**A.1.7 Fine-Tuning: From A Pre-Trained Checkpoint → [07]**

````markdown
```bash
genml-kit-train --model convvit \
                --dataset my-org/my-labeled-images \
                --label_column category \
                --source_checkpoint ./checkpoints/convvit_simmim_latest.pt \
                --epochs 100
```
````

**A.1.8 `--seed 42 ...` → [08] (UNMARKED)**

````markdown
```bash
genml-kit-train --model convvit --seed 42 ...
```
````

**A.1.9 Cross-Dataset Resume → [09]**

````markdown
```bash
genml-kit-train --model facebook/convnextv2-base-22k-224 \
                --dataset my-org/other-labeled-images \
                --label_column category \
                --checkpoint genml_kit \
                --state_load none \
                --epochs 10 \
                --batch_size 16 \
                --lr 3e-5 \
                --mixup_alpha 0.2 \
                --amp_dtype bfloat16
```
````

**A.1.10 Gradient Checkpointing → [10]**

````markdown
```bash
genml-kit-train \
    --model timm:eva02_base_patch14_448.mim_in22k_ft_in22k_in1k \
    --batch_size 32 \
    --grad_accum_steps 2 \
    --grad_checkpoint \
    --amp_dtype bfloat16 \
    ...
```
````

**A.1.11 LoRA + Custom Classifier → [11]**

````markdown
```bash
genml-kit-train \
    --model cls_model_wrapper:facebook/dinov2-with-registers-large \
    --model_arg classifier=cls_attention:num_encoder_layers=2 \
    --lora --lora_r 16 --lora_alpha 32 \
    --freeze 'classifier\.(head|pool|encoder)' \
    --lr_group 'backbone.*=1e-5' 'classifier.*=3e-4' \
    ...
```
````

**A.1.12 Checkpoint prefix + param rename → [12]**

```bash
genml-kit-train \
    --checkpoint /content/eva02_finetune \
    --source_checkpoint /content/eva02_supcon_latest.pt \
    --param_rename 'encoder\\.model\\.(.*);model.$1' \
    ...
```

**A.1.13 Custom Loss → [13]**

````markdown
```bash
genml-kit-train --model google/vit-base-patch16-224 \
  --loss ./my_loss.py --loss_args '{"alpha": 2.0}' ...
```
````

**A.1.14 RL Loss → [15]**

````markdown
```bash
genml-kit-train --method ppo --loss ./my_ppo_entropy.py ...
```
````

**A.1.15 Custom Models → [14]**

````markdown
```bash
genml-kit-train --model ./my_model.py --loss focal ...
```
````

**A.1.16 Detection: tiny-coco note (two chunks) → [16]**

````markdown
```bash
python scripts/make_tiny_coco.py tiny_coco -n 200 --num_classes 3
genml-kit-train --pipeline detection --method detection \
  --dataset detfolder/tiny_coco --model det_wrapper:timm/resnet18 ...
```
````

**A.1.17 Detection: Example → [17]**

````markdown
```bash
genml-kit-train \
  --pipeline detection --method detection \
  --dataset detfolder//data/coco_tiny \
  --model det_wrapper:timm/resnet50 \
  --model_arg head=fcos_head:num_convs=2 \
  --image_size 640 --batch_size 4 \
  --epochs 20 --lr 1e-4
```
````

**A.1.18 Segmentation: Quick start → [18] (TYPO)**

````markdown
```bash
genml_kit-train \
  --pipeline segmentation --method segmentation \
  --dataset segfolder:/data/voc2012 \
  --model seg_wrapper:timm/resnet34 \
  --model_arg decoder=unet:depth=4 \
  --loss dice_ce \
  --image_size 512 --batch_size 8 --epochs 40
```
````

**A.1.19 Segmentation: HF `--mask_column` → [19] (TYPO)**

````markdown
```bash
genml_kit-train --pipeline segmentation --method segmentation \
  --dataset scene_parse_150 --mask_column annotation \
  --model seg_wrapper:timm/convnext_tiny --model_arg decoder=fpn
```
````

**A.1.20 VO: Quick start (two commands) → [27]**

````markdown
```bash
# Supervised VO on synthetic pairs (default npu-small profile)
genml-kit-train \
  --pipeline vo_pair --method vo_pair \
  --model vo/npu-small \
  --image_size 64 --batch_size 32 --epochs 10

# Photometric self-supervised stage (adds the photometric residual term)
genml-kit-train \
  --pipeline vo_pair --method vo_pair \
  --vo_stage photometric --epochs 10
```
````

**A.1.21 Inference Guide → [32]**

````markdown
```bash
genml-kit-infer --model facebook/convnextv2-base-22k-224 \
                --checkpoint genml_kit_best.pt \
                path/to/image.jpg path/to/other_image.png
```
````

**A.1.22 causal LM: Quick start → [20]**

````markdown
```bash
# Fine-tune a tiny GPT-2 on a local text corpus
genml-kit-train --pipeline text \
                --method causal_lm \
                --model sshleifer/tiny-gpt2 \
                --dataset ./my_text_corpus \
                --text_column text \
                --max_seq_len 256 \
                --lora
```
````

**A.1.23 causal LM: LoRA + grad-ckpt + bf16 → [21]**

````markdown
```bash
# LoRA + gradient checkpointing + bf16 on a small LM
genml-kit-train --pipeline text \
                --method causal_lm \
                --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
                --dataset ./my_text_corpus \
                --text_column text \
                --max_seq_len 1024 \
                --lora --lora_r 16 --lora_alpha 32 \
                --grad_checkpoint \
                --amp_dtype bf16
```
````

**A.1.24 `genml-kit-llm generate` (two commands) → [22]**

````markdown
```bash
# Prompts from a file (one per line, or JSONL with --prompt_column)
genml-kit-llm generate --model sshleifer/tiny-gpt2 \
                       --prompts prompts.txt \
                       --max_new_tokens 64 \
                       --do_sample --temperature 0.8 --top_p 0.95 \
                       --output completions.jsonl

# Prompts from a HuggingFace dataset column, with a fine-tuned checkpoint
genml-kit-llm generate --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
                       --dataset my-org/prompts --split test \
                       --prompt_column question \
                       --checkpoint ./checkpoints/llm_best.pt \
                       --lora_adapter ./adapters/run1
```
````

**A.1.25 `genml-kit-llm eval` → [23]**

````markdown
```bash
genml-kit-llm eval --model sshleifer/tiny-gpt2 \
                   --dataset ./my_eval_corpus --split test \
                   --text_column text --prompt_column prompt \
                   --reference_column reference \
                   --metrics ppl exact_match token_f1 \
                   --output report.json
```
````

**A.1.26 Preference: SFT → DPO → reward (three commands) → [29]**

````markdown
```bash
# 1. Supervised fine-tune (masking the prompt out of the loss)
genml-kit-train --pipeline text --method causal_lm \
                --model sshleifer/tiny-gpt2 --dataset ./sft_corpus \
                --instruction_column prompt --response_column response \
                --mask_prompt --output_dir ./sft

# 2. DPO against a frozen reference (defaults to --model)
genml-kit-train --pipeline preference --method dpo \
                --model ./sft --reference_checkpoint ./sft/checkpoint_best.pt \
                --dataset ./preferences --chosen_column chosen \
                --rejected_column rejected --prompt_column prompt \
                --mask_prompt --beta 0.1 --lora

# 3. Train a reward model on the same pairs
genml-kit-train --pipeline preference --method reward_model \
                --model ./sft --dataset ./preferences \
                --chosen_column chosen --rejected_column rejected
```
````

**A.1.27 RL Quick start (three commands, UNLABELED fence) → [24]**

````markdown
```
# DQN on CartPole (discrete)
genml-kit-train --pipeline rl --method dqn --env_id CartPole-v1 \
    --epochs 20 --batch_size 64

# PPO on CartPole (discrete)
genml-kit-train --pipeline rl --method ppo --env_id CartPole-v1 \
    --epochs 20 --ppo_rollout_len 2048

# SAC on a continuous env (e.g. LunarLanderContinuous-v3)
genml-kit-train --pipeline rl --method sac --env_id LunarLanderContinuous-v3 \
    --epochs 20
```
````

**A.1.28 RL model registry (four commands, UNLABELED fence) → [25]**

````markdown
```
# DQN: dueling Q-head (also the default for --dqn_dueling)
genml-kit-train --pipeline rl --method dqn --model rl/qnet_dueling

# DQN: configure the MLP width via --model_arg
genml-kit-train --pipeline rl --method dqn \
    --model rl/qnet --model_arg hidden_dims=[512,256]

# PPO: wider actor-critic trunk
genml-kit-train --pipeline rl --method ppo \
    --model rl/actor_critic --model_arg hidden_dims=[64,64]

# SAC: critic head width (the actor defaults to rl/actor_critic)
genml-kit-train --pipeline rl --method sac --env_id Pendulum-v1 \
    --model rl/sac_critic --model_arg hidden_dims=[256,256]
```
````

**A.1.29 RL serious PPO (UNLABELED fence) → [26]**

````markdown
```
genml-kit-train --pipeline rl --method ppo --env_id Pendulum-v1 \
    --ppo_continuous --obs_normalize --reward_normalize --param_init ortho \
    --ppo_target_kl 0.03 --epochs 500
```
````

**A.1.30 Distillation Quick start (two commands) → [30]**

````markdown
```bash
# Self-distillation: teacher is an EMA of the student (no external model)
genml-kit-train \
  --pipeline images --method distillation \
  --model google/vit-base-patch16-224 \
  --dataset cifar10 --label_column label \
  --teacher_kind ema --kd_temperature 4.0 --kd_alpha 1.0 \
  --epochs 10

# External teacher: frozen checkpoint hands over its soft targets
genml-kit-train \
  --pipeline images --method distillation \
  --model google/vit-base-patch16-224 \
  --dataset cifar10 --label_column label \
  --teacher_kind external \
  --teacher_model google/vit-large-patch16-224 \
  --teacher_checkpoint /path/to/teacher.pt \
  --kd_temperature 4.0 --kd_alpha 1.0 --epochs 10
```
````

**A.1.31 Pre-Training: prepared_images SimMIM → [28]**

````markdown
```bash
genml-kit-train --pipeline images --method simmim \
                --datasets ./prepared_images ./other-images \
                --image_size 448 --batch_size 32 ...
```
````

### A.2 `scdiag/README.md`

**A.2.1 Canonical Commands (five commands) → [31]**

````markdown
```bash
# Fine-tune on the skin cancer dataset
genml-kit-train --model google/vit-base-patch16-224 \
                --dataset marmal88/skin_cancer \
                --label_column diagnosis \
                --epochs 5 \
                --batch_size 32 \
                --lr 3e-5 \
                --image_size 448

# Supervised-contrastive pre-training on HAM10000
genml-kit-train --pipeline images --method supcon \
                --model convvit \
                --datasets marmal88/skin_cancer \
                --label_column diagnosis \
                --image_size 448 --batch_size 64 \
                --samples_per_class 16 --proj_dim 128 --temperature 0.07 \
                --epochs 100 --lr 1e-4 --amp_dtype bfloat16 \
                --checkpoint ./checkpoints/convvit_supcon

# SimMIM pre-training on the large extracted corpora
genml-kit-train --pipeline images --method simmim \
                --model convvit \
                --datasets ./derm1m_images ./ham10000_grouped \
                --image_size 448 --batch_size 32 ...

# Fine-tune from the pre-trained encoder
genml-kit-train --model convvit --dataset marmal88/skin_cancer \
                --label_column diagnosis \
                --source_checkpoint ./checkpoints/convvit_simmim_latest.pt \
                --epochs 100 --lr 3e-5 --batch_size 32 --amp_dtype bfloat16

# Inference on single images
genml-kit-infer --model facebook/convnextv2-base-22k-224 \
                --checkpoint genml_kit_best.pt \
                path/to/lesion.jpg
```
````

> The **first** `scdiag/README.md` fence (the `prepare_derm1m.py` /
> `prepare_ham10000.py` block) contains no core CLI command and is therefore
> out of scope automatically — no marker needed.

**A.2.2 Not-a-command fragment (out of scope)**

The root-README block

```bash
--model cls_model_wrapper:<hf_name> --model_arg classifier=mlp:hidden=512,dropout=0.3
```

is a bare fragment (no leading program), not a command.  A marker on it is a
**hard error**; it is not in scope and stays untouched.

---

## Appendix B — Marker quick reference

````markdown
<!-- genml-cli-test
name: <str>              # pytest id (optional)
skip: <reason>           # recognised but not run
network: <bool>          # needs internet, no credentials
hf_token: <bool>         # needs HF_TOKEN (implies network)
timeout: <int seconds>   # default 300
extra: <argv string>     # appended to every command in the fence
override:                # list; "--flag val" replaces, "-.--flag" deletes
  - --dataset {tmp}/img
  - -.--source_checkpoint
extras: <str|list>       # pip extras -> importorskip (lora|llm|rl|timm)
env: { K: "{tmp}/v" }    # env for the run
cwd: <path with {tmp}>   # default {tmp}
-->
````
