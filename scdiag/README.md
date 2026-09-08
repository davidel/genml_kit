# scdiag — skin-lesion classification workflow

[![CI](https://github.com/davidel/genml_kit/actions/workflows/ci.yml/badge.svg)](https://github.com/davidel/genml_kit/actions/workflows/ci.yml)

> **Where did the toolkit go?** The Python package formerly named `scdiag` is
> now **[genml_kit](../README.md)** — a general-purpose, domain-agnostic
> toolkit. This folder is no longer a Python package; it keeps only the
> skin-lesion-specific assets: the dataset preparation scripts and the
> dermoscopy recipes below. Everything generic (training, pre-training,
> inference, checkpoints, storage, custom models) is documented in the
> [root README](../README.md).

## Contents

- [Datasets](#datasets)
- [Preparing the dermoscopy corpora](#preparing-the-dermoscopy-corpora)
- [Canonical commands](#canonical-commands)
- [Dermoscopy recipes](#dermoscopy-recipes)

## Datasets

| Dataset | Source | Label column | Notes |
|---|---|---|---|
| `marmal88/skin_cancer` | HuggingFace | `diagnosis` | HAM10000 derivative: 10 015 dermoscopy images, 7 classes |
| HAM10000 | local files / `imagefolder/` tree via `prepare_ham10000.py` | `dx` | 10 015 images with lesion-id-grouped splits |
| ISIC 2019 | ISIC archive via `prepare_isic2019_cls.py` | — | 25 331 images, 8 classes |
| Derm1M | `redlessone/Derm1M` (HuggingFace) | — | ~1M images, shipped inside zip archives |

`genml_kit` no longer hard-codes any of these: pass `--dataset` explicitly,
and use `--label_column` when the label column is not auto-detected (it is
`diagnosis` for `marmal88/skin_cancer`).

## Preparing the dermoscopy corpora

The scripts in [`scripts/`](scripts/) turn the public corpora into
`imagefolder/`-compatible directory trees that
[`genml-kit-pretrain`](../README.md#pre-training-guide) can consume directly:

| Script | Purpose |
|---|---|
| `prepare_ham10000.py` | HAM10000 with lesion-id-grouped train/val/test splits |
| `prepare_isic.py` | ISIC archive images via `isic-cli` |
| `prepare_isic2019_cls.py` | ISIC 2019 challenge classification set |
| `prepare_derm1m.py` | Derm1M: extracts images from the zip archives referenced by its CSV metadata |

```bash
# Derm1M: download + extract zips into a flat ImageFolder tree
python scdiag/scripts/prepare_derm1m.py --output_dir ./derm1m_images --token hf_XXX

# HAM10000 with grouped splits
python scdiag/scripts/prepare_ham10000.py --output_dir ./ham10000_grouped
```

## Canonical commands

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
genml-kit-pretrain --method supcon \
                   --model convvit \
                   --datasets marmal88/skin_cancer \
                   --label_column diagnosis \
                   --image_size 448 \
                   --batch_size 64 \
                   --samples_per_class 16 \
                   --proj_dim 128 \
                   --temperature 0.07 \
                   --epochs 100 \
                   --lr 1e-4 \
                   --amp_dtype bfloat16 \
                   --checkpoint ./checkpoints/convvit_supcon

# SimMIM pre-training on the large extracted corpora
genml-kit-pretrain --method simmim \
                   --model convvit \
                   --datasets ./derm1m_images ./ham10000_grouped \
                   --image_size 448 --batch_size 32 ...

# Fine-tune from the pre-trained encoder
genml-kit-train --model convvit \
                --dataset marmal88/skin_cancer \
                --label_column diagnosis \
                --source_checkpoint ./checkpoints/convvit_simmim_latest.pt \
                --epochs 100 \
                --lr 3e-5 \
                --batch_size 32 \
                --amp_dtype bfloat16

# Inference on single images
genml-kit-infer --model facebook/convnextv2-base-22k-224 \
                --checkpoint genml_kit_best.pt \
                path/to/lesion.jpg
```

## Dermoscopy recipes

Task-specific advice that does not belong in the generic docs:

- **Class imbalance is clinical, not just statistical.** Melanoma is rare but
  the class you must not miss. Combine `--sampler weighted` with explicit
  priorities, e.g. `--class_multipliers "melanoma=4.0,melanocytic_Nevi=0.5"`
  (loss weighting and sampler multipliers are documented in the
  [root README](../README.md#fine-tuning-guide)).
- **SupCon batch shape.** With the 7 skin_cancer classes,
  `--samples_per_class 16 --batch_size 64` gives 4 classes per batch; tune so
  `batch_size` is divisible by `samples_per_class × num_classes`.
- **Augmentations must be dermatologically plausible.** Horizontal flips and
  mild colour jitter are safe; aggressive crops can cut off the lesion or
  remove the very cue that distinguishes classes. Verify visually with the
  `ImageDump` transform (`genml_kit.utils.image_dump`).
- **Checkpoint selection is a medical decision.** Prefer balanced accuracy
  and per-class recall of malignant classes over plain top-1 accuracy; see
  "Evaluation Metrics" in the [root README](../README.md).
- **Pre-training corpus size.** No single dermoscopy dataset is large enough
  for strong SSL pre-training; ensemble several (Derm1M + ISIC + HAM10000)
  with the `--datasets` flag as shown above.

For everything else — method internals (SimMIM / I-JEPA / DINO / BYOL /
SupCon), LLRD, LoRA, mixup, gradient monitoring, remote checkpoints — read
the [root README](../README.md).
