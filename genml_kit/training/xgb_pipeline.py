"""XGBoost-on-backbone pipeline.

Extracted from ``train.py`` to reduce module size.  Orchestrates loading
the best checkpoint, extracting backbone features, and training / saving
an XGBoost classifier on top.
"""

import gc
import logging

import torch

from genml_kit.datasets.hf_proxy import HFDatasetProxy


def train_xgboost_on_backbone(train_ds,
                              val_ds,
                              device,
                              num_labels,
                              *,
                              checkpoint_dir,
                              model_spec,
                              cache_dir=None,
                              proc_kwargs=None,
                              image_size,
                              output_path,
                              batch_size=32,
                              use_gpu=False,
                              max_depth=6,
                              n_estimators=200,
                              learning_rate=0.1,
                              subsample=0.8,
                              colsample_bytree=0.8,
                              min_child_weight=1,
                              gamma=0.0,
                              reg_alpha=0.0,
                              random_state=42):
  """Train XGBoost on top of backbone features extracted from a model.

  Loads the best checkpoint found under *checkpoint_dir* into a fresh
  model (backbone features must come from the selected checkpoint, not
  the in-memory fine-tuned model), extracts features from the raw
  train/val datasets with validation transforms, then trains and
  evaluates an XGBoost classifier and saves it to *output_path*.

  Args:
      train_ds: Training HF Dataset (raw, before proxy wrapping).
      val_ds: Validation HF Dataset (raw, before proxy wrapping).
      device: torch device for feature extraction.
      num_labels: Number of output classes (forwarded to model loader).
      checkpoint_dir: Checkpoint path prefix; ``_best.pt``/``_latest.pt``
          suffixes are resolved by :func:`select_best_checkpoint`.
      model_spec: Model spec for :func:`load_model_for_inference` (e.g.
          ``convvit`` or ``cls_model_wrapper:<hf_name>``).
      cache_dir: Optional HF/timm cache directory.
      proc_kwargs: Optional extra processor kwargs (KEY=VALUE pairs).
      image_size: Eval image size used to build the validation transform.
      output_path: Where to save the fitted XGBoost model.
      batch_size: Feature-extraction batch size.
      use_gpu: Train XGBoost on GPU (requires xgboost with CUDA support).
          Defaults to ``False``, matching the ``--xgb_use_gpu`` CLI
          default.
      max_depth: XGBoost max tree depth.
      n_estimators: XGBoost number of trees.
      learning_rate: XGBoost learning rate.
      subsample: XGBoost row sampling ratio.
      colsample_bytree: XGBoost column sampling ratio.
      min_child_weight: XGBoost min child weight.
      gamma: XGBoost min split loss.
      reg_alpha: XGBoost L1 regularization.
      random_state: Seed for XGBoost's row/column sampling.
  """
  from genml_kit.io.checkpointing import select_best_checkpoint
  from genml_kit.training.model_utils import (
      build_val_transform,
      collect_features,
      load_model_for_inference,
  )
  from genml_kit.training.xgb_utils import eval_xgboost, train_xgboost

  logging.info("=" * 60)
  logging.info("XGBoost training on backbone features")
  logging.info("=" * 60)

  # 1. Load the best checkpoint into a fresh model
  ckpt_path = select_best_checkpoint(checkpoint_dir)
  if ckpt_path is not None:
    logging.info(f"Loading checkpoint: {ckpt_path}")
    model_best, xgb_processor = load_model_for_inference(
        model_spec,
        ckpt_path,
        device="cpu",
        cache_dir=cache_dir,
        num_labels=num_labels,
        image_size=image_size,
        proc_kwargs=proc_kwargs,
    )
    model_best = model_best.to(device)

    # 2. Rebuild train and val datasets with val transforms (not train augs)
    val_transform = build_val_transform(xgb_processor, image_size)
    train_proxy = HFDatasetProxy(train_ds, transform=val_transform)
    val_proxy = HFDatasetProxy(val_ds, transform=val_transform)

    # 3. Collect features
    logging.info("Extracting train features...")
    train_features, train_labels = collect_features(model_best,
                                                    train_proxy,
                                                    device,
                                                    batch_size=batch_size)
    logging.info(f"  Train features shape: {train_features.shape}")

    logging.info("Extracting val features...")
    val_features, val_labels = collect_features(model_best,
                                                val_proxy,
                                                device,
                                                batch_size=batch_size)
    logging.info(f"  Val features shape: {val_features.shape}")

    # 4. Free the PyTorch model before XGBoost training
    del model_best
    gc.collect()
    torch.cuda.empty_cache()

    # 5. Train the XGBoost classifier
    xgb_model = train_xgboost(train_features,
                              train_labels,
                              max_depth=max_depth,
                              n_estimators=n_estimators,
                              learning_rate=learning_rate,
                              subsample=subsample,
                              colsample_bytree=colsample_bytree,
                              min_child_weight=min_child_weight,
                              gamma=gamma,
                              reg_alpha=reg_alpha,
                              use_gpu=use_gpu,
                              random_state=random_state)

    # 6. Evaluate on val set
    val_metrics = eval_xgboost(xgb_model,
                               val_features,
                               val_labels,
                               id2label=train_proxy.id2label)
    logging.info(f"XGBoost val accuracy: {val_metrics['accuracy']:.2%}")
    for cls, acc in val_metrics["per_class_accuracy"].items():
      logging.info(f"  {cls}: {acc:.2%}")
    logging.info(f"Classification report:\n"
                 f"{val_metrics['classification_report']}")
    logging.info(f"Confusion matrix:\n{val_metrics['confusion_matrix']}")

    # 7. Save the XGBoost model
    xgb_model.save_model(output_path)
    logging.info(f"XGBoost model saved: {output_path}")
