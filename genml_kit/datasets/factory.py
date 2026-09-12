"""Dataset-name parsing shared by all data CLIs.

Extracted verbatim from ``pretrain/cli.py::build_pretrain_dataset``
(plan §12.6, item 2): ``pretrain/cli.py`` and the VO data CLI both parse
``imagefolder/<path>`` vs HuggingFace repo names without duplicating the
logic.
"""

import os

from genml_kit.utils.logging import fatal


def parse_dataset_specs(names, image_column="image", label_column=None, split="train"):
  """Parse dataset name strings into ensemble-ready config dicts.

  Args:
      names: Sequence of dataset specs; a spec starting with
          ``imagefolder/`` refers to a local ``ImageFolder`` tree, a spec
          that is an existing directory path is treated the same way,
          anything else is a HuggingFace dataset name.  Blank entries
          are skipped.
      image_column: Image column for HuggingFace sources.
      label_column: Optional label column carried into HuggingFace
          source configs when provided.
      split: Split for HuggingFace sources.

  Returns:
      List of config dicts ready to feed ``DatasetEnsemble``: every
      entry has ``name`` and ``source``; HuggingFace entries also carry
      ``split`` and ``image_column`` (plus ``label_column`` when given).

  Raises:
      ValueError: When no valid spec remains.
  """
  configs = []
  for name in names:
    name = name.strip()
    if not name:
      continue
    if name.startswith("imagefolder/"):
      configs.append({"name": name.split("/", 1)[1], "source": "imagefolder"})
    elif os.path.isdir(name):
      configs.append({"name": name, "source": "imagefolder"})
    else:
      cfg = {
          "name": name,
          "source": "hf",
          "split": split,
          "image_column": image_column,
      }
      if label_column:
        cfg["label_column"] = label_column
      configs.append(cfg)

  if not configs:
    fatal("No datasets specified. Use --datasets <name1> <name2> ...", ValueError)
  return configs
