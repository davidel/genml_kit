"""Tests for genml_kit.metrics.confusion_row_strings."""

import numpy as np

from genml_kit.training.metrics import confusion_row_strings


def _cm(rows):
  """Build a confusion matrix from a list of row lists."""
  return np.array(rows, dtype=np.int64)


def test_perfect_classification():
  cm = _cm([[10, 0, 0], [0, 10, 0], [0, 0, 10]])
  lines = confusion_row_strings(cm)
  for line in lines:
    assert "confused with" not in line
    assert line.endswith("100.0%")


def test_binary_confusion():
  cm = _cm([[80, 20], [10, 90]])
  lines = confusion_row_strings(cm, min_prob=0.0)
  assert len(lines) == 2
  # Class 0: recall=80%, confused with Class 1 (20%)
  assert "Class 0: 80.0%" in lines[0]
  assert "Class 1 (20.0%)" in lines[0]
  # Class 1: recall=90%, confused with Class 0 (10%)
  assert "Class 1: 90.0%" in lines[1]
  assert "Class 0 (10.0%)" in lines[1]


def test_three_classes():
  cm = _cm([[100, 10, 5], [15, 80, 5], [0, 10, 90]])
  lines = confusion_row_strings(cm, min_prob=0.0)
  assert len(lines) == 3
  # Class 0: confused with Class 1 (10/115) and Class 2 (5/115).
  assert "confused with" in lines[0]


def test_id2label_used():
  cm = _cm([[80, 20], [10, 90]])
  id2label = {"0": "cat", "1": "dog"}
  lines = confusion_row_strings(cm, id2label=id2label, min_prob=0.0)
  assert lines[0].startswith("cat:")
  assert "dog (20.0%)" in lines[0]
  assert lines[1].startswith("dog:")
  assert "cat (10.0%)" in lines[1]


def test_id2label_integer_keys():
  cm = _cm([[80, 20], [10, 90]])
  id2label = {0: "cat", 1: "dog"}
  lines = confusion_row_strings(cm, id2label=id2label, min_prob=0.0)
  assert lines[0].startswith("cat:")
  assert lines[1].startswith("dog:")


def test_id2label_missing_key_falls_back():
  cm = _cm([[80, 20], [10, 90]])
  lines = confusion_row_strings(cm, id2label={"0": "cat"}, min_prob=0.0)
  # Class 1 has no label -> falls back to "Class 1".
  assert "Class 1 (20.0%)" in lines[0]


def test_top_n_limits_output():
  # 5 classes, Class 0 is confused with all others equally.
  cm = _cm([
      [20, 20, 20, 20, 20],
      [0, 100, 0, 0, 0],
      [0, 0, 100, 0, 0],
      [0, 0, 0, 100, 0],
      [0, 0, 0, 0, 100],
  ])
  lines = confusion_row_strings(cm, top_n=2, min_prob=0.0)
  assert len(lines) == 5
  # Class 0 line: only 2 confused classes shown.
  confused_part = lines[0].split("|")[1]
  assert confused_part.count("(") == 2


def test_top_n_three():
  cm = _cm([[20, 20, 20, 20, 20], [0, 100, 0, 0, 0], [0, 0, 100, 0, 0],
            [0, 0, 0, 100, 0], [0, 0, 0, 0, 100]])
  lines = confusion_row_strings(cm, top_n=3, min_prob=0.0)
  confused_part = lines[0].split("|")[1]
  assert confused_part.count("(") == 3


def test_min_prob_filters_small_confusions():
  # Class 0: 95 correct, 4 confused with Class 1, 1 confused with Class 2.
  cm = _cm([[95, 4, 1], [0, 100, 0], [0, 0, 100]])
  # min_prob=0.05 means >5% — only Class 1 (4/100 = 4%) is below.
  lines = confusion_row_strings(cm, min_prob=0.05)
  assert "confused with" not in lines[0]

  # With min_prob=0.0, both show up.
  lines_all = confusion_row_strings(cm, min_prob=0.0)
  assert "confused with" in lines_all[0]


def test_min_prob_zero_shows_all():
  cm = _cm([[90, 5, 3, 2], [0, 100, 0, 0], [0, 0, 100, 0], [0, 0, 0, 100]])
  lines = confusion_row_strings(cm, min_prob=0.0)
  assert "Class 1 (5.0%)" in lines[0]
  assert "Class 2 (3.0%)" in lines[0]
  assert "Class 3 (2.0%)" in lines[0]


def test_zero_row_skipped():
  cm = _cm([[0, 0], [10, 90]])
  lines = confusion_row_strings(cm)
  assert len(lines) == 1
  assert lines[0].startswith("Class 1:")


def test_all_zero_matrix():
  cm = _cm([[0, 0], [0, 0]])
  lines = confusion_row_strings(cm)
  assert lines == []


def test_single_class():
  cm = _cm([[100]])
  lines = confusion_row_strings(cm)
  assert lines == ["Class 0: 100.0%"]


def test_sorted_by_probability():
  """Confused classes should appear in descending probability order."""
  # Class 0: 50 correct, 30 confused with Class 1, 10 with Class 2, 5 with Class 3.
  cm = _cm([[50, 30, 10, 5], [0, 100, 0, 0], [0, 0, 100, 0], [0, 0, 0, 100]])
  lines = confusion_row_strings(cm, top_n=10, min_prob=0.0)
  # Extract the confused class names in order of appearance.
  part = lines[0].split("|")[1]
  # " confused with: Class 1 (30.0%), Class 2 (10.0%), Class 3 (5.0%)"
  entries = part.strip().removeprefix("confused with: ").split(", ")
  names = [e.rsplit(" (", 1)[0] for e in entries]
  assert names == ["Class 1", "Class 2", "Class 3"]


def test_no_confusions_format():
  cm = _cm([[100, 0], [0, 100]])
  lines = confusion_row_strings(cm)
  for line in lines:
    assert "|" not in line
    assert line.endswith("%")


def test_confusions_format():
  cm = _cm([[80, 20], [10, 90]])
  lines = confusion_row_strings(cm, id2label={"0": "A", "1": "B"})
  # "A: 80.0% | confused with: B (20.0%)"
  assert lines[0] == "A: 80.0% | confused with: B (20.0%)"
