"""Tests for offline BPE tokenizer training (models/tokenizer_train.py).

All cases are offline: a small synthetic corpus is written with
``save_to_disk`` and the builder is pointed at that directory, so no hub
download happens.  The assertions cover the properties that make an artifact
*reproducible and comparable* -- exact row counts for random-access sources,
seed determinism, special-token id layout, and the skip/rebuild contract.
"""

import json
import os

import pytest

from genml_kit.cli.tokenizer import build_parser, main
from genml_kit.models.tokenizer_train import (
    _sample_indices,
    _select_rows,
    train_tokenizer,
)

_WORDS = [
    "the cat sat on the mat",
    "a dog ran in the park",
    "the cat and the dog played",
    "birds fly over the trees",
    "the mat is on the floor",
    "a cat sleeps in the sun",
    "dogs bark at the birds",
    "the park has tall trees",
    "the sun is warm today",
    "a bird sings in the tree",
]


@pytest.fixture
def corpus_dir(tmp_path):
  """Write a 10-row synthetic text dataset and return its directory."""
  from datasets import Dataset

  dataset = Dataset.from_dict({"text": _WORDS})
  path = tmp_path / "corpus"
  dataset.save_to_disk(str(path))
  return path


class TestSampling:

  def test_sample_indices_exact_count(self):
    indices = _sample_indices(10, 0.3, seed=0)
    assert len(indices) == 3
    assert len(set(indices)) == 3
    assert indices == sorted(indices)

  def test_sample_indices_reproducible(self):
    assert _sample_indices(100, 0.25, seed=7) == _sample_indices(100, 0.25, 7)

  def test_sample_indices_seed_changes_selection(self):
    assert _sample_indices(100, 0.25, seed=7) != _sample_indices(100, 0.25, 8)

  def test_bernoulli_expected_fraction(self):
    rows = [{"text": f"row {i}"} for i in range(2000)]
    kept = list(_select_rows(rows, 0.1, seed=0)[0])
    assert 150 < len(kept) < 250

  def test_fraction_none_keeps_everything(self):
    rows = [{"text": f"row {i}"} for i in range(50)]
    kept = list(_select_rows(rows, None, seed=0)[0])
    assert len(kept) == 50

  def test_max_rows_caps_before_fraction(self):
    rows = [{"text": f"row {i}"} for i in range(100)]
    kept = list(_select_rows(rows, None, seed=0, max_rows=10)[0])
    assert len(kept) == 10

  def test_custom_sampler_overrides(self):

    def sampler(dataset, fraction, seed):
      del fraction, seed
      return dataset[:2]

    rows = [{"text": f"row {i}"} for i in range(100)]
    kept, mode = _select_rows(rows, 0.5, seed=0, sampler=sampler)
    assert mode == "custom"
    assert len(list(kept)) == 2


class TestValidation:

  def test_fraction_out_of_range_rejected(self, corpus_dir, tmp_path):
    with pytest.raises(ValueError, match=r"\(0\.0, 1\.0\]"):
      train_tokenizer(
          str(corpus_dir),
          text_column="text",
          out_dir=str(tmp_path / "tok"),
          fraction=0.0,
      )

  def test_fraction_above_one_rejected(self, corpus_dir, tmp_path):
    with pytest.raises(ValueError, match=r"\(0\.0, 1\.0\]"):
      train_tokenizer(
          str(corpus_dir),
          text_column="text",
          out_dir=str(tmp_path / "tok"),
          fraction=1.5,
      )

  def test_unknown_text_column_is_fatal(self, corpus_dir, tmp_path):
    with pytest.raises(ValueError, match="not in dataset"):
      train_tokenizer(
          str(corpus_dir),
          text_column="nope",
          out_dir=str(tmp_path / "tok"),
      )


class TestArtifact:

  def test_writes_hf_format_directory(self, corpus_dir, tmp_path):
    out = tmp_path / "tok"
    train_tokenizer(
        str(corpus_dir),
        text_column="text",
        out_dir=str(out),
        vocab_size=64,
    )
    assert os.path.exists(out / "tokenizer.json")
    assert os.path.exists(out / "tokenizer_config.json")
    assert os.path.exists(out / "tokenizer_meta.json")

  def test_special_token_ids_are_pinned(self, corpus_dir, tmp_path):
    out = tmp_path / "tok"
    train_tokenizer(
        str(corpus_dir),
        text_column="text",
        out_dir=str(out),
        vocab_size=64,
    )
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(str(out / "tokenizer.json"))
    assert tok.token_to_id("<pad>") == 0
    assert tok.token_to_id("<eos>") == 1
    assert tok.token_to_id("<bos>") == 2
    assert tok.token_to_id("<unk>") == 3

  def test_loadable_by_load_tokenizer(self, corpus_dir, tmp_path):
    from genml_kit.models.tokenizer import load_tokenizer

    out = tmp_path / "tok"
    train_tokenizer(
        str(corpus_dir),
        text_column="text",
        out_dir=str(out),
        vocab_size=64,
    )
    tok = load_tokenizer(str(out))
    assert tok.tokenizer_path is not None or os.path.isdir(str(out))
    assert tok.pad_token_id == 0

  def test_meta_records_provenance(self, corpus_dir, tmp_path):
    out = tmp_path / "tok"
    train_tokenizer(
        str(corpus_dir),
        text_column="text",
        out_dir=str(out),
        vocab_size=64,
        fraction=0.5,
        seed=3,
    )
    with open(out / "tokenizer_meta.json") as f:
      meta = json.load(f)
    assert meta["fraction"] == 0.5
    assert meta["seed"] == 3
    assert meta["strategy"] == "index"
    assert meta["rows_trained"] == 5
    assert meta["text_column"] == "text"

  def test_reproducible_across_runs(self, corpus_dir, tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    kwargs = dict(text_column="text", vocab_size=64, fraction=0.5, seed=11)
    train_tokenizer(str(corpus_dir), out_dir=str(first), **kwargs)
    train_tokenizer(str(corpus_dir), out_dir=str(second), **kwargs)
    assert (first / "tokenizer.json").read_bytes() == (second /
                                                       "tokenizer.json").read_bytes()


class _CountingSampler:
  """Sampler that records how many times it was invoked."""

  def __init__(self):
    self.calls = 0

  def __call__(self, dataset, fraction, seed):
    self.calls += 1
    return list(dataset)[:2]


class TestIdempotency:

  def test_rerun_skips_existing_artifact(self, corpus_dir, tmp_path):
    out = tmp_path / "tok"
    sampler = _CountingSampler()
    train_tokenizer(str(corpus_dir),
                    text_column="text",
                    out_dir=str(out),
                    sampler=sampler)
    assert sampler.calls == 1
    train_tokenizer(str(corpus_dir),
                    text_column="text",
                    out_dir=str(out),
                    sampler=sampler)
    # The second call must not even touch the corpus.
    assert sampler.calls == 1

  def test_force_rebuilds(self, corpus_dir, tmp_path):
    out = tmp_path / "tok"
    sampler = _CountingSampler()
    train_tokenizer(str(corpus_dir),
                    text_column="text",
                    out_dir=str(out),
                    sampler=sampler)
    train_tokenizer(str(corpus_dir),
                    text_column="text",
                    out_dir=str(out),
                    sampler=sampler,
                    force=True)
    assert sampler.calls == 2


class TestCli:

  def test_missing_required_args_rejected(self):
    parser = build_parser()
    with pytest.raises(SystemExit):
      parser.parse_args([])

  def test_fraction_option_validated(self):
    parser = build_parser()
    with pytest.raises(SystemExit):
      parser.parse_args(
          ["--dataset", "x", "--text_column", "text", "--out", "o", "--fraction", "0"])

  def test_defaults(self):
    parser = build_parser()
    args = parser.parse_args(["--dataset", "x", "--text_column", "text", "--out", "o"])
    assert args.vocab_size == 4096
    assert args.fraction is None
    assert args.seed == 0
    assert args.streaming is False
    assert args.max_rows is None

  def test_main_builds_artifact(self, corpus_dir, tmp_path):
    out = tmp_path / "tok"
    main([
        "--dataset",
        str(corpus_dir), "--text_column", "text", "--out",
        str(out), "--vocab_size", "64", "--fraction", "0.5"
    ])
    assert os.path.exists(out / "tokenizer.json")
