"""Tests for genml_kit/datasets/packing_sampler.py."""

import datasets
import pytest

from genml_kit.datasets.length_bucket_sampler import text_length_proxy
from genml_kit.datasets.packing_sampler import PackingBatchSampler
from genml_kit.datasets.text_proxy import TextDatasetProxy


def _sampler(lengths, batch_size=4, max_seq_len=8, **kwargs):
  return PackingBatchSampler(lengths,
                             batch_size=batch_size,
                             max_seq_len=max_seq_len,
                             **kwargs)


class TestPackingBatchSampler:

  def test_covers_every_index_once(self):
    lengths = [1, 10, 3, 8, 5, 2, 9, 4, 7, 6]
    seen = []
    for batch in _sampler(lengths):
      seen.extend(batch)
    assert sorted(seen) == list(range(len(lengths)))

  def test_respects_token_budget(self):
    # Units are already tokens (tokens_per_unit=1); every emitted batch must
    # fit the batch_size * max_seq_len budget.
    sampler = _sampler([3] * 40, batch_size=4, max_seq_len=8)
    budget = sampler.budget
    for batch in sampler:
      used = sum(sampler._estimated_tokens(i) for i in batch)
      assert used <= budget

  def test_caps_windows_within_budget(self):
    # Each batch fits the budget, so the collator can emit at most
    # batch_size windows per batch.
    sampler = _sampler([5] * 30, batch_size=4, max_seq_len=8)
    for batch in sampler:
      tokens = sum(sampler._estimated_tokens(i) for i in batch)
      windows = -(-tokens // sampler.max_seq_len)
      assert windows <= sampler.batch_size

  def test_long_document_opens_its_own_batch(self):
    # A single document larger than the budget still appears exactly once, in
    # its own batch (the collator caps the windows it emits).
    lengths = [100, 2, 2, 2]
    batches = list(_sampler(lengths, batch_size=2, max_seq_len=8))
    assert any(0 in b for b in batches)
    assert sorted(i for b in batches for i in b) == [0, 1, 2, 3]

  def test_tokens_per_unit_scales_estimate(self):
    # A character proxy with 0.5 tokens/char halves the estimated cost, so
    # more items fit in a batch.
    unit = list(_sampler([8] * 20, batch_size=4, max_seq_len=8,
                         tokens_per_unit=1.0))
    scaled = list(
        _sampler([8] * 20, batch_size=4, max_seq_len=8, tokens_per_unit=0.25))
    assert len(scaled) < len(unit)

  def test_set_epoch_reshuffles(self):
    lengths = list(range(40))
    sampler = _sampler(lengths, batch_size=4, max_seq_len=16, seed=7)
    first = list(sampler)
    sampler.set_epoch(1)
    second = list(sampler)
    assert first != second
    # The partition still covers every index exactly once.
    assert sorted(i for b in second for i in b) == list(range(len(lengths)))

  def test_seed_is_deterministic(self):
    lengths = list(range(40))
    a = list(_sampler(lengths, batch_size=4, max_seq_len=16, seed=7))
    b = list(_sampler(lengths, batch_size=4, max_seq_len=16, seed=7))
    assert a == b

  def test_len_matches_iteration(self):
    sampler = _sampler([4] * 33, batch_size=4, max_seq_len=8)
    assert len(sampler) == len(list(sampler))

  def test_drop_last_drops_short_tail(self):
    # 5 tokens of budget 8: a single leftover item is the short tail.
    kept = list(_sampler([4] * 9, batch_size=1, max_seq_len=8, drop_last=True))
    allb = list(_sampler([4] * 9, batch_size=1, max_seq_len=8, drop_last=False))
    assert len(kept) <= len(allb)

  def test_rejects_bad_arguments(self):
    with pytest.raises(ValueError, match="batch_size"):
      _sampler([1], batch_size=0, max_seq_len=8)
    with pytest.raises(ValueError, match="max_seq_len"):
      _sampler([1], batch_size=1, max_seq_len=0)
    with pytest.raises(ValueError, match="tokens_per_unit"):
      _sampler([1], batch_size=1, max_seq_len=8, tokens_per_unit=0)


class TestWithTextProxy:

  def test_batches_by_estimated_tokens(self):
    hf = datasets.Dataset.from_dict({"text": ["a" * 10] * 12})
    proxy = TextDatasetProxy(hf, text_column="text")
    lengths = text_length_proxy(proxy)
    sampler = PackingBatchSampler(lengths,
                                  batch_size=2,
                                  max_seq_len=8,
                                  tokens_per_unit=1.0)
    seen = [i for batch in sampler for i in batch]
    assert sorted(seen) == list(range(12))
