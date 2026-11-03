"""Tests for genml_kit/datasets/length_bucket_sampler.py."""

import datasets

from genml_kit.datasets.length_bucket_sampler import (LengthBucketBatchSampler,
                                                      text_length_proxy)
from genml_kit.datasets.text_proxy import TextDatasetProxy


class TestLengthBucketBatchSampler:

  def test_covers_every_index_once(self):
    lengths = [1, 10, 3, 8, 5, 2, 9, 4]
    sampler = LengthBucketBatchSampler(lengths, batch_size=3)
    seen = sorted(idx for batch in sampler for idx in batch)
    assert seen == list(range(len(lengths)))

  def test_batches_are_length_homogeneous(self):
    lengths = list(range(1000))
    sampler = LengthBucketBatchSampler(lengths, batch_size=4, bucket_size_multiplier=5)
    spreads = []
    for batch in sampler:
      vals = [lengths[i] for i in batch]
      spreads.append(max(vals) - min(vals))
    # Bucketing must make batches far tighter than the global [0, 999] range.
    assert max(spreads) < 100

  def test_len_matches_batch_count(self):
    sampler = LengthBucketBatchSampler([1, 2, 3, 4, 5], batch_size=2)
    assert len(sampler) == 3  # ceil(5 / 2)

  def test_drop_last(self):
    sampler = LengthBucketBatchSampler([1, 2, 3, 4, 5], batch_size=2, drop_last=True)
    assert len(sampler) == 2
    assert all(len(b) == 2 for b in sampler)

  def test_deterministic_with_seed(self):
    lengths = list(range(50))
    a = [b for b in LengthBucketBatchSampler(lengths, 4, seed=7)]
    b = [b for b in LengthBucketBatchSampler(lengths, 4, seed=7)]
    assert a == b


class TestLengthProxy:

  def test_plain_text_column(self):
    hf = datasets.Dataset.from_dict({"text": ["a", "bbb", "cc"]})
    proxy = TextDatasetProxy(hf, text_column="text")
    assert text_length_proxy(proxy) == [1, 3, 2]
