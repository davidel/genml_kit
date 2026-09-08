"""Tests for getitem_retry utility."""

import pytest

from genml_kit.datasets.retry import getitem_retry


class TestGetitemRetry:

  def test_returns_on_first_try(self):
    """No retry needed — fn succeeds immediately."""

    def fn(i):
      return f"item_{i}"

    item, idx = getitem_retry(3, fn, size=10)
    assert item == "item_3"
    assert idx == 3

  def test_retries_on_failure(self, monkeypatch):
    """First call fails; retry picks a new random index."""
    call_count = 0

    def fn(i):
      nonlocal call_count
      call_count += 1
      if call_count == 1:
        raise OSError("simulated corruption")
      return f"ok_{i}"

    # Pin the random fallback to index 42 so the assertion is deterministic.
    monkeypatch.setattr("genml_kit.datasets.retry.random.randrange", lambda size: 42)

    item, idx = getitem_retry(0, fn, size=100, max_retry=5)
    assert item.startswith("ok_")
    assert call_count == 2
    assert idx == 42

  def test_all_retries_exhausted(self):
    """Every call fails — exception propagates after max_retry attempts."""

    def fn(i):
      raise OSError("always broken")

    with pytest.raises(OSError, match="always broken"):
      getitem_retry(0, fn, size=10, max_retry=3)

  def test_max_retry_zero_means_no_retry(self):
    """With max_retry=0 the fn is called exactly once."""
    call_count = 0

    def fn(i):
      nonlocal call_count
      call_count += 1
      raise OSError("fail")

    with pytest.raises(OSError):
      getitem_retry(0, fn, size=10, max_retry=0)
    assert call_count == 1

  def test_fallback_index_in_range(self):
    """When a retry is triggered, the new index is within [0, size)."""
    seen_indices = []

    def fn(i):
      seen_indices.append(i)
      if len(seen_indices) == 1:
        raise OSError("fail once")
      return "ok"

    item, idx = getitem_retry(0, fn, size=10, max_retry=5)
    assert item == "ok"
    assert len(seen_indices) == 2
    assert seen_indices[0] == 0
    assert 0 <= seen_indices[1] < 10
    assert idx == seen_indices[1]

  def test_size_one_fallback(self):
    """With size=1 the only possible fallback index is 0."""
    call_count = 0

    def fn(i):
      nonlocal call_count
      call_count += 1
      if call_count <= 2:
        raise OSError("fail")
      return f"ok_{i}"

    item, idx = getitem_retry(0, fn, size=1, max_retry=5)
    assert item == "ok_0"
    assert idx == 0
    assert call_count == 3
