"""Streaming HF proxies: iterate without an index (§15.2, §15.9).

A streaming HF dataset has no ``__len__`` and no indexable rows, so the proxies
must (a) convert rows from an iterator, and (b) report their length as
``TypeError`` so ``num_samples`` falls back to an estimate instead of exploding.
"""

import pytest

from genml_kit.datasets.base import is_random_access, num_samples
from genml_kit.datasets.text_proxy import TextDatasetProxy
from genml_kit.datasets.iterable import is_streaming_source


class _FakeStreamingHF:
  """Stand-in for a ``datasets`` IterableDataset: iterable, indexless."""

  def __init__(self, rows):
    self._rows = rows

  @property
  def column_names(self):
    return list(self._rows[0]) if self._rows else []

  def __iter__(self):
    yield from self._rows

  def __getitem__(self, idx):
    raise TypeError("streaming datasets do not support indexing")


class _IndexableHF:
  """Stand-in for a map-style ``datasets.Dataset``."""

  def __init__(self, rows):
    self._rows = rows

  @property
  def column_names(self):
    return list(self._rows[0]) if self._rows else []

  def __len__(self):
    return len(self._rows)

  def __getitem__(self, idx):
    return self._rows[idx]

  def __iter__(self):
    yield from self._rows


class TestTextProxyIteration:

  def test_iterates_plain_lm_rows(self):
    proxy = TextDatasetProxy(_FakeStreamingHF([{
        "text": "a"
    }, {
        "text": "b"
    }]),
                             text_column="text")
    assert [item["text"] for item in proxy] == ["a", "b"]

  def test_iterates_instruction_rows(self):
    proxy = TextDatasetProxy(_FakeStreamingHF([{
        "p": "x",
        "r": "y"
    }]),
                             prompt_column="p",
                             response_column="r")
    assert list(proxy) == [{"prompt": "x", "response": "y"}]

  def test_getitem_and_iter_agree(self):
    rows = [{"text": "a"}, {"text": "b"}]
    proxy = TextDatasetProxy(_IndexableHF(rows), text_column="text")
    assert list(proxy) == [proxy[i] for i in range(len(proxy))]

  def test_len_raises_for_stream(self):
    proxy = TextDatasetProxy(_FakeStreamingHF([{"text": "a"}]), text_column="text")
    with pytest.raises(TypeError):
      len(proxy)
    assert num_samples(proxy) is None
    assert is_random_access(proxy) is False


class TestProxyEstimateFallthrough:
  """A stream proxy must not discard an estimate its wrapped source carries."""

  def test_forwards_the_wrapped_source_estimate(self):
    # ``--max_items`` wraps a stream in a ``CappedDataset`` that reports the cap
    # as an upper bound; the proxy has to forward it rather than answer ``None``.
    from genml_kit.datasets.capped import maybe_cap

    capped = maybe_cap(_FakeStreamingHF([{"text": "a"}] * 4), 2)
    proxy = TextDatasetProxy(capped, text_column="text")
    assert num_samples(proxy) == 2

  def test_explicit_estimate_wins_over_the_wrapped_source(self):
    from genml_kit.datasets.capped import maybe_cap

    capped = maybe_cap(_FakeStreamingHF([{"text": "a"}] * 4), 2)
    proxy = TextDatasetProxy(capped, text_column="text", num_samples=99)
    assert num_samples(proxy) == 99

  def test_exact_length_still_wins_over_an_estimate(self):
    proxy = TextDatasetProxy(_IndexableHF([{
        "text": "a"
    }] * 3),
                             text_column="text",
                             num_samples=99)
    assert num_samples(proxy) == 3


class TestStreamingPredicate:

  def test_fake_stream_is_recognised_by_protocol_not_type(self):
    # ``is_streaming_source`` is an ``isinstance`` check against torch's
    # IterableDataset, so a duck-typed stand-in is *not* one; the proxies
    # therefore rely on ``len`` raising rather than on the predicate alone.
    assert is_streaming_source(_FakeStreamingHF([])) is False

  def test_real_torch_streaming_source_recognised(self):
    from tests.iterable_streams import CountingStream
    assert is_streaming_source(CountingStream(3)) is True
