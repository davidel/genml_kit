"""getitem_retry — robust index loader with random-fallback retry."""

import collections
import logging
import random

# What ``getitem_retry`` returns: the loaded item plus the index it was actually
# loaded from (which differs from the requested index after a random fallback).
# Defined here (not in ``pipelines.contracts``) so this low-level dataset helper
# does not depend on the ``pipelines`` package.
RetryResult = collections.namedtuple("RetryResult", ["item", "index"])


def getitem_retry(idx, fn, size, max_retry=5):
  """Call *fn(idx)*, falling back to a random index on failure.

  Parameters
  ----------
  idx : int
      The requested index.
  fn : callable[[int], Any]
      Loader function that takes an index and returns the item.
  size : int
      Total number of items (used to generate random fallback indices).
  max_retry : int
      Maximum number of retry attempts (default 5).

  Returns
  -------
  RetryResult
      ``(item, index)`` — the loaded item and the index that was
      ultimately used (may differ from *idx* after retries).

  Raises
  ------
  Exception
      Re-raises the last exception if all retries are exhausted.
  """
  for attempt in range(max_retry + 1):
    try:
      return RetryResult(item=fn(idx), index=idx)
    except Exception:
      if attempt < max_retry:
        old_idx = idx
        idx = random.randrange(size)
        logging.warning(
            "getitem_retry: attempt %d/%d failed (idx=%d), retrying with "
            "random idx=%d", attempt + 1, max_retry, old_idx, idx)
      else:
        raise
