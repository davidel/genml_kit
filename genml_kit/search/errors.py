"""Exception types for the hyperparameter-search subsystem."""


class SearchError(Exception):
  """Base class for all search-subsystem errors."""


class SearchSpaceExhausted(SearchError):
  """Raised by finite samplers (grid) when no fresh configs remain."""


class TrialFailed(SearchError):
  """Raised when a trial subprocess exits non-zero after all retries."""


class StudyNotFound(SearchError):
  """Raised when a resume/report directory holds no study ledger."""
