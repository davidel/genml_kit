"""Identity processor for the VO pair models (no normalization stats)."""


class VOProcessor:
  """Exposes ``image_mean``/``image_std`` per the registry protocol.

  VO inputs are consumed as raw grayscale (or RGB) pairs with identical
  statistics for both frames of a pair -- there is nothing sensible to
  normalize away between frames, so the processor is an identity.
  """

  image_mean = [0.0]
  image_std = [1.0]
