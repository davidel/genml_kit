"""Generation contract for decoder-only language models.

A model opts into generation by subclassing :class:`GenerativeLM`.  The base
class supplies a working ``generate`` so a custom model that already
implements the training contract -- ``forward(input_ids, attention_mask)``
returning an object with ``.logits`` -- gets generation for free.

Routing, in full:

- **HuggingFace models** never subclass :class:`GenerativeLM` (we cannot
  inject a base class without disturbing their ``state_dict`` key names,
  which ``--resume`` depends on).  Their own ``generate`` is called directly
  by :func:`genml_kit.utils.generation.generate`.
- **:class:`GenerativeLM` subclasses** answer ``model.generate(...)``; the
  default body routes to a native ``generate`` when the instance has one
  (e.g. a subclass that also mixes in a native decoder), otherwise it runs
  the portable token loop over ``self.forward``.
- **Plain modules** with no ``generate`` are handled by the dispatcher's
  portable-loop fallback.

The portable loop itself lives in :mod:`genml_kit.utils.generation`; this
module only defines the contract and its default behaviour.
"""

import torch.nn as nn


class GenerativeLM(nn.Module):
  """Base class marking a model as capable of autoregressive decoding.

    Subclasses inherit a working :meth:`generate`.  Override it only to route
    to a faster or native decoder; the default implementation handles the
    common case (a model whose ``forward`` returns ``.logits``).
    """

  def generate(self, input_ids, attention_mask=None, config=None):
    """Decode tokens after *input_ids* and return full sequences.

        Args:
            input_ids: ``[B, T]`` long tensor of prompt token ids.
            attention_mask: ``[B, T]`` mask (``None`` -> all ones).
            config: a :class:`~genml_kit.utils.generation.SamplingConfig`.

        Returns:
            ``[B', T']`` tensor of prompt + generated ids (``B'`` reflects
            ``num_return_sequences``; ``T'`` includes the prompt).
        """
    # Imported lazily: generation imports this module's class for dispatch.
    from genml_kit.utils.generation import (SamplingConfig, autoregressive_sample,
                                            native_generate, native_generate_callable)

    config = config or SamplingConfig()
    native = native_generate_callable(self)
    if native is not None:
      return native_generate(self, input_ids, attention_mask, config)
    return autoregressive_sample(self, input_ids, attention_mask, config)
