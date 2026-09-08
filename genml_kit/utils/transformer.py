"""Transformer encoder construction helpers.

Kept free of ``genml_kit`` imports: classifier modules are imported while
the ``genml_kit.models`` package is still initializing, so anything they
use at module level must not pull in the package itself.
"""

import torch.nn as nn


def build_transformer_encoder(encoder_layer,
                              num_layers,
                              norm=None,
                              enable_nested_tensor=True,
                              init_std=0.02):
  """Build an ``nn.TransformerEncoder`` with per-layer re-initialization.

  ``nn.TransformerEncoder`` clones a single template layer for every
  block, so all blocks start with bit-identical parameters -- a pitfall
  the PyTorch docs call out explicitly.  This helper re-initializes each
  block after construction: truncated-normal weights for attention and
  linear modules, zeros for their biases.  LayerNorm parameters are left
  at the PyTorch defaults.

  Args:
    encoder_layer: Template ``nn.TransformerEncoderLayer`` to clone.
    num_layers: Number of encoder layers to stack.
    norm: Optional final ``nn.Module`` applied after the last layer.
    enable_nested_tensor: Forwarded to ``nn.TransformerEncoder``.  The
      nested-tensor fast path only applies to post-norm stacks used with
      a padding mask; pass ``False`` for pre-norm stacks to silence the
      construction-time ``UserWarning``.
    init_std: Standard deviation of the truncated-normal init.

  Returns:
    An ``nn.TransformerEncoder`` with independently initialized blocks.
  """
  encoder = nn.TransformerEncoder(
      encoder_layer,
      num_layers,
      norm=norm,
      enable_nested_tensor=enable_nested_tensor,
  )
  for layer in encoder.layers:
    for module in layer.modules():
      if isinstance(module, nn.MultiheadAttention):
        # ``out_proj`` is an ``nn.Linear`` subclass and is handled by the
        # branch below; only the fused input projection lives here.
        if module.in_proj_weight is not None:
          nn.init.trunc_normal_(module.in_proj_weight, std=init_std)
          if module.in_proj_bias is not None:
            nn.init.zeros_(module.in_proj_bias)
      elif isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=init_std)
        if module.bias is not None:
          nn.init.zeros_(module.bias)
  return encoder
