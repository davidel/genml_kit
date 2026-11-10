"""Resolve ``--sequence_mode`` (and the windowing flags) into one object.

A causal-LM run has to answer three coupled questions that used to be set
independently -- and could therefore drift into combinations that silently
train or query the wrong distribution:

- **windowing** -- how documents fill a ``max_seq_len`` window: no packing,
  ``wrapped`` concatenation, ``bfd`` bin packing, or the ``bucket`` sampler.
- **boundary_loss** -- ``mask`` (document-exact: the label opening each packed
  document is ``IGNORE_INDEX``, so the causal shift never crosses a boundary)
  or ``keep`` (the GPT-2/GPT-3 recipe: the boundary token keeps a trained
  next-token distribution, so "what follows a document" is a learned state).
- **prompt_framing** -- how generation encodes a prompt: ``plain`` (a document
  prefix: ``[BOS] ...``) or ``doc`` (a whole-document state: ``[BOS] ... [EOS]``).

``keep`` and ``doc`` are only coherent together and only under packing: the
token after an ``EOS`` inside a window is the next document's first token, so
the boundary label is meaningful.  Unpacked, that token is padding, so
``keep`` would teach ``EOS -> PAD`` -- hence the resolver forbids it.  Fusing
the three answers into one :class:`SequenceMode` makes the invalid pairs
unrepresentable instead of merely rejected.

The module is deliberately torch-free and side-effect-free so both the
trainer (``pipelines/text.py``) and the generation helper can compile the
*same* object from the *same* strings.
"""

from collections import namedtuple

from genml_kit.utils.logging import fatal

#: The compiled form of the coupling.  ``windowing`` is the *resolved* strategy
#: (``None`` / ``"wrapped"`` / ``"bfd"`` / ``"bucket"``), never a raw flag.
SequenceMode = namedtuple(
    "SequenceMode",
    ["name", "windowing", "boundary_loss", "prompt_framing"],
)

#: The windowing values that mean "documents share a window".
_PACKING = ("wrapped", "bfd")

#: The named recipes.  ``windowing`` is the default when neither ``--packing``
#: nor ``--length_bucket`` overrides it.
MODES = {
    # Document-exact: no packing, document-exact loss, and a prompt framed as a
    # document prefix (no trailing EOS).  The historical ppl recipe, with the
    # generation framing corrected so the first sampled token is trained.
    "document": SequenceMode("document", None, "mask", "plain"),
    # GPT-2/GPT-3 open continuation: packed concatenation, loss kept at the
    # boundary, prompt framed as a whole document (trailing EOS is a trained
    # "a fresh, unrelated document follows" cue).
    "open": SequenceMode("open", "wrapped", "keep", "doc"),
}


def mode_names():
  """The valid ``--sequence_mode`` values, in a stable order."""
  return tuple(MODES)


def resolve_sequence_mode(name, packing=None, length_bucket=False):
  """Compile the mode + windowing flags into one validated :class:`SequenceMode`.

  Args:
      name: the ``--sequence_mode`` value (a key of :data:`MODES`).
      packing: the ``--packing`` value -- ``None`` (unset), ``"wrapped"`` or
          ``"bfd"``.
      length_bucket: the ``--length_bucket`` flag.

  Returns:
      A :class:`SequenceMode` whose ``windowing`` field is the resolved
      strategy and the other two fields the mode's boundary/framing policy.

  Raises:
      ValueError: for an unknown *name*, a *packing* value the mode forbids, or
          a *packing* / *length_bucket* clash (the two are mutually exclusive;
          supplying both is an error).
  """
  if name not in MODES:
    fatal(
        f"unknown --sequence_mode {name!r}; expected one of "
        f"{', '.join(mode_names())}.", ValueError)
  if packing is not None and packing not in _PACKING:
    fatal(f"--packing must be one of {_PACKING}, got {packing!r}.", ValueError)
  if packing is not None and length_bucket:
    fatal("--packing and --length_bucket are mutually exclusive.", ValueError)

  mode = MODES[name]
  # The explicit windowing flags override the mode's default.  ``open`` must
  # pack (its ``keep`` boundary signal only exists under concatenation), so a
  # bucketed run cannot satisfy it.
  if packing is not None:
    windowing = packing
  elif length_bucket:
    windowing = "bucket"
  else:
    windowing = mode.windowing

  if mode.boundary_loss == "keep" and windowing not in _PACKING:
    fatal(
        f"--sequence_mode {name!r} requires packing (--packing wrapped|bfd); "
        f"its boundary loss is only defined for concatenated documents.", ValueError)
  return mode._replace(windowing=windowing)
