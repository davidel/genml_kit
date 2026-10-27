"""Tokenizer registry - unified loading for HF and custom tokenizers.

Mirrors the model/processor split in ``models/registry.py``: the entry
point ``load_tokenizer(name, ...)`` returns an object exposing
``tokenizer_path``, ``vocab_size``, ``pad_token_id``, ``eos_token_id``,
``bos_token_id``, ``unk_token_id``, ``apply_chat_template``, ``encode``,
``decode`` and ``__call__``.

``TOKENIZERS`` is an *extension point*, not a populated catalogue (exactly
like ``PROCESSORS``, which only registers custom wrappers and otherwise
falls back to ``AutoImageProcessor``).  Normal model IDs -- ``gpt2``,
``sshleifer/tiny-gpt2``, ``meta-llama/*``, ``mistralai/*`` -- resolve
through ``transformers.AutoTokenizer``, which (>= 4.40) already
auto-detects and converts both SentencePiece ``.model`` files and
tiktoken-format ``.model`` files into the Rust ``tokenizers`` backend.

Only tokenizers that ``AutoTokenizer`` cannot load register an entry:

- ``byte`` -- a dependency-free byte tokenizer for offline tests.
- ``spm``  -- a bare SentencePiece ``tokenizer.model`` (no HF
  ``tokenizer_config.json``), lazily importing ``sentencepiece``.

The ``name:spec`` colon syntax matches ``parse_model_name``: for
``spm:/path/to/tokenizer.model`` the registered name is ``spm`` and the
spec after the colon (``ParsedModelName.processor``) is the model path.
"""

import logging

from genml_kit.models.registry import parse_model_name
from genml_kit.utils.logging import fatal
from genml_kit.utils.registry import Registry
from genml_kit.utils.script import is_script_spec, load_extern

TOKENIZERS = Registry("tokenizer")


class HFTokenizer:
  """Thin wrapper around a HuggingFace tokenizer with LLM-safe defaults.

    The wrapper exists so the pipeline never has to special-case a missing
    pad token: GPT-2-style tokenizers ship without one, so we reuse EOS,
    which makes padded batches and attention masks work out of the box.
    """

  def __init__(self, tokenizer, path):
    self._tok = tokenizer
    self.tokenizer_path = path
    if self._tok.pad_token_id is None and self._tok.eos_token_id is not None:
      self._tok.pad_token = self._tok.eos_token

  @classmethod
  def from_pretrained(cls, path, **kwargs):
    from transformers import AutoTokenizer

    return cls(AutoTokenizer.from_pretrained(path, **kwargs), path)

  @property
  def vocab_size(self):
    return self._tok.vocab_size

  @property
  def pad_token_id(self):
    return self._tok.pad_token_id

  @property
  def eos_token_id(self):
    return self._tok.eos_token_id

  @property
  def bos_token_id(self):
    return self._tok.bos_token_id

  @property
  def unk_token_id(self):
    return self._tok.unk_token_id

  def __call__(self, text, **kwargs):
    return self._tok(text, **kwargs)

  def encode(self, text, **kwargs):
    return self._tok.encode(text, **kwargs)

  def decode(self, ids, **kwargs):
    return self._tok.decode(ids, **kwargs)

  def apply_chat_template(self, conversation, **kwargs):
    return self._tok.apply_chat_template(conversation, **kwargs)

  def __len__(self):
    return len(self._tok)


@TOKENIZERS.register("byte")
class ByteTokenizer:
  """Dependency-free byte-level tokenizer for offline tests.

    Bytes ``0..255`` map to token ids ``0..255`` (identity), and the four
    special tokens occupy ids ``256..259`` so they never collide with a
    byte.  Round-trips are exact and fully deterministic, which is what the
    collator / pipeline tests need without a network download.
    """

  NAME = "byte"

  PAD_TOKEN = "<pad>"
  EOS_TOKEN = "<eos>"
  BOS_TOKEN = "<bos>"
  UNK_TOKEN = "<unk>"

  def __init__(self, tokenizer_path="byte", cache_dir=None):
    del cache_dir  # No artifacts to cache for a byte tokenizer.
    self.tokenizer_path = tokenizer_path
    self.pad_token_id = 256
    self.eos_token_id = 257
    self.bos_token_id = 258
    self.unk_token_id = 259
    self.pad_token = self.PAD_TOKEN
    self.eos_token = self.EOS_TOKEN
    self.bos_token = self.BOS_TOKEN
    self.unk_token = self.UNK_TOKEN
    self._special_ids = {
        self.PAD_TOKEN: self.pad_token_id,
        self.EOS_TOKEN: self.eos_token_id,
        self.BOS_TOKEN: self.bos_token_id,
        self.UNK_TOKEN: self.unk_token_id,
    }

  @property
  def vocab_size(self):
    """Byte ids (0..255) plus the four special tokens."""
    return 260

  def encode(self, text, add_special_tokens=False):
    ids = [b for b in text.encode("utf-8")]
    if add_special_tokens:
      ids = [self.bos_token_id] + ids + [self.eos_token_id]
    return ids

  def decode(self, ids, skip_special_tokens=True):
    if skip_special_tokens:
      ids = [i for i in ids if 0 <= int(i) < 256]
    return bytes(int(i) & 0xFF for i in ids).decode("utf-8", errors="replace")

  def __call__(self, text, add_special_tokens=False, **kwargs):
    return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

  def apply_chat_template(self,
                          conversation,
                          tokenize=True,
                          add_generation_prompt=False,
                          **kwargs):
    """Render a chat conversation with a minimal deterministic format.

        Each turn becomes ``<role>\\n<content>\\n``; a trailing
        ``assistant\\n`` is appended when *add_generation_prompt* is set.
        """
    parts = [f"{turn['role']}\n{turn['content']}\n" for turn in conversation]
    if add_generation_prompt:
      parts.append("assistant\n")
    text = "".join(parts)
    return self.encode(text) if tokenize else text

  def __len__(self):
    return self.vocab_size


@TOKENIZERS.register("spm")
class SpmTokenizer:
  """Bare SentencePiece ``tokenizer.model`` loader.

    ``AutoTokenizer`` needs a ``tokenizer_config.json`` next to the model
    file; this entry covers the ``model``-only case by wrapping
    ``sentencepiece.SentencePieceProcessor`` directly.  ``sentencepiece`` is
    imported lazily so it never becomes a hard dependency.
    """

  NAME = "spm"

  def __init__(self, tokenizer_path, cache_dir=None):
    del cache_dir  # SentencePiece loads from a local path only.
    try:
      import sentencepiece as spm
    except ImportError:
      fatal(
          "The 'spm' tokenizer requires the 'sentencepiece' package. "
          "Install it with: pip install genml_kit[llm-spm]", ImportError)
    self.tokenizer_path = tokenizer_path
    self._sp = spm.SentencePieceProcessor()
    try:
      loaded = self._sp.Load(tokenizer_path)
    except RuntimeError as exc:
      # SentencePiece raises RuntimeError when the model file is absent or
      # malformed; surface it as a ValueError like the rest of the harness.
      fatal(f"Failed to load SentencePiece model {tokenizer_path!r}: {exc}", ValueError)
    if not loaded:
      fatal(f"Failed to load SentencePiece model {tokenizer_path!r}.", ValueError)
    self.pad_token_id = self._sp.pad_id()
    self.eos_token_id = self._sp.eos_id()
    self.bos_token_id = self._sp.bos_id()
    self.unk_token_id = self._sp.unk_id()
    if self.pad_token_id < 0:
      # No pad piece in the model; fall back to EOS so padding still works.
      self.pad_token_id = self.eos_token_id
    self.pad_token = self._sp.IdToPiece(self.pad_token_id)
    self.eos_token = self._sp.IdToPiece(self.eos_token_id)
    self.bos_token = self._sp.IdToPiece(self.bos_token_id)
    self.unk_token = self._sp.IdToPiece(self.unk_token_id)

  @property
  def vocab_size(self):
    return self._sp.GetPieceSize()

  def encode(self, text, add_special_tokens=False):
    ids = self._sp.EncodeAsIds(text)
    if add_special_tokens:
      ids = [self.bos_token_id] + ids + [self.eos_token_id]
    return ids

  def decode(self, ids, skip_special_tokens=True):
    if skip_special_tokens:
      specials = {self.pad_token_id, self.bos_token_id, self.eos_token_id}
      ids = [i for i in ids if int(i) not in specials]
    return self._sp.DecodeIds([int(i) for i in ids])

  def __call__(self, text, add_special_tokens=False, **kwargs):
    return {"input_ids": self.encode(text, add_special_tokens=add_special_tokens)}

  def apply_chat_template(self,
                          conversation,
                          tokenize=True,
                          add_generation_prompt=False,
                          **kwargs):
    raise NotImplementedError(
        "The 'spm' tokenizer has no chat template; tokenize the prompt text "
        "directly or use a HuggingFace tokenizer with a chat template.")

  def __len__(self):
    return self.vocab_size


def load_tokenizer(name, *, cache_dir=None, **kwargs):
  """Load a tokenizer by *name* -- custom, script, or HuggingFace.

    Custom tokenizers are dispatched to the factory registered via
    ``TOKENIZERS.register``.  HuggingFace tokenizers are loaded via
    ``AutoTokenizer``.  External scripts: if *name* is a ``.py`` path or
    HTTP(S) URL the script may define ``create_tokenizer(**kwargs)``.

    Args:
        name: Registered custom name (``"byte"``, ``"spm:/path/to/model"``,
            ``"spm"``), a HuggingFace model identifier, or a ``.py``
            path/URL.
        cache_dir: HuggingFace cache directory.

    Returns:
        A tokenizer object exposing the interface described in the module
        docstring.
    """
  if is_script_spec(name):
    create_tokenizer = load_extern(name, "create_tokenizer")
    logging.info("Loading external tokenizer from %s", name)
    return create_tokenizer(**kwargs)

  parsed = parse_model_name(name)
  if TOKENIZERS.contains(parsed.model):
    logging.info("Loading custom tokenizer '%s' from registry.", parsed.model)
    return TOKENIZERS.get(parsed.model)(tokenizer_path=parsed.processor,
                                        cache_dir=cache_dir,
                                        **kwargs)

  logging.info("Loading HuggingFace tokenizer '%s'.", parsed.model)
  return HFTokenizer.from_pretrained(parsed.model, cache_dir=cache_dir, **kwargs)
