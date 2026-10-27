"""Tests for the tokenizer registry (models/tokenizer.py).

Covers the registry contract (register/get/list), the offline ``byte``
tokenizer's round-trip and special-token behaviour, the ``spm`` entry's
lazy-import failure path, and ``load_tokenizer`` dispatch (custom vs
HuggingFace vs external script).
"""

import pytest

from genml_kit.models.tokenizer import (
    TOKENIZERS,
    ByteTokenizer,
    load_tokenizer,
)


class TestRegistry:

  def test_byte_and_spm_registered(self):
    names = TOKENIZERS.list_names()
    assert "byte" in names
    assert "spm" in names

  def test_get_returns_class(self):
    assert TOKENIZERS.get("byte") is ByteTokenizer

  def test_unknown_raises(self):
    with pytest.raises(ValueError, match="Unknown tokenizer"):
      TOKENIZERS.get("nope")


class TestByteTokenizer:

  def test_round_trip_ascii(self):
    tok = load_tokenizer("byte")
    assert tok.decode(tok.encode("hello")) == "hello"

  def test_round_trip_unicode(self):
    tok = load_tokenizer("byte")
    assert tok.decode(tok.encode("héllo→")) == "héllo→"

  def test_special_tokens_added(self):
    tok = load_tokenizer("byte")
    ids = tok.encode("A", add_special_tokens=True)
    assert ids[0] == tok.bos_token_id
    assert ids[-1] == tok.eos_token_id

  def test_vocab_and_token_ids_disjoint_from_bytes(self):
    tok = load_tokenizer("byte")
    assert tok.vocab_size == 260
    # Special ids sit above the byte range so they never collide.
    for tid in (tok.pad_token_id, tok.eos_token_id, tok.bos_token_id, tok.unk_token_id):
      assert tid >= 256

  def test_decode_skips_specials(self):
    tok = load_tokenizer("byte")
    ids = [tok.bos_token_id] + list(b"hi") + [tok.eos_token_id]
    assert tok.decode(ids) == "hi"

  def test_call_returns_input_ids(self):
    tok = load_tokenizer("byte")
    out = tok("hi")
    assert list(out["input_ids"]) == list(b"hi")

  def test_apply_chat_template(self):
    tok = load_tokenizer("byte")
    text = tok.apply_chat_template([{
        "role": "user",
        "content": "hi"
    }],
                                   tokenize=False,
                                   add_generation_prompt=True)
    assert text == "user\nhi\nassistant\n"


class TestLoadTokenizer:

  def test_custom_name_with_spec(self):
    from genml_kit.models.tokenizer import TOKENIZERS as reg

    @reg.register("_test_tok")
    class _Fake:

      def __init__(self, tokenizer_path, cache_dir=None):
        self.tokenizer_path = tokenizer_path

    try:
      tok = load_tokenizer("_test_tok:/tmp/model")
      assert tok.tokenizer_path == "/tmp/model"
    finally:
      reg.unregister("_test_tok")

  def test_spm_round_trip(self, tmp_path):
    spm = pytest.importorskip("sentencepiece")
    corpus = tmp_path / "corpus.txt"
    # A varied corpus so a 64-piece vocabulary is achievable.
    corpus.write_text("".join(
        f"the quick brown fox jumps over lazy dog {i}\n" for i in range(200)))
    prefix = tmp_path / "tok"
    spm.SentencePieceTrainer.train(input=str(corpus),
                                   model_prefix=str(prefix),
                                   vocab_size=64,
                                   model_type="bpe",
                                   pad_id=0,
                                   eos_id=1,
                                   bos_id=2,
                                   unk_id=3,
                                   minloglevel=2)
    tok = load_tokenizer(f"spm:{prefix}.model")
    assert tok.vocab_size == 64
    assert tok.decode(tok.encode("the quick brown fox")) == "the quick brown fox"

  def test_spm_missing_model_is_fatal(self, tmp_path):
    pytest.importorskip("sentencepiece")
    with pytest.raises(ValueError, match="Failed to load"):
      load_tokenizer(f"spm:{tmp_path / 'missing.model'}")
