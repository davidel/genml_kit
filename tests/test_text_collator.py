"""Tests for the text collator (training/text_collator.py).

Uses the offline ``byte`` tokenizer so the tests need no network.  Covers
padding/masking, prompt masking, truncation, label shifting inputs, packing,
and the DataBlob contract.
"""

from genml_kit.models.tokenizer import load_tokenizer
from genml_kit.pipelines.contracts import DataBlob
from genml_kit.training.text_collator import IGNORE_INDEX, TextCollator


def _tok():
  return load_tokenizer("byte")


class TestPlainLM:

  def test_returns_datablob(self):
    coll = TextCollator(_tok(), max_seq_len=16)
    out = coll([{"text": "hi"}, {"text": "bye"}])
    assert isinstance(out, DataBlob)
    assert set(out.data) == {"input_ids", "attention_mask"}
    assert "labels" in out.meta

  def test_shapes_and_padding(self):
    coll = TextCollator(_tok(), max_seq_len=16)
    out = coll([{"text": "hi"}, {"text": "longer"}])
    assert out.data["input_ids"].shape == (2, 8)
    # First row: bos + "hi" + eos = 4 real tokens, rest padded.
    assert out.data["attention_mask"][0].sum().item() == 4
    assert out.meta["labels"][0].tolist()[-1] == IGNORE_INDEX

  def test_truncation(self):
    coll = TextCollator(_tok(), max_seq_len=4)
    out = coll([{"text": "abcdefgh"}])
    assert out.data["input_ids"].shape == (1, 4)

  def test_pad_to_multiple_of(self):
    coll = TextCollator(_tok(), max_seq_len=16, pad_to_multiple_of=8)
    out = coll([{"text": "hi"}])
    assert out.data["input_ids"].shape[1] % 8 == 0


class TestInstruction:

  def test_prompt_not_masked_by_default(self):
    coll = TextCollator(_tok(), max_seq_len=16, mask_prompt=False)
    labels = coll([{"prompt": "Q", "response": "A"}]).meta["labels"][0]
    assert IGNORE_INDEX not in labels.tolist()

  def test_prompt_masked(self):
    coll = TextCollator(_tok(), max_seq_len=16, mask_prompt=True)
    out = coll([{"prompt": "Q", "response": "A"}])
    labels = out.meta["labels"][0].tolist()
    # "Q" is one token; response "A" + eos follow, unmasked.
    assert labels[0] == IGNORE_INDEX
    assert labels[1] != IGNORE_INDEX


class TestPretokenized:

  def test_passthrough(self):
    coll = TextCollator(_tok(), max_seq_len=8)
    out = coll([{"input_ids": [1, 2, 3]}])
    assert out.data["input_ids"][0].tolist()[:3] == [1, 2, 3]


class TestPacking:

  def test_pack_concatenates(self):
    coll = TextCollator(_tok(), max_seq_len=4, packing=True)
    out = coll([{"text": "ab"}, {"text": "cd"}, {"text": "ef"}])
    assert out.data["input_ids"].shape[1] == 4
    # Each item is BOS+2 bytes+EOS plus one EOS separator = 5 tokens, so
    # three items (15 tokens) fill three windows and leave 3 in the last.
    full = out.data["attention_mask"].sum(dim=1)
    assert full.tolist() == [4, 4, 4, 3]


class TestErrors:

  def test_missing_fields_is_fatal(self):
    import pytest

    coll = TextCollator(_tok(), max_seq_len=8)
    with pytest.raises(ValueError, match="none of"):
      coll([{"something": "else"}])
