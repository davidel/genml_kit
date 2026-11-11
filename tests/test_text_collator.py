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

  def test_pack_emits_boundary_tensors(self):
    coll = TextCollator(_tok(), max_seq_len=4, packing=True)
    out = coll([{"text": "ab"}, {"text": "cd"}])
    assert {"segment_ids", "position_ids"} <= set(out.data)
    seg, pos = out.data["segment_ids"], out.data["position_ids"]
    assert seg.shape == pos.shape == out.data["input_ids"].shape
    # Padding positions are marked -1 in segment_ids and 0 in position_ids.
    pad = out.data["attention_mask"] == 0
    assert (seg[pad] == -1).all()
    assert (pos[pad] == 0).all()
    # Within a document, position_ids start at 0 and increase by one; a new
    # document (segment change) resets the running position back to 0.
    real = out.data["attention_mask"] == 1
    for row_seg, row_pos, row_real in zip(seg, pos, real):
      rs, rp = row_seg[row_real].tolist(), row_pos[row_real].tolist()
      prev_seg, prev_pos = None, None
      for s, p in zip(rs, rp):
        if s != prev_seg:
          assert p == 0
        else:
          assert p == prev_pos + 1
        prev_seg, prev_pos = s, p

  def test_no_boundary_tensors_when_not_packing(self):
    coll = TextCollator(_tok(), max_seq_len=8)
    out = coll([{"text": "ab"}, {"text": "cd"}])
    assert set(out.data) == {"input_ids", "attention_mask"}

  def test_bfd_uses_no_more_windows_than_wrapped(self):
    items = [{"text": "a" * (2 + i)} for i in range(6)]
    wrapped = TextCollator(_tok(),
                           max_seq_len=8,
                           packing=True,
                           packing_strategy="wrapped")(items)
    bfd = TextCollator(_tok(), max_seq_len=8, packing=True,
                       packing_strategy="bfd")(items)
    assert bfd.data["input_ids"].shape[0] <= wrapped.data["input_ids"].shape[0]

  def test_bfd_never_exceeds_window_with_eos(self):
    # A document of exactly max_seq_len tokens plus its EOS would formerly
    # become a max_seq_len + 1 window; the BFD strategy must cap it.
    coll = TextCollator(_tok(), max_seq_len=8, packing=True, packing_strategy="bfd")
    long_ids = list(range(100, 108))  # exactly max_seq_len tokens
    out = coll([{"input_ids": long_ids}, {"input_ids": [1, 2, 3]}])
    assert out.data["input_ids"].shape[1] <= 8
    # position_ids must stay inside the model's window (no overflow index).
    assert int(out.data["position_ids"].max()) < 8

  def test_max_windows_caps_rows(self):
    # Without a cap, a batch of long documents fans out into many rows; the
    # cap bounds the emitted row count so the packed batch respects the same
    # memory envelope as --batch_size documents.
    items = [{"input_ids": list(range(100, 200))} for _ in range(4)]
    uncapped = TextCollator(_tok(), max_seq_len=8, packing=True)(items)
    capped = TextCollator(_tok(), max_seq_len=8, packing=True,
                          max_windows=2)(items)
    assert uncapped.data["input_ids"].shape[0] > 2
    assert capped.data["input_ids"].shape[0] == 2
    assert capped.data["input_ids"].shape[1] == 8

  def test_max_windows_warns_on_truncation(self, caplog):
    import logging
    items = [{"input_ids": list(range(100, 200))}]
    coll = TextCollator(_tok(), max_seq_len=8, packing=True, max_windows=2)
    with caplog.at_level(logging.WARNING):
      coll(items)
    assert any("truncated" in r.message for r in caplog.records)

  def test_max_windows_keeps_short_documents_intact(self):
    # A batch that already fits the cap is untouched (no silent token loss).
    coll = TextCollator(_tok(), max_seq_len=8, packing=True, max_windows=4)
    out = coll([{"text": "ab"}, {"text": "cd"}])
    assert out.data["input_ids"].shape[0] <= 4
    assert int(out.data["attention_mask"].sum()) == 10  # 2 x (BOS+2+EOS)

  def test_max_windows_rejects_non_positive(self):
    import pytest
    with pytest.raises(ValueError, match="max_windows"):
      TextCollator(_tok(), max_seq_len=8, packing=True, max_windows=0)

  def test_mask_prompt_is_honored_under_packing(self):
    # Regression: packing used to rebuild labels from ids, silently dropping
    # the prompt masking (prompt tokens got trained on).
    coll = TextCollator(_tok(), max_seq_len=16, mask_prompt=True, packing=True)
    out = coll([{"prompt": "abc", "response": "de"}])
    labels = out.meta["labels"][0]
    assert (labels == IGNORE_INDEX).sum() == 3  # the whole 'abc' prompt
    # The prompt 'abc' (3 bytes, no special tokens) is the masked head.  The
    # response gets an EOS ('de' -> d, e, EOS) and packing appends a separator
    # EOS, so four labels are scored and the first three are ignored.
    scored = labels[labels != IGNORE_INDEX]
    assert scored.tolist() == [100, 101, 257, 257]  # 'd', 'e', EOS, separator

  def test_no_cross_document_label_leak(self):
    # The shifted cross-entropy pairs logits[j-1] with labels[j].  A label that
    # opens a new document must not be predicted from the previous document, so
    # it is masked.  Two docs -> the first token of the second run is -100.
    coll = TextCollator(_tok(), max_seq_len=16, packing=True)
    out = coll([{"text": "ab"}, {"text": "cd"}])
    labels = out.meta["labels"][0]
    seg = out.data["segment_ids"][0]
    # Establish the leak condition: a run boundary exists at position 5.
    assert seg[5] != seg[4]
    assert int(labels[5]) == IGNORE_INDEX

  def test_keep_retains_the_boundary_label(self):
    # ``boundary_loss='keep'`` (--sequence_mode open) leaves the label opening
    # a new document intact, so the model is trained on what follows a document.
    coll = TextCollator(_tok(), max_seq_len=16, packing=True, boundary_loss="keep")
    out = coll([{"text": "ab"}, {"text": "cd"}])
    labels = out.meta["labels"][0]
    seg = out.data["segment_ids"][0]
    assert seg[5] != seg[4]  # the same boundary as the masked case
    assert int(labels[5]) != IGNORE_INDEX

  def test_keep_without_packing_is_rejected(self):
    import pytest
    with pytest.raises(ValueError, match="requires packing"):
      TextCollator(_tok(), boundary_loss="keep")

  def test_bad_boundary_loss_is_rejected(self):
    import pytest
    with pytest.raises(ValueError, match="boundary_loss"):
      TextCollator(_tok(), packing=True, boundary_loss="nope")

  def test_bad_strategy_is_rejected(self):
    import pytest
    with pytest.raises(ValueError, match="packing_strategy"):
      TextCollator(_tok(), packing=True, packing_strategy="nope")

  def test_non_positive_max_seq_len_is_rejected(self):
    import pytest
    with pytest.raises(ValueError, match="max_seq_len"):
      TextCollator(_tok(), max_seq_len=0)


class TestErrors:

  def test_missing_fields_is_fatal(self):
    import pytest

    coll = TextCollator(_tok(), max_seq_len=8)
    with pytest.raises(ValueError, match="none of"):
      coll([{"something": "else"}])
