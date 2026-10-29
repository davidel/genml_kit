"""Tests for the preference collator and dataset proxy.

Uses the offline ``byte`` tokenizer so no download is needed; checks the
prompt-masking contract, independent padding of the two members, and the
proxy's column handling.
"""

import pytest

from genml_kit.datasets.preference_proxy import PreferenceDatasetProxy
from genml_kit.models.tokenizer import ByteTokenizer
from genml_kit.training.preference_collator import IGNORE_INDEX, PreferenceCollator


class TestPreferenceCollator:

  def test_batch_shapes(self):
    collator = PreferenceCollator(ByteTokenizer(), max_seq_len=16)
    batch = [{
        "chosen": "good",
        "rejected": "bad"
    }, {
        "chosen": "better",
        "rejected": "worse"
    }]
    blob = collator(batch)
    assert blob.data["chosen_input_ids"].shape[0] == 2
    assert blob.data["rejected_input_ids"].shape[0] == 2
    assert blob.meta["chosen_labels"].shape == blob.data["chosen_input_ids"].shape

  def test_prompt_masked_out_of_labels(self):
    collator = PreferenceCollator(ByteTokenizer(), max_seq_len=32, mask_prompt=True)
    blob = collator([{"prompt": "Q:", "chosen": "yes", "rejected": "no"}])
    labels = blob.meta["chosen_labels"][0].tolist()
    # Prompt tokens are 2 chars ("Q:") -> first two labels ignored.
    assert labels[0] == IGNORE_INDEX and labels[1] == IGNORE_INDEX
    assert labels[2] != IGNORE_INDEX

  def test_no_prompt_scores_everything(self):
    collator = PreferenceCollator(ByteTokenizer(), max_seq_len=32, mask_prompt=True)
    blob = collator([{"chosen": "yes", "rejected": "no"}])
    labels = blob.meta["chosen_labels"][0].tolist()
    # Without a prompt, the first real token is scored.
    assert labels[0] != IGNORE_INDEX

  def test_independent_padding(self):
    collator = PreferenceCollator(ByteTokenizer(), max_seq_len=32)
    blob = collator([{"chosen": "a", "rejected": "a much longer response"}])
    # Each member is padded to its own max -> different widths are fine.
    assert blob.data["chosen_input_ids"].shape[1] != blob.data[
        "rejected_input_ids"].shape[1]

  def test_pad_positions_have_zero_mask_and_ignored_labels(self):
    collator = PreferenceCollator(ByteTokenizer(), max_seq_len=32)
    blob = collator([{"chosen": "ab", "rejected": "cdefg"}])
    mask = blob.data["chosen_attention_mask"][0].tolist()
    labels = blob.meta["chosen_labels"][0].tolist()
    pad_start = mask.index(0) if 0 in mask else len(mask)
    assert all(m == 0 for m in mask[pad_start:])
    assert all(v == IGNORE_INDEX for v in labels[pad_start:])

  def test_missing_columns_fail(self):
    collator = PreferenceCollator(ByteTokenizer(), max_seq_len=16)
    with pytest.raises(ValueError):
      collator([{"chosen": "a"}])


class TestPreferenceProxy:

  def _proxy(self, **kwargs):
    import datasets

    hf = datasets.Dataset.from_dict({
        "chosen": ["good", "great"],
        "rejected": ["bad", "awful"],
        "prompt": ["q1", "q2"],
    })
    return PreferenceDatasetProxy(hf, **kwargs)

  def test_len_and_items(self):
    proxy = self._proxy(prompt_column="prompt")
    assert len(proxy) == 2
    item = proxy[0]
    assert item == {"chosen": "good", "rejected": "bad", "prompt": "q1"}

  def test_no_prompt_column(self):
    proxy = self._proxy()
    assert proxy.has_prompt is False
    assert proxy[0] == {"chosen": "good", "rejected": "bad"}

  def test_max_items_caps_len(self):
    import datasets

    from genml_kit.datasets.capped import maybe_cap

    hf = datasets.Dataset.from_dict({
        "chosen": ["good", "great"],
        "rejected": ["bad", "awful"],
        "prompt": ["q1", "q2"],
    })
    proxy = PreferenceDatasetProxy(maybe_cap(hf, 1))
    assert len(proxy) == 1

  def test_missing_column_fails(self):
    import datasets

    hf = datasets.Dataset.from_dict({"chosen": ["a"]})
    with pytest.raises(ValueError):
      PreferenceDatasetProxy(hf, rejected_column="rejected")
