"""Text generation: samplers, the portable decoding loop, and dispatch.

Three layers, deliberately separated:

- **Samplers** (:func:`apply_temperature`, :func:`filter_top_k`,
  :func:`filter_top_p`, :func:`apply_repetition_penalty`,
  :func:`sample_next_token`) are pure functions over a ``[B, V]`` logits
  tensor.  They carry no model or tokenizer state and are unit-tested in
  isolation.
- **The portable loop** (:func:`autoregressive_sample`, :func:`beam_search`)
  decodes token by token using only the architecture-agnostic model
  contract that ``CausalLMMethod.train_step`` already relies on::

      out = model(input_ids=..., attention_mask=...)
      logits = out.logits

  It therefore works for *any* ``torch.nn.Module`` satisfying that
  contract -- a HuggingFace ``PreTrainedModel`` or a registered custom
  model.  The loop is cache-free (it re-runs ``forward`` on the full
  prefix each step); native backends supply their own KV-cache.
- **The dispatcher** (:func:`generate`) picks a backend: a
  :class:`~genml_kit.models.generative.GenerativeLM` contract implementation
  first, then the model's own native ``generate`` (HuggingFace), then the
  portable loop.

HuggingFace models are never wrapped -- they keep their own ``generate``
(and therefore their ``state_dict`` key names, which ``--resume`` depends
on); the dispatcher calls it directly, translating :class:`SamplingConfig`
into the keyword arguments ``transformers`` expects.
"""

import torch

from genml_kit.utils.logging import fatal

# Logit value that a masked-out position receives; softmax over a row that
# contains only this value is uniform, so callers must guarantee at least
# one surviving position (the samplers below always keep one).
_NEG_INF = float("-inf")


class SamplingConfig:
  """Decoding knobs, with the familiar ``transformers`` names.

    A plain container (no dataclass) so it carries no typing annotations and
    can be built incrementally.  ``to_hf_kwargs`` renders it into the
    keyword arguments ``model.generate`` expects, including only the keys
    that are actually meaningful for the selected mode.
    """

  def __init__(self,
               max_new_tokens=64,
               do_sample=False,
               temperature=1.0,
               top_k=0,
               top_p=1.0,
               repetition_penalty=1.0,
               num_beams=1,
               num_return_sequences=1,
               eos_token_id=None,
               pad_token_id=None,
               seed=None):
    self.max_new_tokens = max_new_tokens
    self.do_sample = do_sample
    self.temperature = temperature
    self.top_k = top_k
    self.top_p = top_p
    self.repetition_penalty = repetition_penalty
    self.num_beams = num_beams
    self.num_return_sequences = num_return_sequences
    self.eos_token_id = eos_token_id
    self.pad_token_id = pad_token_id
    self.seed = seed

  def to_hf_kwargs(self):
    """Render into ``model.generate`` keyword arguments.

        Only the keys that matter for the active mode are emitted: sampling
        parameters are omitted under greedy decoding (``transformers`` warns
        otherwise), and ``repetition_penalty`` only when it is not the
        neutral ``1.0``.
        """
    kwargs = {
        "max_new_tokens": self.max_new_tokens,
        "do_sample": self.do_sample,
        "num_beams": self.num_beams,
        "num_return_sequences": self.num_return_sequences,
    }
    if self.do_sample:
      kwargs["temperature"] = self.temperature
      if self.top_k:
        kwargs["top_k"] = self.top_k
      if self.top_p < 1.0:
        kwargs["top_p"] = self.top_p
    if self.repetition_penalty != 1.0:
      kwargs["repetition_penalty"] = self.repetition_penalty
    if self.eos_token_id is not None:
      kwargs["eos_token_id"] = self.eos_token_id
    if self.pad_token_id is not None:
      kwargs["pad_token_id"] = self.pad_token_id
    return kwargs


def apply_temperature(logits, temperature):
  """Scale logits by ``1 / temperature`` (a no-op at ``temperature == 1``)."""
  if temperature is None or temperature <= 0 or temperature == 1.0:
    return logits
  return logits / temperature


def apply_repetition_penalty(logits, generated_ids, penalty):
  """Down-weight tokens already present in ``generated_ids``.

    Mirrors the ``transformers`` formulation: a positive logit is divided by
    *penalty*, a negative one multiplied -- so the token always becomes
    relatively less likely.  ``generated_ids`` is ``[B, T]``; positions with
    the pad id are ignored by the caller (they simply do not appear).
    """
  if penalty is None or penalty == 1.0:
    return logits
  scores = logits.clone()
  vocab = scores.size(-1)
  for row in range(scores.size(0)):
    seen = torch.unique(generated_ids[row])
    seen = seen[(seen >= 0) & (seen < vocab)]
    if seen.numel() == 0:
      continue
    values = scores[row, seen]
    scores[row, seen] = torch.where(values > 0, values / penalty, values * penalty)
  return scores


def filter_top_k(logits, k):
  """Keep only the ``k`` highest logits per row; mask the rest to ``-inf``."""
  if not k or k <= 0 or k >= logits.size(-1):
    return logits
  topk_values = torch.topk(logits, k, dim=-1).values
  threshold = topk_values[..., -1, None]
  return torch.where(logits < threshold, torch.full_like(logits, _NEG_INF), logits)


def filter_top_p(logits, p):
  """Nucleus filter: keep the smallest set whose cumulative mass exceeds ``p``.

    The top token is always kept, so the filtered row is never empty.
    """
  if p is None or p >= 1.0 or p <= 0.0:
    return logits
  sorted_logits, sorted_index = torch.sort(logits, dim=-1, descending=True)
  cumulative = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
  # Drop every token whose *preceding* cumulative mass already exceeds p,
  # i.e. shift the threshold right by one so the boundary token survives.
  remove = (cumulative - torch.softmax(sorted_logits, dim=-1)) > p
  sorted_logits = sorted_logits.masked_fill(remove, _NEG_INF)
  return torch.full_like(logits, _NEG_INF).scatter(-1, sorted_index, sorted_logits)


def sample_next_token(logits, do_sample, temperature, top_k, top_p, generator=None):
  """Pick one token id per row: greedy argmax or a filtered multinomial draw."""
  if not do_sample:
    return torch.argmax(logits, dim=-1)
  logits = apply_temperature(logits, temperature)
  logits = filter_top_k(logits, top_k)
  logits = filter_top_p(logits, top_p)
  probs = torch.softmax(logits, dim=-1)
  return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def native_generate_callable(model):
  """Return *model*'s native ``generate`` if it advertises one, else ``None``.

    "Native" is detected through the HuggingFace ``can_generate`` API, which
    ``PreTrainedModel`` (and PEFT wrappers around one) answer.  This is the
    check :class:`~genml_kit.models.generative.GenerativeLM` uses for its
    default body: a :class:`GenerativeLM` subclass is *not* a
    ``PreTrainedModel``, so it has no ``can_generate`` and therefore never
    matches its own inherited ``generate`` -- which is what avoids infinite
    recursion.
    """
  can_generate = getattr(model, "can_generate", None)
  if not callable(can_generate) or not bool(can_generate()):
    return None
  generate = getattr(model, "generate", None)
  return generate if callable(generate) else None


def has_native_generate(model):
  """Return True if *model* exposes a HuggingFace-style native ``generate``.

    Detected through ``can_generate``; :class:`GenerativeLM` subclasses are
    handled by the dispatcher before this check is reached, and plain
    modules with a custom (non-HF) ``generate`` signature fall through to
    the portable loop rather than being called with HF keyword arguments.
    """
  return native_generate_callable(model) is not None


def native_generate(model, input_ids, attention_mask=None, config=None):
  """Call a model's own ``generate`` with the translated keyword arguments."""
  config = config or SamplingConfig()
  kwargs = config.to_hf_kwargs()
  if config.seed is not None:
    torch.manual_seed(config.seed)
  with torch.no_grad():
    return model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)


def autoregressive_sample(model, input_ids, attention_mask=None, config=None):
  """Portable greedy / sampling loop over ``model(...) -> out.logits``.

    Cache-free: ``forward`` runs on the full prefix each step.  Sequences
    are right-padded with ``pad_token_id`` once they emit EOS.  Returns the
    full sequences (prompt + generated) as ``[B * num_return_sequences, T]``.
    """
  config = config or SamplingConfig()
  if config.num_beams > 1:
    if config.do_sample:
      fatal(
          "num_beams > 1 with do_sample is not supported by the portable "
          "generation loop; use a native model.generate() backend.", ValueError)
    return beam_search(model, input_ids, attention_mask, config)

  model.eval()
  device = input_ids.device
  if attention_mask is None:
    attention_mask = torch.ones_like(input_ids)
  input_ids = input_ids.to(device)
  attention_mask = attention_mask.to(device)

  num_return = max(1, config.num_return_sequences)
  prompt = input_ids.repeat_interleave(num_return, dim=0)
  mask = attention_mask.repeat_interleave(num_return, dim=0)

  eos = config.eos_token_id
  pad = config.pad_token_id
  finished = torch.zeros(prompt.size(0), dtype=torch.bool, device=device)
  generator = None
  if config.seed is not None:
    generator = torch.Generator(device=device).manual_seed(config.seed)

  generated = []
  with torch.no_grad():
    for _ in range(config.max_new_tokens):
      logits = model(input_ids=prompt, attention_mask=mask).logits[:, -1, :]
      if generated and config.repetition_penalty != 1.0:
        logits = apply_repetition_penalty(logits, torch.stack(generated, dim=1),
                                          config.repetition_penalty)
      next_token = sample_next_token(logits, config.do_sample, config.temperature,
                                     config.top_k, config.top_p, generator)
      if eos is not None and pad is not None:
        next_token = next_token.masked_fill(finished, pad)
      prompt = torch.cat([prompt, next_token[:, None]], dim=1)
      mask = torch.cat([mask, torch.ones_like(next_token[:, None])], dim=1)
      generated.append(next_token)
      if eos is not None:
        finished = finished | (next_token == eos)
        if bool(finished.all()):
          break

  if not generated:
    return prompt
  new_tokens = torch.stack(generated, dim=1)
  return torch.cat([prompt[:, :input_ids.size(1)], new_tokens], dim=1)


def beam_search(model, input_ids, attention_mask, config):
  """Greedy beam search (no sampling); returns the best beam per prompt.

    Kept in the portable layer so custom models get beams without a native
    ``generate``.  ``num_return_sequences`` is ignored (the single best beam
    per prompt is returned, shaped ``[B, T]``).
    """
  model.eval()
  device = input_ids.device
  if attention_mask is None:
    attention_mask = torch.ones_like(input_ids)
  num_beams = config.num_beams
  batch_size = input_ids.size(0)
  eos = config.eos_token_id

  prompt = input_ids.repeat_interleave(num_beams, dim=0)
  mask = attention_mask.repeat_interleave(num_beams, dim=0)
  beam_scores = torch.zeros(batch_size, num_beams, device=device)
  beam_scores[:, 1:] = -1e9
  beam_scores = beam_scores.reshape(-1)
  finished = torch.zeros(batch_size, num_beams, dtype=torch.bool, device=device)

  generated = []
  with torch.no_grad():
    for _ in range(config.max_new_tokens):
      logits = model(input_ids=prompt, attention_mask=mask).logits[:, -1, :]
      vocab = logits.size(-1)
      next_scores = torch.log_softmax(logits, dim=-1) + beam_scores[:, None]
      next_scores = next_scores.view(batch_size, num_beams * vocab)
      top_scores, top_index = torch.topk(next_scores, num_beams, dim=-1)
      beam_index = top_index // vocab
      token_index = top_index % vocab

      prompt = prompt.view(batch_size, num_beams, -1)
      mask = mask.view(batch_size, num_beams, -1)
      prompt = prompt.gather(1, beam_index[..., None].expand_as(prompt))
      mask = mask.gather(1, beam_index[..., None].expand_as(mask))
      prompt = prompt.reshape(batch_size * num_beams, -1)
      mask = mask.reshape(batch_size * num_beams, -1)
      beam_scores = top_scores.reshape(-1)

      token_flat = token_index.reshape(-1)
      prompt = torch.cat([prompt, token_flat[:, None]], dim=1)
      mask = torch.cat([mask, torch.ones_like(token_flat[:, None])], dim=1)
      generated.append(token_flat)

      finished = finished.gather(1, beam_index)
      if eos is not None:
        finished = finished | (token_index == eos)
        if bool(finished.all()):
          break

  # ``prompt`` already carries the generated tokens (appended each step), so
  # it is the full sequence set -- just pick the best beam per prompt.
  sequences = prompt.view(batch_size, num_beams, -1)
  best = beam_scores.view(batch_size, num_beams).argmax(dim=-1)
  return sequences[torch.arange(batch_size, device=device), best]


def generate(model, input_ids, attention_mask=None, config=None):
  """Dispatch generation to the right backend for *model*.

    Order: a :class:`~genml_kit.models.generative.GenerativeLM` contract
    implementation, then the model's native ``generate`` (HuggingFace), then
    the portable loop.
    """
  from genml_kit.models.generative import GenerativeLM

  config = config or SamplingConfig()
  if isinstance(model, GenerativeLM):
    return model.generate(input_ids, attention_mask, config)
  if has_native_generate(model):
    return native_generate(model, input_ids, attention_mask, config)
  return autoregressive_sample(model, input_ids, attention_mask, config)


def has_chat_template(tokenizer):
  """Return True if *tokenizer* carries a usable chat template."""
  return bool(getattr(tokenizer, "chat_template", None))


def apply_chat_template(tokenizer, prompts, enabled=True):
  """Wrap each prompt as a single user turn when a template is available.

    Falls back to the raw prompts when templates are disabled, when the
    tokenizer has none, or when it advertises one it cannot actually render
    (e.g. the SentencePiece wrapper).  The returned string still needs
    tokenizing -- the template is applied with ``tokenize=False``.
    """
  if not enabled or not has_chat_template(tokenizer):
    return list(prompts)
  templated = []
  for prompt in prompts:
    conversation = [{"role": "user", "content": prompt}]
    try:
      templated.append(
          tokenizer.apply_chat_template(conversation,
                                        tokenize=False,
                                        add_generation_prompt=True))
    except NotImplementedError:
      return list(prompts)
  return templated


def generate_from_prompts(model,
                          tokenizer,
                          prompts,
                          config=None,
                          device="cpu",
                          batch_size=8,
                          chat_template=True):
  """Generate one completion per prompt (batched, left-padded).

    Args:
        model: a module satisfying the ``.logits`` contract, or a native
            ``generate``-capable model.
        tokenizer: an object exposing ``encode`` / ``decode`` and the pad /
            eos ids (the ``load_tokenizer`` interface).
        prompts: list of prompt strings.
        config: a :class:`SamplingConfig`; defaults to greedy 64 tokens.
        device: torch device string or object.
        batch_size: prompts per forward batch.
        chat_template: apply the tokenizer's chat template when available.

    Returns:
        A list of completion strings, aligned with *prompts*.
    """
  config = config or SamplingConfig()
  if config.pad_token_id is None:
    config.pad_token_id = tokenizer.pad_token_id
  if config.eos_token_id is None:
    config.eos_token_id = tokenizer.eos_token_id
  pad_id = config.pad_token_id if config.pad_token_id is not None else 0

  texts = apply_chat_template(tokenizer, prompts, enabled=chat_template)
  device = torch.device(device)
  completions = []
  for start in range(0, len(texts), batch_size):
    chunk = texts[start:start + batch_size]
    encoded = [tokenizer.encode(text, add_special_tokens=True) for text in chunk]
    width = max(len(ids) for ids in encoded)
    input_ids = []
    attention_mask = []
    for ids in encoded:
      pad = width - len(ids)
      input_ids.append([pad_id] * pad + list(ids))
      attention_mask.append([0] * pad + [1] * len(ids))
    input_ids = torch.tensor(input_ids, dtype=torch.long, device=device)
    attention_mask = torch.tensor(attention_mask, dtype=torch.long, device=device)
    output = generate(model, input_ids, attention_mask, config)
    for row in output.tolist():
      completions.append(tokenizer.decode(row[width:], skip_special_tokens=True))
  return completions
