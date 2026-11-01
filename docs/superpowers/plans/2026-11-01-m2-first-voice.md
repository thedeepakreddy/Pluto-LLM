# M2 First Voice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Pluto speaks: a ~30M speech+text model (brain + voice box) trained on self-made TinyStories-Speech, with time-aligned text, KV-cached generation, Whisper-scored evaluation and a demo.

**Architecture:** External audio models (Kokoro TTS, Mimi codec, Whisper ASR) sit behind three small interfaces with in-repo fakes, so every pipeline stage is testable on CPU. A resumable builder turns M1's cleaned TinyStories splits into versioned speech shards (text ids, per-token start frames, Mimi codes). Sequence builders produce aligned or baseline layouts; `SpeechGPT` reuses M1's Transformer block for the brain and adds a depth Transformer (voice box). The M1 training loop is refactored into a shared base so the speech trainer inherits resume, Hub sync, NaN guards and cost logging.

**Tech Stack:** Python ≥3.10, PyTorch ≥2.2, numpy; optional extra `speech`: `kokoro>=0.9`, `misaki[en]`, `transformers>=4.45`, `soundfile`, `gradio`.

**Spec:** `docs/superpowers/specs/2026-11-01-m2-first-voice-design.md` (decisions in `docs/01-decisions.md`, M1 code in `src/llm`).

## Global Constraints

- All M1 tests stay green; M1 configs remain valid (new config sections are optional, default `None`).
- No `kokoro`, `transformers`, `soundfile` or `gradio` import at module top level outside `src/llm/speech/adapters.py` and `src/llm/speech/app.py`; CI never downloads models.
- Mimi: `sample_rate = 24000`, `frame_rate = 12.5` (1,920 samples per frame), `codebooks = 8`, `codebook_size = 2048`. Code index `2048` is `EMPTY` (no audio at this position/codebook).
- Brain: 8 layers, d_model 384, 6 heads, context 640. Voice box: 3 layers, d_model 192, 8 heads, 8 positions. Reported total 25–31M parameters.
- Text vocab for the speech model = M1 tokenizer vocab `V` (4,096) + 4 specials: `BOS=V`, `AUDIO=V+1`, `END_AUDIO=V+2`, `PAD_TEXT=V+3`.
- Acoustic delay default `1`; code loss weights default `[3, 1, 1, 1, 1, 1, 1, 1]`; text-only batch every 10th step (`text_every = 10`).
- Voice default `bf_emma`; chunk limit 30 s; pilot `hours = 200`; QA fraction 0.02, QA max WER 0.3.
- Frozen demo set version `speech-demo-v1` (10 sentences); held-out eval = first 200 test chunks.
- Creator identity: raw PCID never committed to the repo (local path or private HF dataset); tests use a fictional fixture only; aspirations never phrased as achievements; `current_location` excluded by default.
- Checkpoints load with `weights_only=True`; manifests use the deterministic checksum from M1; no session links in commits.

## Review Focus

1. **TTS text normalisation mismatch** (e.g. "Mr.", numbers, curly quotes): Kokoro's word list doesn't match the chunk text → chunk dropped and counted as `timing_mismatch`, never a crash. Test `test_build_counts_timing_mismatch` (Task 4).
2. **Session killed mid-append** while building speech data → resume truncates a partial codes/index record and continues with no duplicate or missing ids. Test `test_build_resume_after_partial_write` (Task 4).
3. **Model never emits END_AUDIO** → generation stops at `max_frames`, returns codes, and evaluation reports a duration ratio > 1.5 rather than hanging. Test `test_speak_stops_at_max_frames` (Task 9).
4. **Prompt audio longer than the context** in continue-speech → keeps the most recent frames that fit, no error. Test `test_continue_crops_long_prompt` (Task 9).
5. **Empty or whitespace text to `speak`** → `ValueError("text is empty")` before any model work. Test `test_speak_rejects_empty_text` (Task 9).

---

## File Structure

```
src/llm/config.py                 + SpeechConfig, AudioConfig (optional Config sections)
src/llm/model/gpt.py              + KV cache: CausalSelfAttention/Block.forward_cached
src/llm/model/voice.py            SpeechGPT (brain), DepthTransformer (voice box), SpeechOutput
src/llm/speech/__init__.py
src/llm/speech/interfaces.py      WordTiming, Synthesis, Transcript, TTS/Codec/ASR protocols, make_*()
src/llm/speech/fakes.py           FakeTTS, FakeCodec, FakeASR (deterministic test doubles)
src/llm/speech/adapters.py        KokoroTTS, MimiCodec, WhisperASR (real models, lazy imports)
src/llm/speech/wer.py             normalize_words, word_error_rate, char_error_rate
src/llm/speech/chunk.py           split_sentences, chunk_story, token_start_frames
src/llm/speech/build.py           build_speech_data (resumable), speech_dir, SpeechStats
src/llm/speech/layout.py          Example, build_aligned, build_text_then_audio, build_text_only, extract_*
src/llm/speech/dataset.py         SpeechShards, SpeechBatch, collate, next_speech_batch
src/llm/speech/generate.py        speak, continue_speech (KV-cached, forced-choice text)
src/llm/speech/evaluate.py        DEMO_SENTENCES, evaluate_speech, ceiling_wer
src/llm/speech/app.py             Gradio demo (lazy import)
src/llm/identity/pcid.py          creator identity: load PCID, statements, probes, prepare_identity
src/llm/train/loop.py             refactor: BaseTrainer + TextTrainer (behaviour unchanged)
src/llm/train/speech_loop.py      SpeechTrainer, warm_start_from_text
src/llm/report.py                 + speech section
src/llm/cli.py                    + build-speech, speech-check, eval-speech, speak, continue-speech, demo
configs/m2-stage0.yaml  configs/m2-warmstart.yaml  configs/m2-speech-aligned.yaml  configs/m2-speech-baseline.yaml
notebooks/m2_kaggle.ipynb
tests/test_speech_*.py, tests/test_voice_model.py, tests/test_kv_cache.py, tests/test_speech_train.py
```

---

### Task 1: Config sections for speech data and the audio model

**Files:** Modify `src/llm/config.py`, `pyproject.toml`; Test `tests/test_speech_config.py`

**Interfaces:**
- Produces:
  - `SpeechConfig(name: str, version: int, voice: str = "bf_emma", max_seconds: float = 30.0, hours: float = 200.0, val_chunks: int = 100, test_chunks: int = 200, qa_fraction: float = 0.02, qa_max_wer: float = 0.3, tts: str = "kokoro", codec: str = "mimi", asr: str = "whisper", asr_model: str = "openai/whisper-small.en", keep_wavs: int = 20, root: str = "speech")`
  - `AudioConfig(layout: Literal["aligned", "text_then_audio"] = "aligned", codebooks: int = 8, codebook_size: int = 2048, acoustic_delay: int = 1, depth_layers: int = 3, depth_d_model: int = 192, depth_n_head: int = 8, code_loss_weights: tuple[float, ...] = (3,1,1,1,1,1,1,1), text_loss_weight: float = 1.0, text_every: int = 10, init_from: str | None = None)`
  - `Config` gains `speech: SpeechConfig | None = None`, `audio: AudioConfig | None = None`.
  - `pyproject.toml` extra `speech = ["kokoro>=0.9", "misaki[en]", "transformers>=4.45", "soundfile", "gradio"]`.

- [ ] **Step 1: Write failing tests:** `test_m1_config_still_loads` (tiny config has `speech is None and audio is None`); `test_audio_defaults` (add `audio: {}` → `layout == "aligned"`, `acoustic_delay == 1`, `code_loss_weights == (3,1,1,1,1,1,1,1)`); `test_loss_weights_length_must_match_codebooks` (`audio.codebooks=4` with default weights → `ConfigError` matching `code_loss_weights`); `test_depth_heads_divide` (`audio.depth_n_head=5` → `ConfigError`); `test_audio_requires_speech` (`audio` set, `speech` missing → `ConfigError` matching `speech`); `test_speech_context_fits_30s` (`audio` + `speech` with `model.context_length=300` and `max_seconds=30` → `ConfigError`, since 30 s × 12.5 + delay + 2 = 378 > 300).
- [ ] **Step 2: Run** `pytest tests/test_speech_config.py -v` → FAIL.
- [ ] **Step 3: Implement** the dataclasses and validation in `_validate`.
- [ ] **Step 4: Run** → PASS; `pytest -q -m "not slow"` → all green.
- [ ] **Step 5: Commit** `feat(m2): speech and audio config sections`.

---

### Task 2: Speech interfaces, fakes, WER

**Files:** Create `src/llm/speech/{__init__,interfaces,fakes,wer}.py`; Test `tests/test_speech_basics.py`

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) WordTiming(word: str, start_s: float, end_s: float)`
  - `@dataclass Synthesis(audio: np.ndarray, sample_rate: int, words: list[WordTiming] | None)` (mono float32)
  - `@dataclass Transcript(text: str, words: list[WordTiming])`
  - Protocols: `TTS.synthesize(text: str) -> Synthesis`; `Codec` with attributes `sample_rate: int`, `frame_rate: float`, `codebooks: int`, methods `encode(audio: np.ndarray, sample_rate: int) -> np.ndarray[int64, (frames, K)]`, `decode(codes: np.ndarray) -> np.ndarray`; `ASR.transcribe(audio: np.ndarray, sample_rate: int) -> Transcript`.
  - `make_tts(cfg: SpeechConfig) -> TTS`, `make_codec(cfg: SpeechConfig) -> Codec`, `make_asr(cfg: SpeechConfig) -> ASR`: name `"fake"` → fakes; real names → adapters (Task 14) imported lazily.
  - Fakes (deterministic, CPU, no downloads): `FakeTTS` makes 0.4 s of a word-specific tone per word, sample rate 24000, exact `WordTiming`s; it also registers `hash(audio bytes) → text` in a module-level dict so `FakeASR` returns the exact text for audio it produced and `""` for unknown audio. `FakeCodec`: 24 kHz, 12.5 fps, K=8; `encode` maps each 1,920-sample frame to codes `(int(|mean| * 1e6) + k) % 2048`; `decode` returns `frames * 1920` samples, where each frame is a constant equal to the frame's first code / 1e6 (so FakeASR sees "unknown" audio from models).
  - `normalize_words(text: str) -> list[str]` (lowercase, strip punctuation except apostrophes, split), `word_error_rate(ref: str, hyp: str) -> float` (word-level Levenshtein / len(ref words); empty ref → 0.0 if hyp empty else 1.0), `char_error_rate(ref: str, hyp: str) -> float`.

- [ ] **Step 1: Write failing tests:** `test_wer_known_values` (`("the cat sat", "the cat sat") → 0.0`; `("the cat sat", "the cat") → 1/3`; `("the cat sat", "a cat sat on") → 2/3`; `("Hello, World!", "hello world") → 0.0`); `test_cer_known_values`; `test_fake_tts_timings` (`"Lily went home"` → 3 words, starts `[0.0, 0.4, 0.8]`, audio length `1.2 * 24000`); `test_fake_codec_shapes` (1.2 s → `(15, 8)`, values in `[0, 2048)`, decode length `15 * 1920`); `test_fake_asr_roundtrip` (transcribe(synthesize(t).audio) == t; unknown audio → `""`); `test_make_fake_components` (`make_*(SpeechConfig(..., tts="fake", codec="fake", asr="fake"))` return the fakes).
- [ ] **Step 2: Run** `pytest tests/test_speech_basics.py -v` → FAIL.
- [ ] **Step 3: Implement.**
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(m2): speech interfaces, deterministic fakes, WER`.

---

### Task 3: Chunking and token start frames

**Files:** Create `src/llm/speech/chunk.py`; Test `tests/test_speech_chunk.py`

**Interfaces:**
- Consumes: `BPETokenizer` (M1), `WordTiming`.
- Produces:
  - `split_sentences(text: str) -> list[str]` (split after `.`, `!`, `?` followed by whitespace, keeping closing quotes with their sentence; strips).
  - `chunk_story(text: str, max_words: int) -> list[str]` (greedy whole sentences ≤ `max_words`; a single longer sentence becomes its own chunk). Builder uses `max_words = int(max_seconds * 2.5)`.
  - `class TimingMismatch(Exception)`
  - `token_start_frames(text: str, token_ids: list[int], tok: BPETokenizer, words: list[WordTiming], frame_rate: float) -> list[int]`: match `normalize_words(text)` to the TTS words in order (normalised the same way); raise `TimingMismatch` on any difference. Each token's start frame = `floor(start_s * frame_rate)` of the word containing the token's first non-space character (char offsets found by decoding the token prefix); tokens of pure whitespace/punctuation take the frame of the preceding word (0 for the first). Result is non-decreasing.

- [ ] **Step 1: Write failing tests:** `test_split_sentences` (`'She said, "Hi!" Then she left. The end.'` → 3 sentences, quote kept with the first); `test_chunk_story_respects_limit`; `test_chunk_story_long_sentence_alone`; `test_token_start_frames_fake_tts` (fixture tokenizer from M1 `tok` fixture or trained on stories; FakeTTS timings → frames non-decreasing, first token 0, the token starting "went" maps to `floor(0.4*12.5)=5`); `test_token_start_frames_mismatch` (words `["lily", "goes"]` for text "Lily went" → `TimingMismatch`).
- [ ] **Step 2: Run** → FAIL.  **Step 3: Implement.**  **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(m2): sentence chunking and token-to-frame alignment`.

---

### Task 4: Resumable speech data builder + `llm build-speech`

**Files:** Create `src/llm/speech/build.py`; Modify `src/llm/cli.py`; Test `tests/test_speech_build.py`

**Interfaces:**
- Consumes: Tasks 1–3, M1 `data_dir`, `tokenizer_dir`, `BPETokenizer`, manifest helpers.
- Produces:
  - `speech_dir(cfg) -> Path` = `<speech.root>/<speech.name>-v<speech.version>`
  - On-disk format per split: `<split>_codes.bin` (uint16, row-major `[sum frames, K]`), `<split>_index.jsonl` lines `{"id", "text", "text_ids", "token_frames", "offset", "frames", "seconds"}`; `stats.json` (counts by reason); `samples/<id>.wav` for the first `keep_wavs` train chunks; `manifest.json` written only when all splits are complete (`extra`: voice, codec, tts, asr names, frame_rate, codebooks, per-split chunks/frames/hours, drops by reason; `inputs`: data + tokenizer manifest checksums; `info`: build seconds).
  - `build_speech_data(cfg, tts: TTS, codec: Codec, asr: ASR, max_seconds_this_session: float | None = None) -> tuple[Path, bool]` → `(dir, complete)`. Order: test (until `test_chunks`), val (until `val_chunks`), train (until `hours` of audio). Docs read in file order; chunk id `f"{split}-{doc_index}-{chunk_index}"`. Per chunk: synthesize → drop `too_long` if > `max_seconds` → `token_start_frames` (drop `timing_mismatch`) → if `sha256(id) % 10000 < qa_fraction*10000`: ASR on the TTS audio, drop `qa_wer` if WER > `qa_max_wer` → `codec.encode` → append codes, then index line (codes first, index second, both flushed + fsynced).
  - Resume: read the index (ignore a final line that isn't valid JSON, rewriting the file without it); truncate the codes file to the end of the last indexed record; skip ids already present; drop counts are persisted in `stats.json` after every chunk.
  - CLI `llm build-speech <config> [--max-hours-session H]` prints progress and whether complete.

- [ ] **Step 1: Write failing tests** (fakes + tiny config with `speech: {tts: fake, codec: fake, asr: fake, hours: 0.02, val_chunks: 3, test_chunks: 3}` built on `prepared_tokenized`):
  `test_build_outputs_and_manifest` (complete; `verify_files` passes; index offsets contiguous; codes file size == total frames × 8 × 2 bytes); `test_build_split_isolation` (no chunk id from test/val docs appears in train); `test_build_counts_timing_mismatch` (monkeypatched TTS that returns `"mister"` for `"Mr."` docs → counted, not raised); `test_build_qa_drops_bad_audio` (ASR returning `"wrong words"` with `qa_fraction=1.0` → all chunks dropped as `qa_wer`, stats sum equals attempted); `test_build_resume_after_partial_write` (stop after N chunks via `max_seconds_this_session`; append half a JSON line to the index and 100 junk bytes to codes; resume → complete, ids unique, offsets contiguous, same final index as an uninterrupted build); `test_build_deterministic` (two fresh builds → identical manifest checksum); `test_cli_build_speech`.
- [ ] **Step 2: Run** → FAIL.  **Step 3: Implement.**  **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(m2): resumable TinyStories-Speech builder`.

---

### Task 5: Sequence layouts

**Files:** Create `src/llm/speech/layout.py`; Test `tests/test_speech_layout.py`

**Interfaces:**
- Produces:
  - `EMPTY = 2048`; `SpecialIds(bos, audio, end_audio, pad_text)` from `special_ids(vocab_size: int)` = `(V, V+1, V+2, V+3)`.
  - `@dataclass Example(text: np.ndarray[int64,(T,)], codes: np.ndarray[int64,(T,K)], text_mask: np.ndarray[bool,(T,)], code_mask: np.ndarray[bool,(T,K)])`. Masks mark positions that are prediction **targets**.
  - `build_aligned(text_ids, token_frames, codes, sp: SpecialIds, delay: int) -> Example`:
    positions = `[BOS] + N + delay frame positions + [END_AUDIO]`. Frame position `p` (1-based after BOS) holds codebook 0 of frame `p-1` and codebooks 1..K-1 of frame `p-1-delay` (EMPTY where out of range). Text slots: token `i` goes to slot `max(token_frames[i] - 1, last_slot + 1)` (one frame of lead; spill forward when crowded); slots without a token hold `PAD_TEXT`; if tokens would spill past the last frame position, extra positions with all-EMPTY codes are appended before END_AUDIO. Masks: text target at every position after BOS; code target where code != EMPTY.
  - `build_text_then_audio(text_ids, codes, sp, delay) -> Example`: `[BOS] + text tokens (codes EMPTY) + [AUDIO] + N + delay frame positions (text PAD_TEXT) + [END_AUDIO]`.
  - `build_text_only(text_ids, sp) -> Example`: `[BOS] + tokens`, codes all EMPTY, no code targets.
  - `extract_codes(ex: Example, delay: int) -> np.ndarray[(N,K)]` and `extract_text(ex: Example, sp) -> list[int]` (inverse; drops specials and PAD_TEXT).
  - `max_text_lag(ex, token_frames, delay) -> int` (max over tokens of slot − (start frame + 1)); used in tests and the data report.

- [ ] **Step 1: Write failing tests:** `test_aligned_roundtrip` and `test_baseline_roundtrip` (random codes `(37, 8)`, tokens with frames → extract gives back codes and text exactly, delay 0 and 1); `test_acoustic_delay_positions` (codebook 0 of frame 0 at position 1; codebook 3 of frame 0 at position 2 with delay 1); `test_aligned_text_leads_when_sparse` (tokens 5 frames apart → each slot == frame − 1 + 1 offset for BOS, lag ≤ 0); `test_aligned_spill_when_crowded` (5 tokens in one frame → consecutive slots, `max_text_lag ≤ 4`); `test_aligned_extends_when_text_outlasts_audio`; `test_masks` (no target at position 0; END_AUDIO is a text target; EMPTY never a code target); `test_text_only`.
- [ ] **Step 2: Run** → FAIL.  **Step 3: Implement.**  **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(m2): aligned and baseline speech-text sequence layouts`.

---

### Task 6: Speech dataset and deterministic batching

**Files:** Create `src/llm/speech/dataset.py`; Test `tests/test_speech_dataset.py`

**Interfaces:**
- Consumes: Task 4 format, Task 5 builders, M1 `TokenChunks`, `DataPosition`, `epoch_order`.
- Produces:
  - `class SpeechShards(dir: Path, split: str)`: memmaps codes; `__len__`; `__getitem__(i) -> (text_ids: list[int], token_frames: list[int], codes: np.ndarray[(N,K)])`; `seconds_total`.
  - `@dataclass SpeechBatch(text: LongTensor[B,T], codes: LongTensor[B,T,K], text_mask: BoolTensor[B,T], code_mask: BoolTensor[B,T,K])`; `to(device) -> SpeechBatch`.
  - `collate(examples: list[Example], sp) -> SpeechBatch` (right-pad to the longest: text `PAD_TEXT`, codes `EMPTY`, masks False).
  - `next_speech_batch(ds: SpeechShards, pos: DataPosition, batch_size: int, seed: int, audio: AudioConfig, sp, step: int, text_chunks: TokenChunks | None) -> tuple[SpeechBatch, DataPosition]`: if `text_chunks` and `(step + 1) % audio.text_every == 0` → a text-only batch from `text_chunks` (position unchanged; chunk choice from `epoch_order(len, seed, step)`); else the next `batch_size` speech examples in `epoch_order(len(ds), seed, epoch)` order, built with the configured layout and delay.

- [ ] **Step 1: Write failing tests:** `test_shards_roundtrip` (built fake data → item 0 equals what the builder wrote); `test_collate_padding_masks`; `test_same_seed_same_batches`; `test_resume_position_identical` (10 batches straight vs 4 + resume from returned position + 6: equal); `test_text_every_10th_step` (steps 9 and 19 are text-only, others speech; text batch has no code targets); `test_layout_switch` (baseline layout batch contains `AUDIO` token, aligned does not).
- [ ] **Step 2: Run** → FAIL.  **Step 3: Implement.**  **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(m2): speech dataset with deterministic, resumable batching`.

---

### Task 7: KV cache in the Transformer block

**Files:** Modify `src/llm/model/gpt.py`, `src/llm/generate.py`; Test `tests/test_kv_cache.py`

**Interfaces:**
- Produces:
  - `CausalSelfAttention.forward_cached(x: Tensor[B,t,C], past: tuple[Tensor, Tensor] | None) -> tuple[Tensor, tuple[Tensor, Tensor]]` (keys/values `[B,H,T_total,Dh]`; attention over past + new with causal mask for the new block).
  - `Block.forward_cached(x, past) -> (x, kv)`.
  - `GPT.forward_cached(idx: LongTensor[B,t], cache: list | None, start_pos: int) -> tuple[logits, cache]`.
  - `generate(...)` (M1) uses the cache while the total length ≤ `context_length`, and falls back to full recompute over the cropped window once it is exceeded. The M1 API and outputs are unchanged.

- [ ] **Step 1: Write failing tests:** `test_cached_equals_full_forward` (prompt of 10 then 5 single tokens: cached logits allclose(atol 1e-5) full-forward logits, both attention impls); `test_generate_unchanged` (greedy `generate` output equals a reference recompute loop on a seeded model, prompt shorter and longer than context); `test_generate_faster_with_cache` (CPU, 200 new tokens with context 256: cached path ≥ 2× faster than the recompute reference; mark `slow`).
- [ ] **Step 2: Run** → FAIL.  **Step 3: Implement.**  **Step 4: Run** `pytest -q` → all M1 + new green.
- [ ] **Step 5: Commit** `feat: KV cache for fast generation`.

---

### Task 8: SpeechGPT (brain + voice box)

**Files:** Create `src/llm/model/voice.py`; Test `tests/test_voice_model.py`

**Interfaces:**
- Consumes: `Block`, `ModelConfig`, `AudioConfig`, `SpeechBatch`.
- Produces:
  - `class DepthTransformer(nn.Module)(audio: AudioConfig, brain_d: int)`: `in_proj: Linear(brain_d, depth_d)`, per-codebook input embeddings `Embedding(codebook_size + 1, depth_d)` for codebooks 0..K-2, positional embedding `(K, depth_d)`, `depth_layers` M1 `Block`s (from a `ModelConfig(depth_layers, depth_d_model, depth_n_head, context_length=K)`), `ln_f`, K heads `Linear(depth_d, codebook_size)`. `forward(h: Tensor[N, brain_d], codes: LongTensor[N, K]) -> Tensor[N, K, codebook_size]` (teacher forcing: position 0 = `in_proj(h)`, position k = emb_{k-1}(codes[:, k-1])). `sample(h: Tensor[1, brain_d], temperature, top_k, generator) -> LongTensor[K]` using `forward_cached`.
  - `@dataclass SpeechOutput(loss: Tensor, parts: dict[str, float], text_logits: Tensor)`
  - `class SpeechGPT(nn.Module)(cfg: ModelConfig, audio: AudioConfig, text_vocab: int)`: `tok_emb(text_vocab, d)`, `code_emb`: K × `Embedding(codebook_size + 1, d)`, `pos_emb`, `blocks`, `ln_f`, `text_head` weight-tied to `tok_emb`, `depth`. Init as M1 (normal 0.02, residual projections scaled).
    - `hidden(text: LongTensor[B,T], codes: LongTensor[B,T,K]) -> Tensor[B,T,d]` (input = tok_emb(text) + Σ_k code_emb_k(codes[..., k]) + pos).
    - `forward(batch: SpeechBatch) -> SpeechOutput`: `h = hidden(...)[:, :-1]`; text loss = CE(text_head(h), text[:, 1:]) over `text_mask[:, 1:]`; code loss_k = CE(depth(h[sel], codes[:, 1:][sel]), …) over `code_mask[:, 1:, k]`, where `sel` = positions with any code target; total = `text_loss_weight·text + Σ w_k·code_k / Σ w_k`. `parts` holds `text`, `code_1`..`code_8`, `total` (floats). Empty masks contribute 0.
    - `step(text_t: LongTensor[B], codes_t: LongTensor[B,K], cache, pos: int) -> tuple[Tensor[B,d], cache]` (KV-cached brain step).
    - `num_params() -> int`.

- [ ] **Step 1: Write failing tests:** `test_shapes_and_parts` (tiny dims; loss finite; parts keys `text, code_1..code_8, total`); `test_param_count_m2_config` (the M2 config sizes → `25e6 <= num_params() <= 31e6`); `test_brain_causal` (changing codes at position 10 doesn't change hidden[:, :10]); `test_depth_causal_within_frame` (changing codes[:, 3] doesn't change depth logits for codebooks 0..3); `test_masked_positions_ignored` (all-False masks → loss 0); `test_initial_loss_near_ln` (text part ≈ ln(text_vocab), each code part ≈ ln(2048) within 0.3); `test_step_matches_hidden` (cached `step` loop over 12 positions == `hidden` within 1e-5); `test_depth_sample_matches_forward` (greedy `sample` == argmax chain of teacher-forced `forward`).
- [ ] **Step 2: Run** → FAIL.  **Step 3: Implement.**  **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(m2): SpeechGPT brain and depth voice box`.

---

### Task 9: Speech generation (speak, continue)

**Files:** Create `src/llm/speech/generate.py`; Test `tests/test_speech_generate.py`

**Interfaces:**
- Consumes: `SpeechGPT.step`, `DepthTransformer.sample`, layout helpers, `Codec`, `ASR`, `token_start_frames`.
- Produces:
  - `@dataclass SpeechSampling(temperature: float = 0.8, top_k: int = 50, seed: int = 1234, text_temperature: float = 0.0)`
  - `speak(model: SpeechGPT, text: str, tok, audio: AudioConfig, sampling: SpeechSampling, max_frames: int | None = None) -> np.ndarray[(N,K)]`:
    - empty/whitespace text → `ValueError("text is empty")`;
    - `max_frames` default = `min(context_length - len(tokens) - delay - 3, ceil(2 * words / 2.5 * 12.5))`;
    - **aligned:** at each frame position the text slot is a forced choice from `{PAD_TEXT, next_token}` (both remaining) or `{PAD_TEXT, END_AUDIO}` (all tokens placed), decided by the text head (greedy when `text_temperature == 0`); END_AUDIO ends generation;
    - **baseline:** feed `BOS + tokens + AUDIO`, then frames with text slot `PAD_TEXT` until the text head prefers `END_AUDIO` over `PAD_TEXT`;
    - frame codes come from `depth.sample`; the delay is undone with `extract_codes` semantics. Returns codes only (caller decodes with the codec).
  - `continue_speech(model, prompt_audio: np.ndarray, sample_rate: int, tok, codec: Codec, asr: ASR, audio: AudioConfig, sampling, max_new_frames: int = 250) -> np.ndarray[(N_new,K)]`: encode the prompt, align its transcript with `token_start_frames`, build the aligned prefix without END_AUDIO, keep the most recent positions so prefix + `max_new_frames` ≤ context, then generate with a free text head (full vocab) until END_AUDIO or the limit. Baseline layout → `NotImplementedError("continue-speech requires the aligned layout")`.
  - `real_time_factor(gen_seconds: float, frames: int, frame_rate: float) -> float`.

- [ ] **Step 1: Write failing tests** (tiny random SpeechGPT, fakes): `test_speak_rejects_empty_text`; `test_speak_deterministic_with_seed`; `test_speak_places_all_tokens_aligned` (record text slots via a spy: every token appears once, in order); `test_speak_stops_at_max_frames` (model patched so the text head never prefers END_AUDIO → returns exactly `max_frames` frames); `test_speak_baseline_layout_runs`; `test_continue_crops_long_prompt` (prompt longer than context → runs, output ≤ `max_new_frames`); `test_continue_requires_aligned`; `test_speak_uses_cache` (number of brain `step` calls == frames + prefix length; no full-sequence recompute).
- [ ] **Step 2: Run** → FAIL.  **Step 3: Implement.**  **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(m2): KV-cached speech generation and continuation`.

---

### Task 10: Generic trainer refactor + SpeechTrainer + warm start

**Files:** Modify `src/llm/train/loop.py`, `src/llm/cli.py`; Create `src/llm/train/speech_loop.py`; Test `tests/test_speech_train.py`

**Interfaces:**
- Produces:
  - `BaseTrainer` (in `loop.py`) holding everything model-agnostic from the current `Trainer`: precision, optimizer, scaler, state, provenance, `_account`, `checkpoint`, wall-time/signal/NaN logic, logging, the `run()` loop. Subclass hooks: `build_model() -> nn.Module`, `next_batch(pos: DataPosition) -> tuple[object, DataPosition]`, `loss(batch) -> tuple[Tensor, dict[str, float]]`, `evaluate() -> dict`, `provenance_inputs() -> dict[str, str]`. Logged train records add the `parts` dict.
  - `TextTrainer(BaseTrainer)`: the M1 behaviour. `train(cfg, run_dir, resume, stop_at_step)` keeps its signature and dispatches to `SpeechTrainer` when `cfg.audio is not None`.
  - `SpeechTrainer(BaseTrainer)` (in `speech_loop.py`): model `SpeechGPT(cfg.model, cfg.audio, tokenizer V + 4)`; batches from `next_speech_batch` with M1 `TokenChunks` as the text-only source; provenance = speech + tokenizer + shards manifests; `evaluate()` = mean loss parts on `eval_batches` fixed val batches, plus every `eval_every` steps `speak` on the 10 demo sentences → `samples/step_<n>/<i>.npy` codes (decoded to wav only when a real codec is configured).
  - `warm_start_from_text(model: SpeechGPT, ckpt: Path) -> list[str]`: loads a M1-format `GPT` checkpoint with matching `n_layer/d_model/n_head`, copying `blocks.*`, `ln_f.*`, `pos_emb` (first min(T) rows), and `tok_emb` rows `0..V-1`; returns the loaded parameter names; mismatched dims → `ValueError` naming the key. Applied once at fresh start when `audio.init_from` is set (recorded in `run.json`).

- [ ] **Step 1: Write failing tests:** all existing `tests/test_train.py` and `tests/test_resume.py` unchanged (refactor safety net); `test_speech_overfit` (fake data, tiny dims, 300 steps → code_1 and text parts each fall below 30% of their initial value; `slow`); `test_speech_resume_bitwise` (stop at 7, resume to 20 == uninterrupted, CPU); `test_text_only_steps_logged` (metrics at text-only steps have code parts absent or 0, text present); `test_warm_start_copies_weights` (blocks equal source, new rows differ); `test_warm_start_dim_mismatch`; `test_cli_train_dispatches_speech`.
- [ ] **Step 2: Run** → FAIL (new tests).  **Step 3: Refactor** first (run `pytest -q` → M1 suite green), **then implement** SpeechTrainer and warm start.
- [ ] **Step 4: Run** `pytest -q` → all green.
- [ ] **Step 5: Commit** `refactor: shared BaseTrainer; feat(m2): speech trainer with text warm start`.

---

### Task 11: Speech evaluation + `llm eval-speech` + report section

**Files:** Create `src/llm/speech/evaluate.py`; Modify `src/llm/report.py`, `src/llm/cli.py`; Test `tests/test_speech_eval.py`

**Interfaces:**
- Produces:
  - `DEMO_SET_VERSION = "speech-demo-v1"`; `DEMO_SENTENCES: list[str]` (10 short TinyStories-style sentences, frozen like M1 prompts with a checksum test).
  - `evaluate_speech(model, tok, codec: Codec, asr: ASR, examples: list[tuple[str, np.ndarray]], audio: AudioConfig, sampling: SpeechSampling) -> dict` → `{"wer", "cer", "duration_ratio", "rtf", "n", "per_item": [...]}`; each item: `speak` → `codec.decode` → `asr.transcribe` → WER/CER vs text; duration ratio = generated frames / reference frames (reference = example codes length).
  - `ceiling_wer(codec, asr, examples) -> dict` (decode stored reference codes → ASR → WER: the Kokoro→Mimi ceiling).
  - CLI `llm eval-speech <run_dir> [--n 200] [--floor]` → writes `<run>/speech_eval.json` with `model`, `ceiling`, and optionally `floor` (same eval with a freshly initialised model, seed 0).
  - `build_report` adds a "Speech" section when `speech_eval.json` exists: WER table (ceiling / model / floor), duration ratio, RTF, demo sample links; `compare_runs` includes WER when both runs have it.

- [ ] **Step 1: Write failing tests:** `test_demo_set_frozen` (10 items; sha256 constant); `test_ceiling_with_fakes` (FakeCodec decode isn't recognisable by FakeASR → WER 1.0; with an identity codec + FakeASR → 0.0); `test_evaluate_speech_keys_and_determinism`; `test_duration_ratio_flags_runaway` (patched model that never ends → `duration_ratio > 1.5`); `test_report_speech_section` (report contains "## Speech" and the WER table when `speech_eval.json` exists); `test_cli_eval_speech`.
- [ ] **Step 2: Run** → FAIL.  **Step 3: Implement.**  **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(m2): Whisper-scored speech evaluation and report section`.

---

### Task 12: Creator identity data (PCID): statements, mixing, recall evaluation

**Files:** Create `src/llm/identity/{__init__,pcid}.py`; Modify `src/llm/config.py`, `src/llm/speech/build.py`, `src/llm/speech/dataset.py`, `src/llm/train/speech_loop.py`, `src/llm/speech/evaluate.py`, `src/llm/cli.py`; Test `tests/test_identity.py`, fixture `tests/fixtures/pcid_sample/` (a small **fictional** creator, "Ada Example", with the same file layout — never the real dataset)

**Interfaces:**
- Produces:
  - `IdentityConfig(path: str | None = None, hf_dataset: str | None = None, name: str = "pcid", version: int = 1, exclude_fields: tuple[str, ...] = ("current_location",), holdout_fraction: float = 0.2, text_every: int = 50, speech_every: int = 20, root: str = "identity")`; `Config.identity: IdentityConfig | None = None`; exactly one of `path`/`hf_dataset` required (`ConfigError`).
  - `@dataclass(frozen=True) Statement(text: str, source: str, status: str, fact: str, holdout: bool)`; `@dataclass(frozen=True) Probe(prompt: str, expected: tuple[str, ...], fact: str)`.
  - `load_pcid(root: Path) -> dict[str, object]` (reads the documented files; missing optional files skipped; unknown files ignored).
  - `build_statements(pcid: dict, exclude_fields: tuple[str, ...], holdout_fraction: float) -> list[Statement]` — deterministic templates per fact (creator name ×4 phrasings, short name, personality, naming meaning, origin points, vision/capability goals phrased "Pluto's creator wants Pluto to …", approved creator message sentences, education ×2 each, projects (name + category), seed Q&A as `"Question: {q} Answer: {a}"`); holdout chosen by `sha256(text) % 1000 < holdout_fraction*1000`, never holding out all phrasings of a fact.
  - `build_probes(pcid) -> list[Probe]`: `("Pluto was created by", ("deepak",), "creator_name")`, `("Pluto is named after", ("universe", "space", "planet"), "naming")`, `("Pluto's creator studied", ("computer science", "btech"), "education")`, `("One of the creator's AI projects is", ("echo", "aira", "gita"), "projects")` — expected strings derived from the loaded records, lowercase.
  - `prepare_identity(cfg) -> Path`: writes `identity/<name>-v<version>/{statements.jsonl, probes.jsonl, identity_report.md}` + manifest (deterministic checksum; `extra` = counts by source/status/holdout). HF source via `huggingface_hub.snapshot_download(repo_type="dataset")`.
  - Speech builder: when `cfg.identity` is set, an extra split `identity` (train statements) and `identity_eval` (holdout) synthesised with the same pipeline (no hours limit; all statements).
  - Batching: `next_speech_batch` gains `identity_text: list[list[int]] | None` and `identity_speech: SpeechShards | None`; precedence per step: `(step+1) % identity.text_every == 0` → identity text batch; else `(step+1) % identity.speech_every == 0` → identity speech batch; else the existing text_every/speech logic. Deterministic choice via `epoch_order(len, seed, step)`.
  - `identity_recall(model, tok, probes, mode: Literal["text", "speech"], codec=None, asr=None, audio=None) -> dict` → `{"recall", "per_fact": {fact: recall}, "items": [...]}`; text mode = greedy continuation of 24 tokens contains any expected string (case-insensitive); speech mode = `speak(prompt)` then `continue_speech` 60 frames → ASR transcript contains expected. Written into `speech_eval.json` under `"identity"`; report shows an "Identity recall" table.
  - CLI `llm prepare-identity <config>`.

- [ ] **Step 1: Write failing tests** (fictional fixture only): `test_statements_deterministic`; `test_statements_faithful` (every statement's key value appears verbatim in the source records — no invented names, dates or numbers); `test_aspirations_stay_goals` (statements from `long_term_vision`/`creator_philosophy` all contain "wants" or "aims"); `test_exclude_fields` (fixture location string never appears); `test_holdout_keeps_each_fact_trainable`; `test_probes_from_records`; `test_identity_mixing_schedule` (steps 49, 99 identity text; 19, 39, 59 identity speech; precedence at step 99 = text); `test_identity_recall_text_with_memorising_model` (tiny SpeechGPT overfit on the statements for 300 steps → creator-name recall ≥ 0.9 on held-out phrasings; `slow`); `test_prepare_identity_manifest`; `test_identity_config_requires_one_source`.
- [ ] **Step 2: Run** `pytest tests/test_identity.py -v` → FAIL.  **Step 3: Implement.**  **Step 4: Run** → PASS; `pytest -q` green.
- [ ] **Step 5: Commit** `feat(m2): creator identity statements, mixing and recall evaluation`.

---

### Task 13: `llm speak`, `llm continue-speech`, Gradio demo

**Files:** Create `src/llm/speech/app.py`; Modify `src/llm/cli.py`, `src/llm/train/checkpoint.py`; Test `tests/test_speech_cli.py`

**Interfaces:**
- Produces:
  - `load_speech_model_for_inference(path_or_run_dir) -> tuple[SpeechGPT, BPETokenizer, Config]` (in `checkpoint.py`, alongside the M1 loader).
  - CLI: `llm speak <run> --text "…" --out out.wav [--seed --temperature --top-k]`; `llm continue-speech <run> --audio in.wav --out out.wav [--max-new-frames]`; both use `make_codec`/`make_asr` from the run's `speech` config and write 24 kHz wav via `soundfile` (`.npy` codes when `--codes-only`).
  - `build_app(run_dir: Path)` (Gradio Blocks: text → audio tab; microphone → continuation tab); `llm demo <run> [--share]`.

- [ ] **Step 1: Write failing tests:** `test_cli_speak_codes_only` (fake codec config, trained tiny run from a fixture → `.npy` written, shape `(N, 8)`); `test_cli_speak_empty_text_errors` (exit code 2 with message); `test_app_importable_without_gradio` (importing `llm.speech.app` doesn't import gradio until `build_app` is called).
- [ ] **Step 2: Run** → FAIL.  **Step 3: Implement.**  **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(m2): speak, continue-speech and Gradio demo`.

---

### Task 14: Real adapters + `llm speech-check` (Stage 0)

**Files:** Create `src/llm/speech/adapters.py`; Modify `src/llm/cli.py`; Test `tests/test_speech_adapters.py`

**Interfaces:**
- Produces:
  - `KokoroTTS(voice: str)`: `kokoro.KPipeline(lang_code="b")` (British); `synthesize` concatenates the pipeline's audio segments at 24 kHz and builds `WordTiming`s from each result's token timestamps (`start_ts`/`end_ts`, offset by the segment start); `words=None` if timestamps are absent.
  - `MimiCodec()`: `transformers.MimiModel.from_pretrained("kyutai/mimi")` (eval, on CUDA if available); `encode` → `model.encode(x, num_quantizers=8).audio_codes` → `(frames, 8)`; `decode` → `model.decode(codes).audio_values`.
  - `WhisperASR(model_name)`: transformers ASR pipeline with `return_timestamps="word"` → `Transcript`.
  - The adapter's pinned model ids and library versions are returned by `describe() -> dict` and stored in the speech manifest `extra`.
  - CLI `llm speech-check <config> --n 100`: synthesizes `n` val chunks, reports the share with usable word timings, mean |timing-derived duration − audio duration|, the Mimi round-trip ceiling WER, and measured TTS/codec/ASR seconds per audio second; if Kokoro timings are missing for > 5% of chunks it says so and recommends `--timings whisper` (builder option that takes word timings from `asr.transcribe(...).words` instead).

- [ ] **Step 1: Write failing tests** using fake `kokoro` / `transformers` modules injected via `monkeypatch.setitem(sys.modules, …)` (same pattern as `tests/test_hub.py`): `test_kokoro_adapter_builds_timings` (two segments → offsets added); `test_kokoro_adapter_missing_timestamps` (→ `words is None`); `test_mimi_adapter_shapes`; `test_whisper_adapter_words`; `test_builder_whisper_timings_option`; `test_cli_speech_check_with_fakes`.
- [ ] **Step 2: Run** → FAIL.  **Step 3: Implement.**  **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat(m2): Kokoro, Mimi and Whisper adapters and speech-check`.

---

### Task 15: Configs, Kaggle notebook, README, end-to-end test

**Files:** Create `configs/m2-stage0.yaml`, `configs/m2-warmstart.yaml`, `configs/m2-speech-aligned.yaml`, `configs/m2-speech-baseline.yaml`, `notebooks/m2_kaggle.ipynb`, `tests/test_m2_e2e.py`; Modify `README.md`

Config values:

| | m2-warmstart | m2-speech-aligned | m2-speech-baseline |
|---|---|---|---|
| data/tokenizer/shards | M1 TinyStories v1 | same | same |
| model | 8 L, d 384, 6 H, ctx 640 | same | same |
| audio | — | layout aligned, delay 1, init_from = warm-start best.pt | layout text_then_audio, otherwise identical |
| speech | — | tinystories-speech v1, bf_emma, 200 h | same |
| train | 2,400 steps × 81,920 tokens (micro 32), lr 1e-3, warmup 200 | micro 16, tokens_per_step 40,960, 6,000 steps, lr 6e-4, warmup 300, eval_every 500 | identical to aligned |
| identity | — | path or private HF dataset (`identity.hf_dataset`) | same |
| run.name | m2-warmstart | m2-aligned | m2-baseline |

`m2-stage0.yaml` = aligned config with `speech.hours: 0.3`, `val_chunks: 20`, `test_chunks: 20`, `max_steps: 50`.

- [ ] **Step 1: Write failing tests:** `test_m2_configs_load` (all four load; aligned and baseline differ only in `audio.layout` and `run.name`, via `config_diff`); `test_m2_pipeline_cli` (`slow`; fake components: build-speech → train warm-start (tiny override) → train aligned with `init_from` and `--stop-at-step` → resume → eval-speech → speak `--codes-only` → report with Speech section).
- [ ] **Step 2: Run** → FAIL until the configs exist; fix integration bugs in the owning module with a regression test there.
- [ ] **Step 3: Notebook** `m2_kaggle.ipynb`: install `.[speech,hub]` + `apt-get install -y espeak-ng`; Stage 0 `speech-check`; Stage 1 `build-speech` with `--max-hours-session` and Hub sync of the speech dir (upload after each session; pull at start); Stage 2 warm-start train; Stage 3a/3b training with resume cells; `eval-speech --floor`; play the demo wavs inline; push results (token-safe cell from M1).
- [ ] **Step 4: README:** M2 section (what it is, commands, disclosure of Kokoro/Mimi/Whisper, the comparison and how it's measured).
- [ ] **Step 5: Run** `pytest -q` → all green; **commit** `feat(m2): configs, Kaggle notebook, end-to-end test, README` and push.

---

## Self-review notes

- Spec §2 dataset → Tasks 2–4, 13; §3 model/layouts/KV cache → Tasks 5, 7, 8, 9; §4 training/warm start/text mixing/comparison → Tasks 6, 10, 14; §5 evaluation → Task 11; §2b identity → Task 12; §6 demo → Tasks 13, 15; §7 testing → per task + Task 14; §8 risks → Review Focus + Task 13 (`speech-check`, Whisper timing fallback).
- Spec refinement (ruling): "text never lags its audio" is implemented as one frame of text lead with forward spill when tokens crowd a frame; `max_text_lag` is measured and reported instead of guaranteed 0. Cost if wrong: slot rule change in `build_aligned` only.
- Spec refinement: the voice box uses d_model 192 (not 256) to keep the total within the spec's ~25–30M (computed ≈ 30M).
- Interface names cross-checked: `SpeechConfig`, `AudioConfig`, `special_ids`, `Example`, `SpeechBatch`, `next_speech_batch`, `SpeechGPT.step`, `DepthTransformer.sample`, `speak`, `continue_speech`, `evaluate_speech`, `ceiling_wer`, `warm_start_from_text`, `load_speech_model_for_inference`, `make_tts/make_codec/make_asr`.
