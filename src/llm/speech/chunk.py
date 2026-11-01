"""Sentence chunking of stories and alignment of text tokens to audio frames.

``token_start_frames`` tells the speech model, for every text token, the codec frame at which
the spoken word containing that token starts (from the TTS word timings).
"""

from __future__ import annotations

import math
import re

from llm.speech.interfaces import WordTiming
from llm.speech.wer import normalize_words
from llm.tokenizer.bpe import BPETokenizer

# A sentence ends at ./!/? (runs allowed, e.g. "...") plus any closing quotes/brackets, then whitespace.
_SENTENCE_END = re.compile(r"[.!?]+[\"'”’)\]]*\s+")
# Same word definition as llm.speech.wer.normalize_words: runs of word characters and apostrophes.
_WORD = re.compile(r"[\w']+", re.UNICODE)


class TimingMismatch(Exception):
    """The TTS words do not line up with the words of the text."""


def split_sentences(text: str) -> list[str]:
    """Split after ``.``/``!``/``?`` followed by whitespace; closing quotes stay with their sentence."""
    out, start = [], 0
    for m in _SENTENCE_END.finditer(text):
        out.append(text[start : m.end()].strip())
        start = m.end()
    out.append(text[start:].strip())
    return [s for s in out if s]


def chunk_story(text: str, max_words: int) -> list[str]:
    """Greedily pack whole sentences into chunks of at most ``max_words`` words.

    A single sentence longer than ``max_words`` becomes its own chunk.
    """
    chunks: list[str] = []
    cur: list[str] = []
    cur_words = 0
    for sent in split_sentences(text):
        n = len(sent.split())
        if cur and cur_words + n > max_words:
            chunks.append(" ".join(cur))
            cur, cur_words = [], 0
        cur.append(sent)
        cur_words += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def token_start_frames(
    text: str, token_ids: list[int], tok: BPETokenizer, words: list[WordTiming], frame_rate: float
) -> list[int]:
    """Audio frame at which the word containing each token starts (non-decreasing).

    ``token_ids`` must be (a prefix of) ``tok.encode(text)``. Words are matched to ``words`` in
    order after the same normalisation as ``normalize_words``; any difference raises
    ``TimingMismatch``. A token's frame is that of the word containing its first non-space
    character; tokens starting on punctuation/whitespace take the frame of the preceding word
    (0 before the first word).
    """
    spans = [m.span() for m in _WORD.finditer(text)]
    norm = normalize_words(text)
    tts_words = [w for tw in words for w in normalize_words(tw.word)]
    if len(spans) != len(norm) or norm != tts_words or len(words) != len(norm):
        raise TimingMismatch(f"text words {norm!r} != TTS words {[w.word for w in words]!r}")

    # Frame per character: its word's start frame; non-word chars inherit the preceding word's.
    char_frame = [0] * len(text)
    cur = 0
    prev_end = 0
    for (s, e), tw in zip(spans, words):
        for i in range(prev_end, s):
            char_frame[i] = cur
        cur = math.floor(tw.start_s * frame_rate)
        for i in range(s, e):
            char_frame[i] = cur
        prev_end = e
    for i in range(prev_end, len(text)):
        char_frame[i] = cur

    # Byte offset -> index of the character containing that byte.
    byte_char: list[int] = []
    for ci, ch in enumerate(text):
        byte_char.extend([ci] * len(ch.encode("utf-8")))
    text_bytes = text.encode("utf-8")

    frames: list[int] = []
    offset = 0
    last = 0
    for tid in token_ids:
        piece = tok._bytes[tid]
        end = offset + len(piece)
        if text_bytes[offset:end] != piece:
            raise ValueError("token_ids do not match text (expected a prefix of tok.encode(text))")
        chars = list(dict.fromkeys(byte_char[offset:end]))  # distinct chars covered, in order
        if chars:
            first = next((c for c in chars if not text[c].isspace()), chars[0])
            last = max(last, char_frame[first])
        frames.append(last)
        offset = end
    return frames
