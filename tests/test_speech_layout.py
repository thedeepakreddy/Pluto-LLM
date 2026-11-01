import numpy as np
import pytest

from llm.speech.layout import (
    EMPTY,
    build_aligned,
    build_text_only,
    build_text_then_audio,
    extract_codes,
    extract_text,
    max_text_lag,
    special_ids,
)

V = 4096
K = 8
SP = special_ids(V)


def _codes(n, k=K, seed=0):
    return np.random.default_rng(seed).integers(0, 2048, size=(n, k), dtype=np.int64)


def _toks(n_tokens=12, n_frames=37, seed=1):
    rng = np.random.default_rng(seed)
    ids = [int(x) for x in rng.integers(0, V, size=n_tokens)]
    frames = sorted(int(x) for x in rng.integers(0, n_frames, size=n_tokens))
    return ids, frames


def _text_slots(ex, sp=SP):
    specials = {sp.bos, sp.audio, sp.end_audio, sp.pad_text}
    return [t for t, x in enumerate(ex.text) if int(x) not in specials]


def test_special_ids():
    assert (SP.bos, SP.audio, SP.end_audio, SP.pad_text) == (V, V + 1, V + 2, V + 3)
    assert EMPTY == 2048


@pytest.mark.parametrize("delay", [0, 1])
def test_aligned_roundtrip(delay):
    codes = _codes(37)
    ids, frames = _toks()
    ex = build_aligned(ids, frames, codes, SP, delay)
    assert ex.text.dtype == np.int64 and ex.codes.dtype == np.int64
    assert np.array_equal(extract_codes(ex, delay), codes)
    assert extract_text(ex, SP) == ids
    assert ex.text[0] == SP.bos and ex.text[-1] == SP.end_audio


@pytest.mark.parametrize("delay", [0, 1])
def test_baseline_roundtrip(delay):
    codes = _codes(37)
    ids, _ = _toks()
    ex = build_text_then_audio(ids, codes, SP, delay)
    assert np.array_equal(extract_codes(ex, delay), codes)
    assert extract_text(ex, SP) == ids
    assert ex.text[0] == SP.bos and ex.text[-1] == SP.end_audio
    assert ex.text[1 + len(ids)] == SP.audio
    assert len(ex.text) == 1 + len(ids) + 1 + 37 + delay + 1
    # text part and AUDIO marker carry no codes; frame positions carry PAD_TEXT
    assert (ex.codes[: 2 + len(ids)] == EMPTY).all()
    assert (ex.text[2 + len(ids) : -1] == SP.pad_text).all()


def test_acoustic_delay_positions():
    codes = _codes(10)
    ex0 = build_aligned([5], [0], codes, SP, 0)
    assert ex0.codes[1, 0] == codes[0, 0]
    assert ex0.codes[1, 3] == codes[0, 3]
    ex1 = build_aligned([5], [0], codes, SP, 1)
    assert ex1.codes[1, 0] == codes[0, 0]
    assert ex1.codes[1, 3] == EMPTY
    assert ex1.codes[2, 3] == codes[0, 3]
    assert ex1.codes[2, 0] == codes[1, 0]
    # last frame position holds only the delayed codebooks of the last frame
    assert ex1.codes[10 + 1, 0] == EMPTY
    assert ex1.codes[10 + 1, 3] == codes[9, 3]
    # baseline: same relative shift after the AUDIO marker
    b = build_text_then_audio([5, 6], codes, SP, 1)
    a = 1 + 2  # AUDIO marker position
    assert b.text[a] == SP.audio
    assert b.codes[a + 1, 0] == codes[0, 0] and b.codes[a + 1, 3] == EMPTY
    assert b.codes[a + 2, 3] == codes[0, 3]


def test_aligned_text_leads_when_sparse():
    codes = _codes(40)
    frames = [0, 5, 10, 15, 20]
    ids = [11, 22, 33, 44, 55]
    ex = build_aligned(ids, frames, codes, SP, 1)
    slots = _text_slots(ex)
    # BOS shifts frame f to position f+1; one frame of lead -> position f; frame 0 clamps to 1
    assert slots == [1, 5, 10, 15, 20]
    assert [int(ex.text[s]) for s in slots] == ids
    assert max_text_lag(ex, frames, 1, SP) <= 0
    assert max_text_lag(ex, frames, 1, SP) == 0  # frame-0 token cannot lead further
    # all other frame positions hold PAD_TEXT
    others = [t for t in range(1, len(ex.text) - 1) if t not in slots]
    assert (ex.text[others] == SP.pad_text).all()


def test_aligned_spill_when_crowded():
    codes = _codes(30)
    ids = [1, 2, 3, 4, 5]
    frames = [10] * 5
    ex = build_aligned(ids, frames, codes, SP, 1)
    slots = _text_slots(ex)
    assert slots == [10, 11, 12, 13, 14]
    assert extract_text(ex, SP) == ids
    assert max_text_lag(ex, frames, 1, SP) == 14 - 11 == 3
    assert max_text_lag(ex, frames, 1, SP) <= 4


def test_aligned_extends_when_text_outlasts_audio():
    codes = _codes(3)
    ids = list(range(10, 20))
    frames = [2] * 10  # all at the last frame; far more tokens than positions left
    ex = build_aligned(ids, frames, codes, SP, 1)
    n_frame_pos = 3 + 1
    slots = _text_slots(ex)
    assert slots == list(range(2, 12))
    assert slots[-1] > n_frame_pos  # spilled past the last frame position
    assert len(ex.text) == slots[-1] + 2  # BOS ... last text slot, then END_AUDIO
    assert ex.text[-1] == SP.end_audio
    # extra positions carry no codes
    assert (ex.codes[n_frame_pos + 1 :] == EMPTY).all()
    assert np.array_equal(extract_codes(ex, 1), codes)
    assert extract_text(ex, SP) == ids


def test_masks():
    codes = _codes(20)
    ids, frames = _toks(6, 20)
    for ex in (
        build_aligned(ids, frames, codes, SP, 1),
        build_text_then_audio(ids, codes, SP, 1),
        build_text_only(ids, SP),
    ):
        assert not ex.text_mask[0] and not ex.code_mask[0].any()
        assert ex.text_mask[1:].all()
        flush = (ex.codes[:, 0] == EMPTY) & (ex.codes[:, 1:] != EMPTY).any(axis=1)
        expected = ex.codes != EMPTY
        expected[:, 0] |= flush
        assert np.array_equal(ex.code_mask, expected)
        assert ex.text_mask.dtype == bool and ex.code_mask.dtype == bool
        assert ex.text_mask.shape == ex.text.shape and ex.code_mask.shape == ex.codes.shape
    a = build_aligned(ids, frames, codes, SP, 1)
    assert a.text_mask[-1] and a.text[-1] == SP.end_audio
    assert not a.code_mask[-1].any()
    # a PAD_TEXT frame position is still a text target in both layouts
    pad_pos = [t for t in range(1, len(a.text)) if a.text[t] == SP.pad_text][0]
    assert a.text_mask[pad_pos]
    b = build_text_then_audio(ids, codes, SP, 1)
    assert b.text_mask[-3] and b.text[-3] == SP.pad_text
    # flush positions: codebook-0 EMPTY is a target ("stop audio"), in both layouts
    for ex, delay in [(build_aligned(ids, frames, codes, SP, 2), 2),
                      (build_text_then_audio(ids, codes, SP, 2), 2), (a, 1), (b, 1)]:
        end = len(ex.text) - 1
        last = max(t for t in range(end) if ex.codes[t, 1:].any() and ex.codes[t, 1] != EMPTY)
        fl = [t for t in range(end) if ex.codes[t, 0] == EMPTY and ex.code_mask[t, 0]]
        assert fl == list(range(last - delay + 1, last + 1))
        assert all(ex.codes[t, 0] == EMPTY for t in fl)
        assert np.array_equal(extract_codes(ex, delay), codes)
    # delayed codebooks of the first frame position are EMPTY -> not targets
    assert a.code_mask[1, 0] and not a.code_mask[1, 1:].any()


def test_text_only():
    ids = [7, 8, 9]
    ex = build_text_only(ids, SP)
    assert ex.text.tolist() == [SP.bos, 7, 8, 9]
    assert ex.codes.shape == (4, K) and (ex.codes == EMPTY).all()
    assert not ex.code_mask.any()
    assert ex.text_mask.tolist() == [False, True, True, True]
    assert extract_text(ex, SP) == ids
    assert extract_codes(ex, 1).shape == (0, K)


def test_validation():
    codes = _codes(5)
    with pytest.raises(ValueError):
        build_aligned([1, 2], [0], codes, SP, 1)  # length mismatch
    with pytest.raises(ValueError):
        build_aligned([1, 2], [3, 1], codes, SP, 1)  # decreasing frames
    with pytest.raises(ValueError):
        build_aligned([V], [0], codes, SP, 1)  # id collides with specials
    bad = codes.copy()
    bad[0, 0] = EMPTY
    with pytest.raises(ValueError):
        build_aligned([1], [0], bad, SP, 1)  # EMPTY is not a valid code
    with pytest.raises(ValueError):
        build_aligned([1], [0], codes, SP, -1)
