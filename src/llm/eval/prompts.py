"""Frozen prompt set. Changing PROMPTS or SAMPLERS requires bumping PROMPT_SET_VERSION
(and the checksum in tests/test_eval.py) so checkpoints are only compared within a version."""
from __future__ import annotations

import json
from pathlib import Path

from llm.eval.metrics import repetition_rate
from llm.generate import generate

PROMPT_SET_VERSION = "tinystories-v1"
PROMPTS = [
    "Once upon a time, there was a little dog named",
    "Lily wanted to go to the park, but",
    "Tom found a big red ball in the garden. He",
    "One day, a little bird could not fly because",
    "The sun was hot, so Mia and her mom",
    "Ben was sad because his toy was broken. His friend",
    "In a small house near the forest, there lived",
    "Sara looked at the sky and saw",
    "The cat and the dog did not like each other. One day,",
    "\"Can I help you?\" asked the kind old man.",
]
SAMPLERS = [
    {"name": "greedy", "temperature": 0.0},
    {"name": "t0.8_k50", "temperature": 0.8, "top_k": 50, "seed": 1234},
]
MAX_NEW_TOKENS = 128


def run_prompt_eval(model, tok, step: int, out_path: Path, max_new_tokens: int = MAX_NEW_TOKENS) -> dict:
    was_training = model.training
    model.eval()
    reps, lines = [], []
    for prompt in PROMPTS:
        for s in SAMPLERS:
            ids = generate(model, tok.encode(prompt), max_new_tokens, temperature=s["temperature"],
                           top_k=s.get("top_k"), seed=s.get("seed"), stop_ids=[tok.eot_id])
            rep = repetition_rate(ids)
            reps.append(rep)
            lines.append(json.dumps({"step": step, "prompt_set": PROMPT_SET_VERSION, "prompt": prompt,
                                     "sampler": s["name"], "completion": tok.decode(ids),
                                     "repetition": rep}, ensure_ascii=False))
    Path(out_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    if was_training:
        model.train()
    return {"repetition_mean": sum(reps) / len(reps)}
