"""Speech training (M2): ``SpeechTrainer`` for SpeechGPT on the shared ``BaseTrainer`` loop, and
``warm_start_from_text`` to initialise the brain from an M1 text checkpoint."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from llm.manifest import MANIFEST, atomic_write_bytes, atomic_write_json, manifest_checksum
from llm.model.voice import SpeechGPT
from llm.shards import DataPosition, TokenChunks, shard_dir, verify_shard_provenance
from llm.speech.build import _wav_bytes, speech_dir
from llm.speech.dataset import SpeechShards, next_speech_batch
from llm.speech.evaluate import DEMO_SENTENCES
from llm.speech.generate import SpeechSampling, speak
from llm.speech.layout import special_ids
from llm.tokenizer.bpe import BPETokenizer, tokenizer_dir
from llm.train.loop import BaseTrainer


EVAL_SPLIT = "val"


def warm_start_from_text(model: SpeechGPT, ckpt: Path) -> list[str]:
    """Copy an M1 ``GPT`` checkpoint's brain into ``model``; returns the loaded parameter names.

    Copies ``blocks.*`` and ``ln_f.*`` (exact shapes), the first min(T) rows of ``pos_emb`` and
    ``tok_emb`` rows ``0..V-1`` (V = model text vocab - 4; the special rows keep their init).
    The tied ``lm_head`` is skipped. Any shape mismatch raises ``ValueError`` naming the key.
    """
    payload = torch.load(ckpt, map_location="cpu", weights_only=True)
    saved = (payload.get("config") or {}).get("model")
    if not saved:
        raise ValueError(f"{ckpt}: no config.model in the checkpoint; cannot verify n_layer/d_model/n_head")
    for key in ("n_layer", "d_model", "n_head"):
        if saved.get(key) != getattr(model.cfg, key):
            raise ValueError(f"{key}: checkpoint {saved.get(key)} vs speech model {getattr(model.cfg, key)}")
    src = payload["model"]
    dst = model.state_dict()
    V = model.text_vocab - 4
    loaded: list[str] = []
    with torch.no_grad():
        for key, w in src.items():
            if key.startswith("lm_head."):
                continue
            if not key.startswith(("blocks.", "ln_f.", "pos_emb.", "tok_emb.")):
                continue
            if key not in dst:
                raise ValueError(f"{key}: not in the speech model (n_layer mismatch?)")
            d = dst[key]
            if key == "tok_emb.weight":
                if w.shape != (V, d.shape[1]):
                    raise ValueError(f"{key}: checkpoint {tuple(w.shape)} vs expected {(V, d.shape[1])}")
                d[:V].copy_(w)
            elif key == "pos_emb.weight":
                if w.shape[1] != d.shape[1]:
                    raise ValueError(f"{key}: checkpoint {tuple(w.shape)} vs model {tuple(d.shape)}")
                n = min(w.shape[0], d.shape[0])
                d[:n].copy_(w[:n])
            else:
                if w.shape != d.shape:
                    raise ValueError(f"{key}: checkpoint {tuple(w.shape)} vs model {tuple(d.shape)}")
                d.copy_(w)
            loaded.append(key)
    missing = [k for k in dst if k.startswith(("blocks.", "ln_f.")) and k not in loaded]
    if missing:
        raise ValueError(f"{missing[0]}: missing from the text checkpoint (n_layer mismatch?)")
    return loaded


class SpeechTrainer(BaseTrainer):
    def setup(self) -> None:
        cfg = self.cfg
        verify_shard_provenance(cfg)
        self.sdir = speech_dir(cfg)
        if not (self.sdir / MANIFEST).exists():
            raise FileNotFoundError(f"speech data in {self.sdir} is not complete (no {MANIFEST}); "
                                    "finish `llm build-speech` first")
        self.audio = cfg.audio
        self.tok = BPETokenizer.load(tokenizer_dir(cfg))
        self.sp = special_ids(self.tok.vocab_size)
        self.train_ds = SpeechShards(self.sdir, "train", self.audio.codebooks)
        self.eval_ds = SpeechShards(self.sdir, EVAL_SPLIT, self.audio.codebooks)
        # BOS + tokens must fit the context.
        self.text_chunks = TokenChunks(shard_dir(cfg), "train", seq_len=cfg.model.context_length - 1)
        self._setup_identity()

    def _setup_identity(self) -> None:
        """Creator-identity sources (train statements as text; the ``identity`` speech split)."""
        self.identity_text, self.identity_speech, self.identity_dir = None, None, None
        if self.cfg.identity is None:
            return
        from llm.identity.pcid import identity_dir, load_statements

        self.identity_dir = identity_dir(self.cfg)
        if not (self.identity_dir / MANIFEST).exists():
            raise FileNotFoundError(f"{self.identity_dir}: run `llm prepare-identity` first")
        self.identity_text = [self.tok.encode(s.text) for s in load_statements(self.identity_dir)
                              if not s.holdout]
        if (self.sdir / "identity_index.jsonl").exists():
            self.identity_speech = SpeechShards(self.sdir, "identity", self.audio.codebooks)

    def build_model(self) -> SpeechGPT:
        return SpeechGPT(self.cfg.model, self.audio, self.tok.vocab_size + 4)

    def provenance_inputs(self) -> dict[str, str]:
        return {"speech": manifest_checksum(self.sdir),
                "tokenizer": manifest_checksum(tokenizer_dir(self.cfg)),
                "shards": manifest_checksum(shard_dir(self.cfg)),
                **({"identity": manifest_checksum(self.identity_dir)} if self.identity_dir else {})}

    def on_fresh_start(self) -> None:
        if self.audio.init_from is None:
            return
        path = Path(self.audio.init_from)
        loaded = warm_start_from_text(self.model, path)
        run_json = self.run_dir / "run.json"
        info = json.loads(run_json.read_text()) if run_json.exists() else {}
        info["warm_start"] = {"path": str(path), "loaded": loaded, "at": time.time()}
        atomic_write_json(run_json, info)

    def next_batch(self, pos: DataPosition, micro: int):
        # ``micro`` (global micro-batch index) drives text_every: with accum == 1 every
        # text_every-th optimizer step is text-only.
        return next_speech_batch(self.train_ds, pos, self.t.micro_batch_size, self.t.seed, self.audio,
                                 self.sp, micro, self.text_chunks,
                                 identity_text=self.identity_text, identity_speech=self.identity_speech,
                                 identity=self.cfg.identity, max_text_tokens=self.cfg.model.context_length - 1)

    def loss(self, batch):
        batch = batch.to(self.device)
        with self.autocast():
            out = self.fwd(batch)
        return out.loss, out.parts

    def evaluate(self) -> dict:
        was_training = self.model.training
        self.model.eval()
        sums: dict[str, float] = {}
        pos = DataPosition(0, 0)
        with torch.no_grad():
            for _ in range(self.t.eval_batches):
                batch, pos = next_speech_batch(self.eval_ds, pos, self.t.micro_batch_size, self.t.seed,
                                               self.audio, self.sp, 0, None)
                with self.autocast():
                    out = self.model(batch.to(self.device))
                for k, v in out.parts.items():
                    sums[k] = sums.get(k, 0.0) + v / self.t.eval_batches
        self.model.train(was_training)
        self._speak_demo()
        return {"split": EVAL_SPLIT, "val_loss": sums["total"], **sums}

    def _speak_demo(self) -> None:
        out = self.run_dir / "samples" / f"step_{self.state.step:08d}"
        out.mkdir(parents=True, exist_ok=True)
        codec = None
        if self.cfg.speech.codec != "fake":
            from llm.speech.interfaces import make_codec

            codec = make_codec(self.cfg.speech)
        for i, text in enumerate(DEMO_SENTENCES):
            codes = speak(self.model, text, self.tok, self.audio, SpeechSampling())
            np.save(out / f"{i}.npy", codes)
            if codec is not None and len(codes):
                atomic_write_bytes(out / f"{i}.wav", _wav_bytes(codec.decode(codes), codec.sample_rate))
