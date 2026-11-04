# M3 layer A/B tests (24M model, 300M tokens, one seed, A10G)

Base: `configs/m3-ab-base.yaml` (6L x 384, ctx 1024, learned positions / LayerNorm / GELU / full MHA).
Each variant changes one thing (window also turns on RoPE). Metric: validation bits per byte at step 1144
(lower is better). Single seed: differences below ~0.005 are treated as noise.

| variant | val loss | bits/byte | vs base | decision |
|---|---:|---:|---:|---|
| base | 4.3193 | 1.4156 | — | |
| rope | 4.0538 | 1.3286 | −0.087 | keep (largest gain) |
| rmsnorm | 4.3133 | 1.4136 | −0.002 | keep (neutral, standard, cheaper) |
| swiglu | 4.1986 | 1.3760 | −0.040 | keep |
| gqa (2 of 6 KV heads) | 4.3189 | 1.4154 | ±0 | keep (same quality, +19–25% batch-1 decode speed, smaller KV cache) |
| window 256 / every 4th layer global (+rope) | 4.0376 | 1.3232 | −0.092 (−0.005 vs rope) | keep (no loss; bounded KV memory for long calls) |
| mtp (1 head, weight 0.3) | 4.3296 | 1.4189 | +0.003 | drop (no gain at this size; consistent with Gloeckle et al. 2024) |
| all (incl. mtp) | 4.0247 | 1.3190 | −0.097 | best overall; gains are not additive (rope −0.087 + swiglu −0.040 alone); includes MTP, which alone hurt slightly |

Recipe for M3 pilot/cloud: RoPE + RMSNorm + SwiGLU + GQA + hybrid sliding window (window = context/4,
every 4th layer global), no MTP. Limitations: one seed, a 24M model and 300M tokens; effects may differ at
150M/400M. The MTP speed use (self-speculative decoding) was not tested.

## What happens to MTP (decision 2026-11-04)
Code stays (`model.mtp_heads`, default 0); the 151M pilot trains without it. Revisit twice, each measured:
1. **Scale test** before the 388M/500M run: short MTP on/off A/B at that size (literature: helps larger models).
2. **Speed use (M5, real-time voice):** train only an MTP head on top of the frozen M3 model, then use it for
   self-speculative decoding (draft token t+2, verify in one step; output identical). Measure acceptance rate
   and real decode speed-up; keep only if it helps.
