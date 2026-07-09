<!-- SPDX-License-Identifier: MIT -->

# fable_rd_stack

Rate-aware QAT fine-tune of the PR #101 HNeRV payload, plus a scorer-searched
correction tail (latent-correction sidecar + frame0 pose selector). Trained and
searched end-to-end against the official frozen SegNet/PoseNet scorers on a
single Apple M5 Pro (MPS); no CUDA used.

## Scores (official `evaluate.sh`, this archive)

| axis | seg | pose | bytes | rate | score |
|---|---|---|---|---|---|
| CPU (leaderboard) | 0.00057584 | 0.00002970 | 176,715 | 0.00470669 | **0.192485** |
| MPS (cross-check) | 0.00057585 | 0.00002970 | 176,715 | 0.00470669 | 0.192486 |

## Method

1. **Rate-aware QAT fine-tune** (`work/phase2/train.py` in the campaign branch):
   starting from PR #101's decoder + latents (PR #95 architecture, 229K params),
   fine-tune with straight-through 8-bit per-tensor quantization and a loss of
   SegNet surrogate + β·PoseNet MSE + λ·(differentiable entropy proxy calibrated
   to the #112 range coder within ~2%), λ set to the score's own exchange rate
   (25 bytes⁻¹·orig⁻¹). EMA weights, AdamW (+Muon variants explored). Best:
   EMA@ep40 of the λ=2 run — base archive 176,089 B, CPU 0.19487
   (vs 0.19660 for the same payload without fine-tuning).
2. **Entropy coding**: the decoder/latent streams are coded with PR #112's
   context range coder (`codec_ctx.py`), reused unchanged — we measured it at
   the entropy floor for these streams (offline-compressor slack ≈ 0).
3. **FS1B correction tail** (new, `fs1b.py`): a self-delimiting section after the
   base member carrying (a) a **latent-correction sidecar** — 421 (dim, δ)
   corrections on 397 frame-pairs applied to the decoded fp32 latents before the
   decoder forward (≈454 B, adaptive range-coded, ~1.3 B/correction), and (b) a
   **frame0 pose selector** — per-pair frame0 pixel modes on 245 pairs (concept
   from PR #110's FEC6, re-searched for this payload). Both selected greedily
   against the exact frozen scorers with a mean-perturbation sanity gate.
   A frame1 seg top-up was searched and measured NOT worth shipping.

## Inflate

`inflate.sh <archive_dir> <out_dir> <video_list>` — CPU-pinned, deterministic on
a given machine; deps: `numpy`, `torch`, `constriction` (harness base env; no
network). Chain: FS1B tail parse → #112 ctx decode → #101 tensor reconstruction
→ sidecar corrections on latents → HNeRV decoder in 16-pair batches → bicubic
874×1164 → #98 channel biases → clamp/round → frame0 pose modes → final
clamp/round → uint8 NHWC stream.

`expected_output.sha256` is the canonical decode on the build machine (Apple
M5 Pro, arm64). x86 machines will differ in bicubic LSBs (measured seg wobble
~1.5e-7, see PR #112's README for the same caveat).

## Reproduction

`compress.sh` rebuilds `archive.zip` from the fine-tuned checkpoint + searched
selections (deterministic, seconds, round-trip-asserted). The checkpoint and
selections are produced by the (non-deterministic, ~hours) training/search
pipeline documented in the campaign work tree; see this PR's description for a
summary. No external artifacts are fetched at inflate time.

## Attribution

This submission builds on the published MIT-licensed chain — see
`THIRD_PARTY_NOTICES.md`: PR #95 (@AaronLeslie138, HNeRV decoder architecture +
training recipe), PR #101 (@SajayR, fine-tune + microcodec + sidecar concept),
PR #110 (@adpena, per-pair frame-mode selector concept), PR #112 (@mattneel,
context range coder, reused unchanged).
