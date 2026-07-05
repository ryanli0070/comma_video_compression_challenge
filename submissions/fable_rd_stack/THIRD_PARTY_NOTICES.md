# Third-party notices

All upstream works are MIT-licensed submissions to this repository
(commaai/comma_video_compression_challenge).

- **PR #95 — `hnerv_muon` (@AaronLeslie138)**: HNeRV decoder architecture
  (`model.py`, byte-compatible) and the training recipe this submission's
  fine-tune derives from. https://github.com/commaai/comma_video_compression_challenge/pull/95
- **PR #101 — `hnerv_ft_microcodec` (@SajayR)**: the fine-tuned decoder +
  latents used as initialization; tensor payload reconstruction (`codec.py`,
  `codec_sidecar.py`, reused); the latent-correction sidecar concept our FS1B
  sidecar generalizes. https://github.com/commaai/comma_video_compression_challenge/pull/101
- **PR #110 — `hnerv_fec6_fixed_huffman_k16` (@adpena)**: the per-pair frame0
  perturbation-selector concept (FEC6) our frame0 pose selector re-searches;
  `frame_selector.py` transform families (reused).
  https://github.com/commaai/comma_video_compression_challenge/pull/110
- **PR #112 — `rhnerv_comma` (@mattneel)**: the context-modeled range coder
  (`codec_ctx.py`, reused unchanged) and the base-member wire format FS1B
  extends. https://github.com/commaai/comma_video_compression_challenge/pull/112
- **constriction** (Bamler; MIT/Apache-2.0): range-coding backend.
- **PyTorch, NumPy**: runtime dependencies (BSD-style licenses).

New code in this submission (MIT, © 2026 Ryan Li): the rate-aware QAT
fine-tuning pipeline, the FS1B v2 tail format (`fs1b.py`, `fs1b_palette.py`),
the sidecar/selector search, and the composed `inflate.py`.
