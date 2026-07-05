#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
# Rebuild archive.zip for fable_rd_stack.
#
# The archive = (a) base member: the rate-aware-QAT fine-tuned HNeRV decoder +
# latents, quantized 8-bit per-tensor and entropy-coded with PR #112's context
# range coder; (b) FS1B v2 tail: latent-correction sidecar + frame0 pose
# selector, searched offline against the frozen SegNet/PoseNet scorers.
#
# Producing (a)+(b) from scratch is the campaign pipeline (non-deterministic,
# ~hours on an Apple M5 Pro): QAT fine-tune work/phase2/train.py (λ=2, AdamW,
# EMA, ep40 best) from the PR #101 payload, then work/closure/closure_search.py
# (ceiling/round1/round2/sanity) for the tail selections.
#
# This script performs the DETERMINISTIC final composition from those artifacts
# (base archive + selection JSON), asserting a byte-exact round-trip:
#   compress.sh <base_r8_archive.zip> <selection.json> [out.zip]
# With no args it uses the campaign work-tree paths.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

BASE="${1:-$ROOT/work/closure/r8_archive.zip}"
SEL="${2:-$ROOT/work/closure/selection_r8.json}"
OUT="${3:-$HERE/archive.zip}"

if [ ! -f "$BASE" ] || [ ! -f "$SEL" ]; then
  echo "Missing inputs ($BASE / $SEL)." >&2
  echo "Run the campaign pipeline first (see header), or fetch the shipped archive from the release." >&2
  exit 1
fi

python "$HERE/fs1b_compress.py" "$BASE" "$SEL" "$OUT" --base-kind 0
shasum -a 256 "$OUT"
echo "Rebuilt $OUT ($(stat -f%z "$OUT" 2>/dev/null || stat -c%s "$OUT") bytes)"
