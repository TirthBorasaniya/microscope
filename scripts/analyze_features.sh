#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python -m microscope.sae.feature_analysis \
    --config configs/sae_gemma2_2b.yaml \
    --sae_path checkpoints/sae_trained
