#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python -m microscope.sae.trainer \
    --config configs/sae_gemma2_2b.yaml \
    --o_use_random_model true \
    --output_dir checkpoints/sae_random
