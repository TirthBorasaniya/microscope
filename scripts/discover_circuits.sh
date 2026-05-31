#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate
python -m microscope.circuits.relp_wrapper \
    --config configs/circuit_discovery.yaml \
    --sae_path checkpoints/sae_trained
