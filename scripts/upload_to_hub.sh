#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate

# Upload a trained SAE checkpoint to the Hugging Face Hub.
# Model weights live on the Hub, never in git (see .gitignore). HF_TOKEN comes from the env.
#
# Usage: scripts/upload_to_hub.sh <local_checkpoint_dir> <hub_repo_id>

LOCAL_DIR="${1:?usage: upload_to_hub.sh <local_checkpoint_dir> <hub_repo_id>}"
REPO_ID="${2:?usage: upload_to_hub.sh <local_checkpoint_dir> <hub_repo_id>}"

: "${HF_TOKEN:?HF_TOKEN must be set in the environment}"

huggingface-cli upload "${REPO_ID}" "${LOCAL_DIR}" --repo-type model --token "${HF_TOKEN}"
