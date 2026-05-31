# microscope

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![TransformerLens](https://img.shields.io/badge/TransformerLens-2.8.0-blue.svg)](https://github.com/TransformerLensOrg/TransformerLens)
[![SAELens](https://img.shields.io/badge/SAELens-5.3.0-orange.svg)](https://github.com/jbloomAus/SAELens)

Mechanistic interpretability of `google/gemma-2-2b` via TopK sparse autoencoder training,
RelP circuit discovery on the indirect object identification (IOI) task, and a randomized-weights
sanity-check control. The sanity check runs the identical SAE pipeline on a randomly initialized
model of the same architecture to quantify how much of the observed "interpretability"
is genuine rather than an artifact of the training procedure.

---

## Architecture

```
Gemma-2-2B  (26 layers, hidden_size=2304)
        │
        │  residual stream activations at layer 18
        ▼
TopK SAE  (18,432 features, k=64, 2B training tokens)
        │
        ├── feature analysis  →  top-activating examples, auto-interp scoring
        └── circuit discovery →  IOI task, RelP + activation patching comparison

Sanity check:
  Randomly initialized Gemma-2-2B  →  same SAE config  →  same feature analysis
  Gap (trained − random) = honest measure of what the SAE learns
```

---

## Requirements

- NVIDIA GPU, ≥ 12 GB VRAM (Gemma-2-2B in BF16 ≈ 6 GB)
- CUDA 12.4, Python 3.11, Ubuntu 22.04+

---

## Installation

```bash
git clone --recurse-submodules https://github.com/<your-username>/microscope
cd microscope

bash scripts/setup_env.sh
# edit .env — set HF_TOKEN and WANDB_PROJECT
```

---

## Running experiments

```bash
source .venv/bin/activate

# SAE training on trained model (~14 h on H200)
bash scripts/train_sae.sh

# SAE training on randomized model (~14 h on H200, for sanity check)
bash scripts/train_sae_random_baseline.sh

# Feature interpretability analysis
bash scripts/analyze_features.sh

# IOI circuit discovery with RelP
bash scripts/discover_circuits.sh
# outputs: results/ioi_circuit.json, results/circuit_comparison.json
```

---

## Using the SAE

```python
from transformer_lens import HookedTransformer
from saelens import SAE
import torch

model = HookedTransformer.from_pretrained(
    "google/gemma-2-2b",
    fold_ln=False,
    center_writing_weights=False,
    dtype=torch.bfloat16,
)

sae, _, _ = SAE.from_pretrained(
    release="<your-username>/microscope-gemma2-2b-sae-l18",
    sae_id="blocks.18.hook_resid_post",
)

tokens = model.to_tokens("The Eiffel Tower is in")
_, cache = model.run_with_cache(tokens)
activations = cache["blocks.18.hook_resid_post"]

feature_acts = sae.encode(activations)
top_features = feature_acts.topk(10, dim=-1).indices
print("Top active features:", top_features)
```

---

## Published artifacts

| Artifact | Description |
|---|---|
| `<your-username>/microscope-gemma2-2b-sae-l18` | Trained SAE weights (layer 18) |
| `<your-username>/microscope-gemma2-2b-sae-l18-random` | Randomized-baseline SAE |
| `results/ioi_circuit.json` | IOI circuit nodes and faithfulness metrics |

---

## Design decisions

See [DECISIONS.md](DECISIONS.md) for: layer selection, TopK vs ReLU, expansion factor,
RelP vs ACDC, and an honest interpretation of the sanity-check gap.

---

## Security

See [SECURITY.md](SECURITY.md). Notebooks are committed with cleared output.
All credentials are environment-variable-only.

---

## Citation

```bibtex
@misc{microscope2025,
  title        = {microscope: SAE Feature Discovery and Circuit Analysis on Gemma-2-2B},
  year         = {2025},
  howpublished = {\url{https://github.com/TirthBorasaniya/microscope}},
  note         = {SAE training from Bricken et al. 2023; RelP (NeurIPS 2025);
                  sanity checks from Heap et al. (arXiv 2501.17727)}
}
```
