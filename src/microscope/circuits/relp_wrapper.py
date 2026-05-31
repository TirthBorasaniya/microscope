"""RelP (Relevance Patching) circuit discovery on the IOI task, plus an activation-patching
comparison.

RelP is *not* a standalone ``relp`` package. It ships as a fork of TransformerLens
(``vendor/relp/TransformerLens``) that adds Layer-wise Relevance Propagation: enabling
``model.cfg.use_lrp = True`` makes the backward pass propagate LRP relevance instead of raw
gradients. "Relevance Patching" is then attribution patching computed with those relevance
coefficients (two forward passes + one backward pass)::

    attribution[node] = corrupted_relevance[node] * (clean_act[node] - corrupted_act[node])

This module mirrors the upstream ``demos/lrp_patching.ipynb`` exactly: it loads the forked
TransformerLens, enables LRP, caches forward activations and backward relevance on a clean and a
corrupted IOI run, forms the attribution cache, reduces it to per-head ``(layer, head)`` scores,
thresholds them into a circuit, and scores faithfulness (precision / recall / F1) against the
canonical Wang et al. 2022 IOI circuit. The activation-patching baseline reuses the fork's
``transformer_lens.patching`` helpers.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Callable

from microscope import load_yaml_config
from microscope.utils.hooks import attn_z_name

# The RelP fork installs as ``transformer_lens`` from this directory (see its README:
# "cd RelP/TransformerLens && pip install -e ."). Inserted at the front of sys.path so the
# LRP-enabled fork shadows any pip-installed transformer-lens before it is first imported.
RELP_TL_PATH = "vendor/relp/TransformerLens"
RESULTS_CIRCUIT_PATH = "results/ioi_circuit.json"

# LRP propagation rules, per the RelP README / demo default.
LRP_RULES: list[str] = ["LN-rule", "Identity-rule", "Half-rule"]


# Forward/backward caches skip the qkv "_input" hooks, matching the upstream demo helper.
def _not_qkv_input(name: str) -> bool:
    return "_input" not in name


# Canonical IOI circuit attention heads (Wang et al. 2022, GPT-2 small) used as the
# faithfulness ground truth: name movers, backup/negative name movers, S-inhibition, induction,
# duplicate-token, and previous-token heads.
IOI_CIRCUIT_HEADS: list[tuple[int, int]] = [
    # Name mover heads
    (9, 9),
    (9, 6),
    (10, 0),
    # Negative name mover heads
    (10, 7),
    (11, 10),
    # Backup name mover heads
    (9, 0),
    (9, 7),
    (10, 1),
    (10, 2),
    (10, 6),
    (10, 10),
    (11, 2),
    (11, 9),
    # S-inhibition heads
    (7, 3),
    (7, 9),
    (8, 6),
    (8, 10),
    # Induction heads
    (5, 5),
    (5, 8),
    (5, 9),
    (6, 9),
    # Duplicate-token heads
    (0, 1),
    (0, 10),
    (3, 0),
    # Previous-token heads
    (2, 2),
    (4, 11),
]


def _ensure_relp_on_path() -> None:
    """Put the forked TransformerLens at the front of ``sys.path``.

    Raises an informative error if the submodule is not present, so the failure points at the
    vendored dependency rather than a bare ``ModuleNotFoundError``.
    """
    if not os.path.isdir(os.path.join(RELP_TL_PATH, "transformer_lens")):
        raise FileNotFoundError(
            f"RelP's forked TransformerLens was not found at '{RELP_TL_PATH}'. Run "
            "`git submodule update --init --recursive` to fetch vendor/relp."
        )
    if sys.path[:1] != [RELP_TL_PATH]:
        # Drop any earlier occurrence so the fork wins, then prepend it.
        while RELP_TL_PATH in sys.path:
            sys.path.remove(RELP_TL_PATH)
        sys.path.insert(0, RELP_TL_PATH)


def _slice_dataset(ioi_dataset_dict: dict, n_prompts: int) -> tuple[list, list, list, list]:
    return (
        ioi_dataset_dict["prompts"][:n_prompts],
        ioi_dataset_dict["io_tokens"][:n_prompts],
        ioi_dataset_dict["subj_tokens"][:n_prompts],
        ioi_dataset_dict["positions"][:n_prompts],
    )


def _make_clean_and_corrupted_tokens(model, prompts: list[str], io_tokens, subj_tokens):
    """Build clean tokens and a corrupted counterpart that swaps each prompt's IO and subject.

    Swapping the indirect-object and subject token ids throughout a prompt flips which name is
    the giver, so the clean answer becomes wrong -- the standard IOI corruption, expressed at the
    token level so it needs no extra dataset fields.
    """

    clean_tokens = model.to_tokens(prompts, padding_side="right")
    corrupted_tokens = clean_tokens.clone()
    for row, (io_id, subj_id) in enumerate(zip(io_tokens, subj_tokens)):
        io_positions = clean_tokens[row] == io_id
        subj_positions = clean_tokens[row] == subj_id
        corrupted_tokens[row][io_positions] = subj_id
        corrupted_tokens[row][subj_positions] = io_id
    return clean_tokens, corrupted_tokens


def _raw_ioi_metric(io_tokens, subj_tokens, positions) -> Callable:
    """Mean IOI logit difference at the per-prompt answer positions."""
    from microscope.circuits.ioi_metric import logit_difference

    def raw(logits):
        return logit_difference(logits, io_tokens, subj_tokens, positions).mean()

    return raw


def _normalized_metric(raw: Callable, clean_baseline: float, corrupted_baseline: float) -> Callable:
    """Normalize so the clean run scores 1.0 and the corrupted run scores 0.0."""
    denominator = clean_baseline - corrupted_baseline
    if denominator == 0:
        denominator = 1.0

    def metric(logits):
        return (raw(logits) - corrupted_baseline) / denominator

    return metric


def _cache_fwd_and_bwd(model, tokens, metric: Callable):
    """Run ``metric(model(tokens))`` and cache forward activations + backward relevance.

    Mirrors the upstream ``get_cache_fwd_and_bwd`` helper. With ``cfg.use_lrp = True`` the cached
    "gradients" are LRP relevance coefficients.
    """
    import transformer_lens.utils as tl_utils  # noqa: F401 - ensures fork is importable
    from transformer_lens import ActivationCache

    model.reset_hooks()
    forward_cache: dict = {}
    relevance_cache: dict = {}

    def forward_hook(act, hook):
        forward_cache[hook.name] = act.detach()

    def backward_hook(act, hook):
        relevance_cache[hook.name] = act.detach()

    model.add_hook(_not_qkv_input, forward_hook, "fwd")
    model.add_hook(_not_qkv_input, backward_hook, "bwd")

    value = metric(model(tokens))
    value.backward()
    model.reset_hooks()

    return (
        value.item(),
        ActivationCache(forward_cache, model),
        ActivationCache(relevance_cache, model),
    )


def _relp_head_scores(model, clean_tokens, corrupted_tokens, metric: Callable):
    """Per-head ``(n_layers, n_heads)`` relevance-patching attribution scores.

    Attribution uses corrupted relevance times the clean-minus-corrupted activation delta on the
    per-head attention output (``hook_z``), summed over batch, position, and head dimension.
    """
    import torch

    _, clean_cache, _ = _cache_fwd_and_bwd(model, clean_tokens, metric)
    _, corrupted_cache, corrupted_relevance = _cache_fwd_and_bwd(model, corrupted_tokens, metric)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    scores = torch.zeros(n_layers, n_heads)
    for layer in range(n_layers):
        hook_name = attn_z_name(layer)
        attribution = corrupted_relevance[hook_name] * (
            clean_cache[hook_name] - corrupted_cache[hook_name]
        )  # (batch, pos, head, d_head)
        scores[layer] = attribution.sum(dim=(0, 1, 3))
    return scores


def _scores_to_nodes(scores, threshold: float) -> list[tuple[int, int]]:
    """Threshold a ``(n_layers, n_heads)`` score tensor (normalized by its max magnitude)."""
    magnitudes = scores.abs()
    peak = magnitudes.max()
    if peak > 0:
        magnitudes = magnitudes / peak
    selected = (magnitudes >= threshold).nonzero(as_tuple=False)
    return [(int(layer), int(head)) for layer, head in selected.tolist()]


def _node_prf1(
    predicted_nodes: list[tuple[int, int]],
    reference_nodes: list[tuple[int, int]],
) -> tuple[float, float, float]:
    """Precision, recall, and F1 of a predicted node set against a reference node set."""
    predicted = set(predicted_nodes)
    reference = set(reference_nodes)
    if not predicted and not reference:
        return 1.0, 1.0, 1.0
    true_positives = len(predicted & reference)
    precision = true_positives / len(predicted) if predicted else 0.0
    recall = true_positives / len(reference) if reference else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return f1, precision, recall


def _activation_patch_head_nodes(
    model,
    ioi_dataset_dict: dict,
    threshold: float,
    n_prompts: int,
) -> list[tuple[int, int]]:
    """Activation-patching baseline: per-head importance via the fork's patching helpers."""
    import torch
    import transformer_lens.patching as patching

    prompts, io_tokens, subj_tokens, positions = _slice_dataset(ioi_dataset_dict, n_prompts)
    clean_tokens, corrupted_tokens = _make_clean_and_corrupted_tokens(
        model, prompts, io_tokens, subj_tokens
    )

    raw = _raw_ioi_metric(io_tokens, subj_tokens, positions)
    with torch.no_grad():
        clean_logits, clean_cache = model.run_with_cache(clean_tokens)
        corrupted_logits = model(corrupted_tokens)
    metric = _normalized_metric(raw, raw(clean_logits).item(), raw(corrupted_logits).item())

    # [patch_type, layer, head]; index 0 is per-head attention output patching.
    head_out_scores = patching.get_act_patch_attn_head_all_pos_every(
        model, corrupted_tokens, clean_cache, metric
    )[0]
    return _scores_to_nodes(head_out_scores, threshold)


def discover_circuit(
    model,
    ioi_dataset_dict: dict,
    threshold: float = 0.1,
    n_prompts: int = 1000,
) -> dict[str, Any]:
    """Run RelP relevance-patching circuit discovery on the IOI task and save the circuit.

    The model must be the LRP-enabled forked TransformerLens (see :func:`make_lrp_model`).
    Computes per-head relevance-patching attribution, thresholds it into ``(layer, head)`` nodes,
    scores faithfulness (precision / recall / F1) against the canonical IOI circuit, writes
    ``results/ioi_circuit.json``, and returns the data-contract dict.
    """
    _ensure_relp_on_path()
    import torch

    prompts, io_tokens, subj_tokens, positions = _slice_dataset(ioi_dataset_dict, n_prompts)
    clean_tokens, corrupted_tokens = _make_clean_and_corrupted_tokens(
        model, prompts, io_tokens, subj_tokens
    )

    raw = _raw_ioi_metric(io_tokens, subj_tokens, positions)
    with torch.no_grad():
        clean_baseline = raw(model(clean_tokens)).item()
        corrupted_baseline = raw(model(corrupted_tokens)).item()
    metric = _normalized_metric(raw, clean_baseline, corrupted_baseline)

    scores = _relp_head_scores(model, clean_tokens, corrupted_tokens, metric)
    nodes = _scores_to_nodes(scores, threshold)
    f1, precision, recall = _node_prf1(nodes, IOI_CIRCUIT_HEADS)

    circuit = {
        "nodes": nodes,
        "faithfulness_f1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "n_nodes": len(nodes),
        "threshold": threshold,
    }

    Path(RESULTS_CIRCUIT_PATH).parent.mkdir(parents=True, exist_ok=True)
    serializable = {**circuit, "nodes": [list(node) for node in nodes]}
    with open(RESULTS_CIRCUIT_PATH, "w", encoding="utf-8") as handle:
        json.dump(serializable, handle, indent=2)

    return circuit


def compare_with_activation_patching(
    model,
    ioi_dataset_dict: dict,
    circuit_dict: dict,
) -> dict[str, float]:
    """Compare the RelP circuit against a standard activation-patching baseline.

    Both circuits are scored by faithfulness F1 against the canonical IOI circuit, at the same
    threshold the RelP circuit used.
    """
    _ensure_relp_on_path()

    threshold = circuit_dict["threshold"]
    patching_nodes = _activation_patch_head_nodes(
        model, ioi_dataset_dict, threshold, len(ioi_dataset_dict["prompts"])
    )
    patching_f1, _, _ = _node_prf1(patching_nodes, IOI_CIRCUIT_HEADS)

    return {
        "relp_faithfulness_f1": float(circuit_dict["faithfulness_f1"]),
        "patching_faithfulness_f1": float(patching_f1),
        "relp_n_nodes": int(circuit_dict["n_nodes"]),
        "patching_n_nodes": len(patching_nodes),
    }


# --------------------------------------------------------------------------------------- CLI


def make_lrp_model(model_name: str, lrp_rules: list[str] | None = None):
    """Load ``model_name`` with the forked TransformerLens and enable LRP for relevance patching.

    Sets the per-head / split hooks the relevance-patching pipeline needs, exactly as the
    upstream ``lrp_patching`` demo does.
    """
    _ensure_relp_on_path()
    import torch
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        model_name,
        fold_ln=False,
        center_writing_weights=False,
        dtype=torch.bfloat16,
    )
    model.cfg.use_lrp = True
    model.cfg.LRP_rules = lrp_rules if lrp_rules is not None else LRP_RULES
    model.set_use_attn_result(True)
    model.set_use_hook_mlp_in(True)
    model.set_use_attn_in(True)
    model.set_use_split_qkv_input(True)
    return model


def main(argv: list[str] | None = None) -> dict[str, Any]:
    import argparse

    from microscope.circuits.ioi_dataset import make_ioi_prompts

    parser = argparse.ArgumentParser(description="Discover the IOI circuit with RelP.")
    parser.add_argument("--config", required=True, help="Path to the circuit-discovery YAML.")
    parser.add_argument("--sae_path", required=True, help="Path to the trained SAE checkpoint.")
    parser.add_argument(
        "--model_name",
        default="google/gemma-2-2b",
        help="Model to run circuit discovery on (use gpt2-small to reproduce the RelP demo).",
    )
    args = parser.parse_args(argv)

    config = load_yaml_config(args.config)
    model = make_lrp_model(args.model_name)
    ioi_dataset_dict = make_ioi_prompts(model, n_prompts=config["n_prompts"])

    circuit = discover_circuit(
        model,
        ioi_dataset_dict,
        threshold=config["threshold"],
        n_prompts=config["n_prompts"],
    )
    comparison = compare_with_activation_patching(model, ioi_dataset_dict, circuit)
    print(json.dumps({**circuit, "nodes": [list(node) for node in circuit["nodes"]]}, indent=2))
    print(json.dumps(comparison, indent=2))
    return circuit


if __name__ == "__main__":
    main()
