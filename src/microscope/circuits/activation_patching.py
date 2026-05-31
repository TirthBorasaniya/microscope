"""Standard residual-stream activation patching (Wang et al. 2022).

For every (layer, sequence-position) we patch the clean run's residual stream at
``blocks.{layer}.hook_resid_post`` with the corresponding activation from a corrupted run, then
record how much the IOI logit difference drops. Larger drop => more important location.

Returns ``dict[(layer, position), float]`` of importance scores, as required by the checklist.
"""

from __future__ import annotations

from typing import Any

from microscope.utils.hooks import make_replace_hook, resid_post_name

# Suffix shared by every residual-stream-post hook; used to cache only those activations.
_RESID_POST_SUFFIX = "hook_resid_post"


def activation_patching(
    model,
    clean_dataset_dict: dict[str, Any],
    corrupted_dataset_dict: dict[str, Any],
) -> dict[tuple[int, int], float]:
    """Patch each (layer, position) of the clean run with corrupted activations.

    Parameters
    ----------
    model : HookedTransformer
    clean_dataset_dict, corrupted_dataset_dict : dict
        IOI-style dicts with ``prompts``, ``io_tokens``, ``subj_tokens``, ``positions``. The
        clean dict supplies the answer tokens / positions used to score the metric. Clean and
        corrupted prompt batches must tokenize to the same shape (same count and length).

    Returns
    -------
    dict[(layer, position), float]
        Mean drop in logit difference (clean minus patched) caused by patching that location.
    """
    import torch

    from microscope.circuits.ioi_metric import logit_difference

    io_tokens = clean_dataset_dict["io_tokens"]
    subj_tokens = clean_dataset_dict["subj_tokens"]
    positions = clean_dataset_dict["positions"]

    clean_tokens = model.to_tokens(clean_dataset_dict["prompts"], padding_side="right")
    corrupted_tokens = model.to_tokens(corrupted_dataset_dict["prompts"], padding_side="right")

    n_layers = model.cfg.n_layers
    seq_len = clean_tokens.shape[1]

    with torch.no_grad():
        clean_logits = model(clean_tokens)
        clean_ld = logit_difference(clean_logits, io_tokens, subj_tokens, positions)

        _, corrupted_cache = model.run_with_cache(
            corrupted_tokens,
            names_filter=lambda name: name.endswith(_RESID_POST_SUFFIX),
        )

    importance_scores: dict[tuple[int, int], float] = {}
    for layer in range(n_layers):
        hook_name = resid_post_name(layer)
        corrupted_activation = corrupted_cache[hook_name]
        for position in range(seq_len):
            replace_hook = make_replace_hook(corrupted_activation, position)
            with torch.no_grad():
                patched_logits = model.run_with_hooks(
                    clean_tokens,
                    fwd_hooks=[(hook_name, replace_hook)],
                )
            patched_ld = logit_difference(patched_logits, io_tokens, subj_tokens, positions)
            importance_scores[(layer, position)] = float((clean_ld - patched_ld).mean().item())

    return importance_scores
