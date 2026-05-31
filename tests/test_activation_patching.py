"""Smoke test for activation patching on GPT-2 small.

Verifies the output is a non-empty dict keyed by (layer, position) int tuples with float scores.
Skips cleanly when the ML stack is not installed.
"""

from __future__ import annotations


from microscope.circuits.activation_patching import activation_patching


def _toy_dataset(model, prompts: list[str], io_names: list[str], subj_names: list[str]) -> dict:
    """Build a small IOI-style dict with matched-length prompts for patching."""
    tokens = model.to_tokens(prompts, padding_side="right")
    seq_len = tokens.shape[1]
    return {
        "prompts": prompts,
        "io_tokens": [model.to_single_token(" " + name) for name in io_names],
        "subj_tokens": [model.to_single_token(" " + name) for name in subj_names],
        "positions": [seq_len - 1] * len(prompts),
    }


def test_activation_patching_returns_nonempty_dict(gpt2_model):
    clean = _toy_dataset(
        gpt2_model,
        prompts=[
            "When John and Mary went to the store, John gave a gift to",
            "When Tom and Paul went to the store, Tom gave a gift to",
        ],
        io_names=["Mary", "Paul"],
        subj_names=["John", "Tom"],
    )
    # Corrupted run swaps the giver so the residual stream carries the wrong subject signal.
    corrupted = _toy_dataset(
        gpt2_model,
        prompts=[
            "When John and Mary went to the store, Mary gave a gift to",
            "When Tom and Paul went to the store, Paul gave a gift to",
        ],
        io_names=["Mary", "Paul"],
        subj_names=["John", "Tom"],
    )

    scores = activation_patching(gpt2_model, clean, corrupted)

    assert isinstance(scores, dict)
    assert len(scores) > 0
    key, value = next(iter(scores.items()))
    assert isinstance(key, tuple) and len(key) == 2
    assert all(isinstance(k, int) for k in key)
    assert isinstance(value, float)
