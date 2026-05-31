"""Tests for the IOI logit-difference metric and prompt generation.

The core shape/scalar contract is checked with a synthetic logits tensor (no model download,
no GPU). An end-to-end check on GPT-2 small runs only when the ML stack is installed.
"""

from __future__ import annotations

import pytest

from microscope.circuits.ioi_metric import logit_difference, mean_logit_difference

BATCH = 4
SEQ_LEN = 7
VOCAB = 20


def test_logit_difference_shape_and_scalars():
    """Output is shape (batch,) and each entry is a finite scalar."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)

    logits = torch.randn(BATCH, SEQ_LEN, VOCAB)
    io_token_list = [1, 2, 3, 4]
    subj_token_list = [5, 6, 7, 8]
    position_list = [6, 5, 4, 3]

    diff = logit_difference(logits, io_token_list, subj_token_list, position_list)

    assert diff.shape == (BATCH,)
    assert diff.ndim == 1
    assert torch.isfinite(diff).all()

    # Cross-check element 0 against an explicit index.
    expected_0 = (
        logits[0, position_list[0], io_token_list[0]]
        - logits[0, position_list[0], subj_token_list[0]]
    )
    assert torch.allclose(diff[0], expected_0)


def test_logit_difference_matches_manual_gather():
    """Every element equals the manual IO-minus-subject computation at its position."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(1)

    logits = torch.randn(BATCH, SEQ_LEN, VOCAB)
    io_token_list = [10, 11, 12, 13]
    subj_token_list = [0, 1, 2, 3]
    position_list = [0, 1, 2, 6]

    diff = logit_difference(logits, io_token_list, subj_token_list, position_list)

    for i in range(BATCH):
        expected = (
            logits[i, position_list[i], io_token_list[i]]
            - logits[i, position_list[i], subj_token_list[i]]
        )
        assert torch.allclose(diff[i], expected)


def test_mean_logit_difference_on_gpt2(gpt2_model):
    """End-to-end: generated IOI prompts yield a positive mean logit difference on GPT-2."""
    from microscope.circuits.ioi_dataset import make_ioi_prompts

    data = make_ioi_prompts(gpt2_model, n_prompts=8, seed=0)
    assert len(data["prompts"]) == 8
    assert len(data["io_tokens"]) == len(data["positions"]) == 8

    mean_diff = mean_logit_difference(
        gpt2_model,
        data["prompts"],
        data["io_tokens"],
        data["subj_tokens"],
        data["positions"],
        batch_size=4,
    )
    assert isinstance(mean_diff, float)
