"""Shared pytest fixtures.

Heavy, model-dependent fixtures use ``importorskip`` so the suite skips cleanly in an
environment without the ML stack (torch / transformer-lens) rather than erroring at collection.
"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session")
def gpt2_model():
    """A tiny GPT-2 small HookedTransformer for shape / smoke tests (no GPU required)."""
    pytest.importorskip("torch")
    pytest.importorskip("transformer_lens")
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained("gpt2")
