"""IOI dataset, logit-difference metric, activation patching, and RelP circuit discovery."""

from __future__ import annotations

from .activation_patching import activation_patching
from .ioi_dataset import make_ioi_prompts
from .ioi_metric import logit_difference, mean_logit_difference
from .relp_wrapper import compare_with_activation_patching, discover_circuit

__all__ = [
    "make_ioi_prompts",
    "logit_difference",
    "mean_logit_difference",
    "activation_patching",
    "discover_circuit",
    "compare_with_activation_patching",
]
