"""Randomized-weights sanity check.

Runs the identical SAE pipeline on a randomly initialized Gemma-2-2B and compares auto-interp
scores against the SAE trained on the real model. A genuine interpretability signal should show
a positive trained-minus-random gap whose bootstrap 95% CI excludes zero.

The model/SAE/dataset loaders are factored into small helpers so the heavy I/O can be mocked in
tests (the SAE training call in particular, per the test requirements).
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from microscope.sae.feature_analysis import analyze_feature_sample
from microscope.sae.trainer import SAEConfig, run_sae_training

# Bootstrap settings for the trained-minus-random gap confidence interval.
BOOTSTRAP_N_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
CI_LOWER_PERCENTILE = 2.5
CI_UPPER_PERCENTILE = 97.5


def _random_output_dir(trained_output_dir: str) -> str:
    """Derive the random-model checkpoint dir from the trained one."""
    return f"{trained_output_dir.rstrip('/')}_random"


def _load_eval_dataset(dataset_path: str):
    """Load the streaming evaluation dataset used for feature analysis."""
    from datasets import load_dataset

    return load_dataset(dataset_path, split="train", streaming=True)


def _load_sae_and_model(sae_path: str, model_name: str, use_random_model: bool):
    """Load a trained SAE and its backing model (real or random-init)."""
    import torch
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    sae = SAE.load_from_pretrained(sae_path)
    if use_random_model:
        from transformers import AutoConfig, AutoModelForCausalLM

        hf_config = AutoConfig.from_pretrained(model_name)
        random_model_hf = AutoModelForCausalLM.from_config(hf_config)
        model = HookedTransformer.from_pretrained_no_processing(
            model_name,
            hf_model=random_model_hf,
            fold_ln=False,
            center_writing_weights=False,
            dtype=torch.bfloat16,
        )
    else:
        model = HookedTransformer.from_pretrained(
            model_name,
            fold_ln=False,
            center_writing_weights=False,
            dtype=torch.bfloat16,
        )
    return sae, model


def _bootstrap_gap_ci(
    trained_stats: dict[str, float],
    random_stats: dict[str, float],
    n_features: int,
) -> tuple[float, float]:
    """Parametric bootstrap 95% CI for the trained-minus-random mean-score gap.

    Resamples ``n_features`` per-feature scores for each model from a normal distribution with
    the observed mean and std, recomputes the gap of means, and returns the requested
    percentiles. Uses only the aggregate statistics returned by ``analyze_feature_sample``.
    """
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    trained_means = rng.normal(
        trained_stats["mean_score"],
        trained_stats["std_score"],
        size=(BOOTSTRAP_N_RESAMPLES, n_features),
    ).mean(axis=1)
    random_means = rng.normal(
        random_stats["mean_score"],
        random_stats["std_score"],
        size=(BOOTSTRAP_N_RESAMPLES, n_features),
    ).mean(axis=1)
    gaps = trained_means - random_means
    return (
        float(np.percentile(gaps, CI_LOWER_PERCENTILE)),
        float(np.percentile(gaps, CI_UPPER_PERCENTILE)),
    )


def run_random_baseline(
    trained_sae_config: SAEConfig,
    n_features_to_analyze: int = 500,
) -> dict[str, Any]:
    """Run the identical SAE pipeline on a random-init Gemma-2-2B and compare to the trained SAE.

    Returns a dict with keys ``trained_mean_score``, ``random_mean_score``, ``gap``,
    ``gap_ci_95``, ``trained_dead_frac``, ``random_dead_frac`` (see the module spec).
    """
    trained_path = run_sae_training(trained_sae_config)
    random_config = replace(
        trained_sae_config,
        o_use_random_model=True,
        output_dir=_random_output_dir(trained_sae_config.output_dir),
    )
    random_path = run_sae_training(random_config)

    dataset = _load_eval_dataset(trained_sae_config.dataset_path)
    trained_sae, trained_model = _load_sae_and_model(
        trained_path, trained_sae_config.model_name, use_random_model=False
    )
    random_sae, random_model = _load_sae_and_model(
        random_path, trained_sae_config.model_name, use_random_model=True
    )

    trained_stats = analyze_feature_sample(
        trained_sae, trained_model, dataset, n_features=n_features_to_analyze
    )
    random_stats = analyze_feature_sample(
        random_sae, random_model, dataset, n_features=n_features_to_analyze
    )

    gap = trained_stats["mean_score"] - random_stats["mean_score"]
    gap_ci_95 = _bootstrap_gap_ci(trained_stats, random_stats, n_features_to_analyze)

    return {
        "trained_mean_score": trained_stats["mean_score"],
        "random_mean_score": random_stats["mean_score"],
        "gap": gap,
        "gap_ci_95": gap_ci_95,
        "trained_dead_frac": trained_stats["dead_feature_fraction"],
        "random_dead_frac": random_stats["dead_feature_fraction"],
    }
