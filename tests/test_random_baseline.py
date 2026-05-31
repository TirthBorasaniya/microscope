"""Tests for the randomized-weights sanity check.

The SAE training call and the model/SAE/dataset loaders are mocked, so the test exercises the
real comparison + bootstrap logic without any ML stack or GPU. Verifies the returned dict has
all required keys with the right types.
"""

from __future__ import annotations

from unittest import mock

import pytest

from microscope.sae.trainer import SAEConfig
from microscope.sanity_checks import random_baseline
from microscope.sanity_checks.random_baseline import run_random_baseline

REQUIRED_KEYS = {
    "trained_mean_score",
    "random_mean_score",
    "gap",
    "gap_ci_95",
    "trained_dead_frac",
    "random_dead_frac",
}

TRAINED_STATS = {"mean_score": 0.62, "std_score": 0.18, "dead_feature_fraction": 0.04}
RANDOM_STATS = {"mean_score": 0.21, "std_score": 0.20, "dead_feature_fraction": 0.31}


def test_run_random_baseline_returns_required_keys():
    with (
        mock.patch.object(
            random_baseline, "run_sae_training", side_effect=lambda cfg: cfg.output_dir
        ),
        mock.patch.object(random_baseline, "_load_eval_dataset", return_value=object()),
        mock.patch.object(
            random_baseline, "_load_sae_and_model", return_value=(object(), object())
        ),
        mock.patch.object(
            random_baseline,
            "analyze_feature_sample",
            side_effect=[TRAINED_STATS, RANDOM_STATS],
        ),
    ):
        result = run_random_baseline(SAEConfig(), n_features_to_analyze=500)

    assert set(result.keys()) == REQUIRED_KEYS
    assert result["trained_mean_score"] == TRAINED_STATS["mean_score"]
    assert result["random_mean_score"] == RANDOM_STATS["mean_score"]
    assert result["gap"] == pytest.approx(TRAINED_STATS["mean_score"] - RANDOM_STATS["mean_score"])
    assert result["trained_dead_frac"] == TRAINED_STATS["dead_feature_fraction"]
    assert result["random_dead_frac"] == RANDOM_STATS["dead_feature_fraction"]

    low, high = result["gap_ci_95"]
    assert isinstance(low, float) and isinstance(high, float)
    assert low <= result["gap"] <= high


def test_run_random_baseline_trains_both_models():
    """run_sae_training is called for the trained model and the random-init control."""
    with (
        mock.patch.object(
            random_baseline, "run_sae_training", side_effect=lambda cfg: cfg.output_dir
        ) as fake_train,
        mock.patch.object(random_baseline, "_load_eval_dataset", return_value=object()),
        mock.patch.object(
            random_baseline, "_load_sae_and_model", return_value=(object(), object())
        ),
        mock.patch.object(
            random_baseline,
            "analyze_feature_sample",
            side_effect=[TRAINED_STATS, RANDOM_STATS],
        ),
    ):
        run_random_baseline(SAEConfig(output_dir="checkpoints/sae_trained"))

    assert fake_train.call_count == 2
    configs = [call.args[0] for call in fake_train.call_args_list]
    assert configs[0].o_use_random_model is False
    assert configs[1].o_use_random_model is True
