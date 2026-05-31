"""Tests for the SAE trainer.

Config construction is verified without the ML stack. The actual training smoke test (a tiny
100k-token run that must leave a checkpoint behind) requires SAELens + a GPU and skips otherwise.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from microscope import load_yaml_config
from microscope.sae.trainer import SAEConfig, _config_from_dict, main, run_sae_training

CONFIG_PATH = "configs/sae_gemma2_2b.yaml"
SMOKE_TRAINING_TOKENS = 100_000


def test_saeconfig_defaults_match_architecture():
    cfg = SAEConfig()
    assert cfg.d_in == 2304
    assert cfg.expansion_factor == 8
    assert cfg.top_k == 64
    assert cfg.activation_fn_name == "topk"
    assert cfg.l1_coefficient == 0.0
    assert cfg.hook_name == "blocks.18.hook_resid_post"
    assert cfg.o_use_random_model is False


def test_config_from_yaml_roundtrip():
    """The shipped YAML populates an SAEConfig with the documented values."""
    raw = load_yaml_config(CONFIG_PATH)
    cfg = _config_from_dict(raw)
    assert cfg.model_name == "google/gemma-2-2b"
    assert cfg.n_training_tokens == 2_000_000_000
    assert cfg.dataset_path == "Skylion007/openwebtext"
    assert cfg.output_dir == "checkpoints/sae_trained"


def test_cli_override_applied(monkeypatch):
    """CLI flags override YAML values, and run_sae_training receives the merged config."""
    captured = {}

    def fake_run(cfg: SAEConfig) -> str:
        captured["cfg"] = cfg
        return cfg.output_dir

    monkeypatch.setattr("microscope.sae.trainer.run_sae_training", fake_run)
    main(
        [
            "--config",
            CONFIG_PATH,
            "--o_use_random_model",
            "true",
            "--output_dir",
            "checkpoints/sae_random",
        ]
    )
    assert captured["cfg"].o_use_random_model is True
    assert captured["cfg"].output_dir == "checkpoints/sae_random"


def test_run_sae_training_smoke(tmp_path):
    """Tiny end-to-end run leaves a checkpoint directory; requires SAELens + GPU."""
    pytest.importorskip("sae_lens")
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("SAE training smoke test requires a GPU")

    cfg = SAEConfig(
        n_training_tokens=SMOKE_TRAINING_TOKENS,
        output_dir=str(tmp_path / "sae_smoke"),
    )
    output_path = run_sae_training(cfg)
    assert Path(output_path).exists()
