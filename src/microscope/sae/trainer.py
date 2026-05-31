"""TopK SAE training on Gemma-2-2B residual-stream activations via the SAELens runner.

The runner handles activation collection, normalization, and optimization internally; we only
translate :class:`SAEConfig` into a ``LanguageModelSAERunnerConfig`` and (optionally) swap in a
randomly initialized model for the sanity-check control.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, fields
from pathlib import Path

from microscope import load_yaml_config

# Gemma-2 requires these non-default flags; loading without them silently degrades the model.
GEMMA2_FROM_PRETRAINED_KWARGS: dict = {
    "fold_ln": False,
    "center_writing_weights": False,
    "dtype": "bfloat16",
}


@dataclass
class SAEConfig:
    model_name: str = "google/gemma-2-2b"
    hook_name: str = "blocks.18.hook_resid_post"
    d_in: int = 2304
    expansion_factor: int = 8
    activation_fn_name: str = "topk"
    top_k: int = 64
    train_batch_size_tokens: int = 4096
    n_training_tokens: int = 2_000_000_000
    lr: float = 4e-4
    l1_coefficient: float = 0.0
    dataset_path: str = "Skylion007/openwebtext"
    output_dir: str = "checkpoints/sae_trained"
    o_use_random_model: bool = False  # True -> train on random-init Gemma-2-2B


def _build_random_model(model_name: str):
    """Construct a randomly initialized Gemma-2-2B wrapped for TransformerLens."""
    import torch
    from transformer_lens import HookedTransformer
    from transformers import AutoConfig, AutoModelForCausalLM

    hf_config = AutoConfig.from_pretrained(model_name)
    random_model_hf = AutoModelForCausalLM.from_config(hf_config)
    return HookedTransformer.from_pretrained_no_processing(
        model_name,
        hf_model=random_model_hf,
        fold_ln=False,
        center_writing_weights=False,
        dtype=torch.bfloat16,
    )


def run_sae_training(cfg: SAEConfig) -> str:
    """Instantiate the SAELens runner from ``cfg`` and train; return the checkpoint dir.

    If ``cfg.o_use_random_model`` is set, a randomly initialized Gemma-2-2B is trained instead
    of the pretrained one (the sanity-check control). All hyperparameters come from ``cfg``,
    which is itself populated from a YAML config.
    """
    import torch
    from sae_lens import LanguageModelSAERunnerConfig, SAETrainingRunner

    override_model = _build_random_model(cfg.model_name) if cfg.o_use_random_model else None

    # Field names follow the "Breaking API notes" in CLAUDE.md for SAELens 5.x.
    runner_cfg = LanguageModelSAERunnerConfig(
        model_name=cfg.model_name,
        hook_name=cfg.hook_name,
        d_in=cfg.d_in,
        expansion_factor=cfg.expansion_factor,
        activation_fn_name=cfg.activation_fn_name,
        activation_fn_kwargs={"k": cfg.top_k},
        l1_coefficient=cfg.l1_coefficient,
        lr=cfg.lr,
        train_batch_size_tokens=cfg.train_batch_size_tokens,
        n_training_tokens=cfg.n_training_tokens,
        dataset_path=cfg.dataset_path,
        model_from_pretrained_kwargs=GEMMA2_FROM_PRETRAINED_KWARGS,
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_path=cfg.output_dir,
    )

    runner = SAETrainingRunner(runner_cfg, override_model=override_model)
    runner.run()

    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    return cfg.output_dir


# --------------------------------------------------------------------------------------- CLI


def _str2bool(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


# Per-field parser for CLI overrides, so values still originate from the YAML config by default.
_FIELD_PARSERS = {
    "str": str,
    "int": int,
    "float": float,
    "bool": _str2bool,
}


def _config_from_dict(raw: dict) -> SAEConfig:
    """Build an ``SAEConfig`` from a parsed YAML mapping, ignoring unknown keys."""
    known = {f.name for f in fields(SAEConfig)}
    filtered = {key: value for key, value in raw.items() if key in known}
    return SAEConfig(**filtered)


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a TopK SAE on Gemma-2-2B activations.")
    parser.add_argument("--config", required=True, help="Path to a YAML SAE config.")
    for field in fields(SAEConfig):
        type_name = field.type if isinstance(field.type, str) else field.type.__name__
        parser.add_argument(
            f"--{field.name}",
            type=_FIELD_PARSERS.get(type_name, str),
            default=None,
            help=f"Override config value for {field.name}.",
        )
    return parser


def main(argv: list[str] | None = None) -> str:
    args = _build_argparser().parse_args(argv)
    cfg = _config_from_dict(load_yaml_config(args.config))

    # Apply only the overrides the user actually passed on the command line.
    for field in fields(SAEConfig):
        override = getattr(args, field.name)
        if override is not None:
            setattr(cfg, field.name, override)

    output_path = run_sae_training(cfg)
    print(output_path)
    return output_path


if __name__ == "__main__":
    main()
