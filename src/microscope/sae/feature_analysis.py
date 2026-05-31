"""SAE feature analysis: top-activating examples, auto-interp scoring, and a feature sample.

Auto-interpretability follows the standard two-step protocol: a judge LLM describes the pattern
behind a feature's top-activating examples, then we test whether that description predicts the
feature's activation on held-out examples (Pearson correlation).
"""

from __future__ import annotations

import argparse
import json
import re
from typing import Any

import numpy as np

from microscope import SAE_HOOK_NAME, load_yaml_config

# A feature whose maximum activation across the sample never exceeds this is counted as dead.
DEAD_FEATURE_ACT_EPS = 1e-6
# Fallback confidence when a judge response cannot be parsed and there is no held-out set.
_DEFAULT_CONFIDENCE = 0.0


def _feature_activations(sae, model, text: str, feature_idx: int):
    """Return the per-token activations of ``feature_idx`` for a single text."""
    tokens = model.to_tokens(text)
    _, cache = model.run_with_cache(tokens, names_filter=SAE_HOOK_NAME)
    feature_acts = sae.encode(cache[SAE_HOOK_NAME])  # (1, seq, n_features)
    return feature_acts[0, :, feature_idx]


def get_top_activating_examples(
    sae,
    model,
    dataset,
    feature_idx: int,
    n: int = 20,
) -> list[dict[str, Any]]:
    """Find the ``n`` dataset examples that maximally activate ``feature_idx``.

    Returns a list of dicts ``{'text': str, 'activation': float, 'token_position': int}``
    sorted by descending activation.
    """
    examples: list[dict[str, Any]] = []
    for item in dataset:
        text = item["text"] if isinstance(item, dict) else item
        feature_acts = _feature_activations(sae, model, text, feature_idx)
        max_activation, max_position = feature_acts.max(dim=0)
        examples.append(
            {
                "text": text,
                "activation": float(max_activation.item()),
                "token_position": int(max_position.item()),
            }
        )

    examples.sort(key=lambda example: example["activation"], reverse=True)
    return examples[:n]


def _query_judge(judge_model_name: str, prompt: str) -> str:
    """Run a text-generation judge model on ``prompt`` and return its raw text response."""
    from transformers import pipeline

    generator = pipeline("text-generation", model=judge_model_name)
    response = generator(prompt, return_full_text=False)
    return response[0]["generated_text"]


def _format_examples_block(example_list: list[dict[str, Any]]) -> str:
    return "\n".join(f"- (act={ex['activation']:.3f}) {ex['text']}" for ex in example_list)


def _extract_first_float(text: str) -> float | None:
    match = re.search(r"[-+]?\d*\.?\d+", text)
    return float(match.group()) if match else None


def compute_auto_interp_score(
    top_example_list: list[dict[str, Any]],
    judge_model_name: str = "Qwen/Qwen3-8B-Instruct",
    held_out_example_list: list[dict[str, Any]] | None = None,
) -> float:
    """Auto-interpretability score for a single feature.

    Step 1 prompts the judge with ``top_example_list`` to describe the activation pattern.
    Step 2, if ``held_out_example_list`` is given, asks the judge to predict each held-out
    activation from that description and returns the Pearson r between predicted and actual
    activations. If no held-out set is given, returns the judge's self-reported confidence
    in [0, 1].
    """
    from scipy.stats import pearsonr

    description_prompt = (
        "Below are text examples that strongly activate a neural network feature, with their "
        "activation strengths. Describe, in one sentence, the pattern they share.\n\n"
        f"{_format_examples_block(top_example_list)}\n\nPattern:"
    )
    description = _query_judge(judge_model_name, description_prompt)

    if held_out_example_list is None:
        confidence_prompt = (
            f"Feature description: {description}\n"
            "On a scale from 0 to 1, how confident are you that this description is correct "
            "and specific? Answer with a single number."
        )
        confidence = _extract_first_float(_query_judge(judge_model_name, confidence_prompt))
        return float(confidence) if confidence is not None else _DEFAULT_CONFIDENCE

    predicted: list[float] = []
    actual: list[float] = []
    for example in held_out_example_list:
        predict_prompt = (
            f"Feature description: {description}\n"
            f"Text: {example['text']}\n"
            "Predict this feature's activation strength on this text as a single number."
        )
        prediction = _extract_first_float(_query_judge(judge_model_name, predict_prompt))
        predicted.append(prediction if prediction is not None else 0.0)
        actual.append(float(example["activation"]))

    correlation, _ = pearsonr(predicted, actual)
    return float(correlation)


def analyze_feature_sample(
    sae,
    model,
    dataset,
    n_features: int = 500,
    seed: int = 42,
) -> dict[str, float]:
    """Sample ``n_features`` features, score each, and aggregate.

    Returns ``{'mean_score', 'std_score', 'dead_feature_fraction'}``. A feature is dead if its
    maximum activation across the sampled top examples does not exceed ``DEAD_FEATURE_ACT_EPS``.
    """
    rng = np.random.default_rng(seed)
    total_features = sae.cfg.d_sae
    sample_size = min(n_features, total_features)
    feature_indices = rng.choice(total_features, size=sample_size, replace=False)

    scores: list[float] = []
    dead_count = 0
    for feature_idx in feature_indices:
        top_examples = get_top_activating_examples(sae, model, dataset, int(feature_idx))
        peak_activation = max((ex["activation"] for ex in top_examples), default=0.0)
        if peak_activation <= DEAD_FEATURE_ACT_EPS:
            dead_count += 1
            continue
        scores.append(compute_auto_interp_score(top_examples))

    scores_array = np.asarray(scores, dtype=float)
    return {
        "mean_score": float(scores_array.mean()) if scores_array.size else 0.0,
        "std_score": float(scores_array.std()) if scores_array.size else 0.0,
        "dead_feature_fraction": float(dead_count / sample_size),
    }


# --------------------------------------------------------------------------------------- CLI


def _load_sae(sae_path: str):
    from sae_lens import SAE

    return SAE.load_from_pretrained(sae_path)


def _load_model(model_name: str):
    import torch
    from transformer_lens import HookedTransformer

    return HookedTransformer.from_pretrained(
        model_name,
        fold_ln=False,
        center_writing_weights=False,
        dtype=torch.bfloat16,
    )


def _load_eval_dataset(dataset_path: str):
    from datasets import load_dataset

    return load_dataset(dataset_path, split="train", streaming=True)


def main(argv: list[str] | None = None) -> dict[str, float]:
    parser = argparse.ArgumentParser(description="Analyze a sample of SAE features.")
    parser.add_argument("--config", required=True, help="Path to the SAE YAML config.")
    parser.add_argument("--sae_path", required=True, help="Path to the trained SAE checkpoint.")
    args = parser.parse_args(argv)

    config = load_yaml_config(args.config)
    sae = _load_sae(args.sae_path)
    model = _load_model(config["model_name"])
    dataset = _load_eval_dataset(config["dataset_path"])

    stats = analyze_feature_sample(sae, model, dataset)
    print(json.dumps(stats, indent=2))
    return stats


if __name__ == "__main__":
    main()
