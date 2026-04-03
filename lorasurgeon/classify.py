"""
classify.py - Thresholded feature classification over differential SAE metrics.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


CLASS_ORDER = [
    "amplified",
    "suppressed",
    "newly_activated",
    "killed",
    "context_shifted",
    "unchanged",
]


def to_native(value):
    """Convert NumPy scalar types into plain Python scalars for JSON output."""
    if isinstance(value, np.generic):
        return value.item()
    return value


@dataclass
class ClassificationThresholds:
    """Empirical thresholds used to classify one adapter's features."""

    effect_quantile: float
    context_quantile: float
    min_present_prompts: int
    min_context_flip_rate: float
    abs_delta_token_freq_threshold: float
    abs_delta_mean_prompt_activation_threshold: float
    context_shift_threshold: float

    def to_dict(self) -> dict:
        return asdict(self)


def empirical_threshold(
    values: pd.Series | np.ndarray,
    quantile: float,
    floor: float = 0.0,
) -> float:
    """Return a quantile-based threshold with a floor for degenerate inputs."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return float(floor)
    return float(max(np.quantile(arr, quantile), floor))


def derive_classification_thresholds(
    frame: pd.DataFrame,
    effect_quantile: float = 0.95,
    context_quantile: float = 0.95,
    min_present_prompts: int = 5,
    min_context_flip_rate: float = 0.03,
) -> ClassificationThresholds:
    """Learn empirical significance thresholds from one adapter's differential table."""
    context_pool = frame.loc[frame["prompt_flip_rate"] >= min_context_flip_rate, "context_shift"]
    if context_pool.empty:
        context_pool = frame["context_shift"]

    return ClassificationThresholds(
        effect_quantile=effect_quantile,
        context_quantile=context_quantile,
        min_present_prompts=min_present_prompts,
        min_context_flip_rate=min_context_flip_rate,
        abs_delta_token_freq_threshold=empirical_threshold(
            frame["delta_token_freq"].abs(),
            quantile=effect_quantile,
            floor=0.0,
        ),
        abs_delta_mean_prompt_activation_threshold=empirical_threshold(
            frame["delta_mean_prompt_activation"].abs(),
            quantile=effect_quantile,
            floor=0.0,
        ),
        context_shift_threshold=empirical_threshold(
            context_pool,
            quantile=context_quantile,
            floor=0.0,
        ),
    )


def classify_features(
    frame: pd.DataFrame,
    thresholds: ClassificationThresholds,
) -> pd.DataFrame:
    """Assign one Day 14 class to every feature for one adapter."""
    classified = frame.copy()
    min_present = thresholds.min_present_prompts

    freq_denom = thresholds.abs_delta_token_freq_threshold or 1.0
    mass_denom = thresholds.abs_delta_mean_prompt_activation_threshold or 1.0

    classified["significant_frequency_shift"] = (
        classified["delta_token_freq"].abs() >= thresholds.abs_delta_token_freq_threshold
    )
    classified["significant_magnitude_shift"] = (
        classified["delta_mean_prompt_activation"].abs()
        >= thresholds.abs_delta_mean_prompt_activation_threshold
    )
    classified["significant_context_shift"] = (
        (classified["prompt_flip_rate"] >= thresholds.min_context_flip_rate)
        & (classified["context_shift"] >= thresholds.context_shift_threshold)
    )

    classified["amplification_score"] = (
        np.clip(classified["delta_token_freq"] / freq_denom, a_min=0.0, a_max=None)
        + np.clip(classified["delta_mean_prompt_activation"] / mass_denom, a_min=0.0, a_max=None)
    )
    classified["suppression_score"] = (
        np.clip(-classified["delta_token_freq"] / freq_denom, a_min=0.0, a_max=None)
        + np.clip(-classified["delta_mean_prompt_activation"] / mass_denom, a_min=0.0, a_max=None)
    )
    classified["context_shift_score"] = np.where(
        classified["significant_context_shift"],
        classified["context_shift"] / (thresholds.context_shift_threshold or 1.0),
        0.0,
    )

    is_new = (
        (classified["base_prompt_count"] < min_present)
        & (classified["adapted_prompt_count"] >= min_present)
    )
    is_killed = (
        (classified["adapted_prompt_count"] < min_present)
        & (classified["base_prompt_count"] >= min_present)
    )

    is_amplified = (
        ~is_new
        & ~is_killed
        & (classified["amplification_score"] >= 1.0)
        & (classified["amplification_score"] > classified["suppression_score"])
    )
    is_suppressed = (
        ~is_new
        & ~is_killed
        & (classified["suppression_score"] >= 1.0)
        & (classified["suppression_score"] > classified["amplification_score"])
    )
    is_context_shifted = (
        ~is_new
        & ~is_killed
        & ~is_amplified
        & ~is_suppressed
        & classified["significant_context_shift"]
    )

    classification = np.full(classified.shape[0], "unchanged", dtype=object)
    classification[is_context_shifted.to_numpy()] = "context_shifted"
    classification[is_suppressed.to_numpy()] = "suppressed"
    classification[is_amplified.to_numpy()] = "amplified"
    classification[is_killed.to_numpy()] = "killed"
    classification[is_new.to_numpy()] = "newly_activated"
    classified["classification"] = classification

    classified["classification_score"] = 0.0
    classified.loc[classified["classification"] == "amplified", "classification_score"] = classified[
        "amplification_score"
    ]
    classified.loc[classified["classification"] == "suppressed", "classification_score"] = classified[
        "suppression_score"
    ]
    classified.loc[classified["classification"] == "context_shifted", "classification_score"] = classified[
        "context_shift_score"
    ]
    classified.loc[classified["classification"] == "newly_activated", "classification_score"] = (
        classified["adapted_prompt_count"] - classified["base_prompt_count"]
    )
    classified.loc[classified["classification"] == "killed", "classification_score"] = (
        classified["base_prompt_count"] - classified["adapted_prompt_count"]
    )

    return classified.sort_values("feature_id").reset_index(drop=True)


def top_examples_by_class(classified: pd.DataFrame, top_k: int = 10) -> dict[str, list[dict]]:
    """Collect compact per-class examples for reporting."""
    example_specs = {
        "amplified": (
            "classification_score",
            [
                "feature_id",
                "classification_score",
                "delta_token_freq",
                "delta_mean_prompt_activation",
                "delta_prompt_freq",
                "adapted_top_prompt_domain",
                "domain_switch",
            ],
        ),
        "suppressed": (
            "classification_score",
            [
                "feature_id",
                "classification_score",
                "delta_token_freq",
                "delta_mean_prompt_activation",
                "delta_prompt_freq",
                "base_top_prompt_domain",
                "domain_switch",
            ],
        ),
        "newly_activated": (
            "classification_score",
            [
                "feature_id",
                "classification_score",
                "base_prompt_count",
                "adapted_prompt_count",
                "delta_token_freq",
                "adapted_top_prompt_domain",
            ],
        ),
        "killed": (
            "classification_score",
            [
                "feature_id",
                "classification_score",
                "base_prompt_count",
                "adapted_prompt_count",
                "delta_token_freq",
                "base_top_prompt_domain",
            ],
        ),
        "context_shifted": (
            "classification_score",
            [
                "feature_id",
                "classification_score",
                "context_shift",
                "prompt_flip_rate",
                "base_top_prompt_domain",
                "adapted_top_prompt_domain",
                "domain_switch",
            ],
        ),
    }

    examples: dict[str, list[dict]] = {}
    for label, (sort_col, columns) in example_specs.items():
        subset = classified.loc[classified["classification"] == label]
        rows = subset.nlargest(top_k, sort_col)[columns].to_dict(orient="records")
        examples[label] = [{key: to_native(value) for key, value in row.items()} for row in rows]
    return examples


def summarize_classifications(
    adapter: str,
    classified: pd.DataFrame,
    thresholds: ClassificationThresholds,
    top_k: int = 10,
) -> dict:
    """Create a compact summary JSON payload for one adapter's classifications."""
    counts = (
        classified["classification"]
        .value_counts()
        .reindex(CLASS_ORDER, fill_value=0)
        .astype(int)
    )
    count_dict = {label: int(count) for label, count in counts.items()}
    fractions = {label: float(count / classified.shape[0]) for label, count in count_dict.items()}

    return {
        "adapter": adapter,
        "feature_count": int(classified.shape[0]),
        "thresholds": thresholds.to_dict(),
        "class_counts": count_dict,
        "class_fractions": fractions,
        "top_examples": top_examples_by_class(classified, top_k=top_k),
    }
