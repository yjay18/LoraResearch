"""
diff.py - Differential analysis between base and adapted SAE feature activations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .sae_decode import SparseFeatureActs


@dataclass
class DomainFeatureStats:
    """Aggregated per-feature statistics for one model configuration."""

    adapter: str
    prompt_ids: np.ndarray
    prompt_domains: list[str]
    total_tokens: int
    num_prompts: int
    d_sae: int
    token_active_counts: np.ndarray
    activation_sums: np.ndarray
    prompt_active_counts: np.ndarray
    prompt_presence: np.ndarray
    prompt_activation_sums: np.ndarray

    @property
    def token_frequency(self) -> np.ndarray:
        return self.token_active_counts / self.total_tokens

    @property
    def prompt_frequency(self) -> np.ndarray:
        return self.prompt_active_counts / self.num_prompts

    @property
    def mean_active_magnitude(self) -> np.ndarray:
        return safe_divide(self.activation_sums, self.token_active_counts, fill_value=0.0)

    @property
    def mean_prompt_activation(self) -> np.ndarray:
        return self.prompt_activation_sums.mean(axis=0)

    def prompt_domain_means(self) -> tuple[list[str], np.ndarray]:
        """Mean per-prompt activation mass for each prompt domain and feature."""
        ordered_domains = list(dict.fromkeys(self.prompt_domains))
        matrix = np.zeros((len(ordered_domains), self.d_sae), dtype=np.float32)
        prompt_domains_arr = np.array(self.prompt_domains, dtype=object)

        for idx, domain in enumerate(ordered_domains):
            mask = prompt_domains_arr == domain
            matrix[idx] = self.prompt_activation_sums[mask].mean(axis=0)

        return ordered_domains, matrix


def safe_divide(
    numerator: np.ndarray,
    denominator: np.ndarray,
    fill_value: float = 0.0,
) -> np.ndarray:
    """Elementwise division that fills zero-denominator entries."""
    result = np.full_like(numerator, fill_value, dtype=np.float64)
    valid = denominator != 0
    result[valid] = numerator[valid] / denominator[valid]
    return result


def load_domain_feature_stats(feature_dir: str | Path) -> DomainFeatureStats:
    """Aggregate sparse Day 11 outputs into per-feature summary arrays."""
    feature_dir = Path(feature_dir)
    metadata_path = feature_dir / "metadata.json"
    metadata = pd.read_json(metadata_path)

    prompts = metadata["prompts"]
    adapter = metadata["adapter"][0]
    d_sae = int(metadata["d_sae"][0])
    num_prompts = int(metadata["num_prompts"][0])

    prompt_ids = np.zeros(num_prompts, dtype=np.int32)
    prompt_domains: list[str] = []
    token_active_counts = np.zeros(d_sae, dtype=np.int64)
    activation_sums = np.zeros(d_sae, dtype=np.float64)
    prompt_active_counts = np.zeros(d_sae, dtype=np.int32)
    prompt_presence = np.zeros((num_prompts, d_sae), dtype=bool)
    prompt_activation_sums = np.zeros((num_prompts, d_sae), dtype=np.float32)
    total_tokens = 0

    for prompt_idx, prompt in enumerate(prompts):
        prompt_ids[prompt_idx] = int(prompt["id"])
        prompt_domains.append(prompt["domain"])

        sparse = SparseFeatureActs.load(feature_dir / prompt["file"])
        total_tokens += sparse.seq_len

        values = sparse.values.astype(np.float32, copy=False)
        counts = np.bincount(sparse.feature_indices, minlength=d_sae)
        sums = np.bincount(sparse.feature_indices, weights=values, minlength=d_sae).astype(np.float32, copy=False)
        unique_features = np.unique(sparse.feature_indices)

        token_active_counts += counts
        activation_sums += sums
        prompt_active_counts[unique_features] += 1
        prompt_presence[prompt_idx, unique_features] = True
        prompt_activation_sums[prompt_idx] = sums

    return DomainFeatureStats(
        adapter=adapter,
        prompt_ids=prompt_ids,
        prompt_domains=prompt_domains,
        total_tokens=total_tokens,
        num_prompts=num_prompts,
        d_sae=d_sae,
        token_active_counts=token_active_counts,
        activation_sums=activation_sums,
        prompt_active_counts=prompt_active_counts,
        prompt_presence=prompt_presence,
        prompt_activation_sums=prompt_activation_sums,
    )


def compute_feature_differential(
    base: DomainFeatureStats,
    adapted: DomainFeatureStats,
) -> pd.DataFrame:
    """Compute per-feature differential metrics for one adapter versus base."""
    if base.d_sae != adapted.d_sae:
        raise RuntimeError(f"d_sae mismatch: {base.d_sae} vs {adapted.d_sae}")
    if base.num_prompts != adapted.num_prompts:
        raise RuntimeError(f"num_prompts mismatch: {base.num_prompts} vs {adapted.num_prompts}")
    if not np.array_equal(base.prompt_ids, adapted.prompt_ids):
        raise RuntimeError("Prompt ids do not align between base and adapted feature dumps")

    feature_ids = np.arange(base.d_sae, dtype=np.int32)
    base_token_freq = base.token_frequency
    adapted_token_freq = adapted.token_frequency
    base_prompt_freq = base.prompt_frequency
    adapted_prompt_freq = adapted.prompt_frequency
    base_mean_active = base.mean_active_magnitude
    adapted_mean_active = adapted.mean_active_magnitude
    base_mean_prompt_activation = base.mean_prompt_activation
    adapted_mean_prompt_activation = adapted.mean_prompt_activation

    intersections = np.logical_and(base.prompt_presence, adapted.prompt_presence).sum(axis=0)
    unions = np.logical_or(base.prompt_presence, adapted.prompt_presence).sum(axis=0)
    context_jaccard = safe_divide(intersections, unions, fill_value=1.0)
    prompt_flip_rate = np.logical_xor(base.prompt_presence, adapted.prompt_presence).sum(axis=0) / base.num_prompts

    dot = (base.prompt_activation_sums * adapted.prompt_activation_sums).sum(axis=0)
    base_norm = np.linalg.norm(base.prompt_activation_sums, axis=0)
    adapted_norm = np.linalg.norm(adapted.prompt_activation_sums, axis=0)
    both_zero = (base_norm == 0) & (adapted_norm == 0)
    context_cosine = safe_divide(dot, base_norm * adapted_norm, fill_value=0.0)
    context_cosine[both_zero] = 1.0
    context_shift = 1.0 - context_cosine

    prompt_diff = adapted.prompt_activation_sums - base.prompt_activation_sums
    top_gain_prompt_index = np.argmax(prompt_diff, axis=0)
    top_loss_prompt_index = np.argmin(prompt_diff, axis=0)
    base_prompt_ids = base.prompt_ids
    base_prompt_domains = np.array(base.prompt_domains, dtype=object)

    domain_names, base_domain_means = base.prompt_domain_means()
    _, adapted_domain_means = adapted.prompt_domain_means()
    base_top_domain_idx = np.argmax(base_domain_means, axis=0)
    adapted_top_domain_idx = np.argmax(adapted_domain_means, axis=0)

    data = {
        "feature_id": feature_ids,
        "base_active_token_count": base.token_active_counts,
        "adapted_active_token_count": adapted.token_active_counts,
        "base_prompt_count": base.prompt_active_counts,
        "adapted_prompt_count": adapted.prompt_active_counts,
        "base_token_freq": base_token_freq,
        "adapted_token_freq": adapted_token_freq,
        "delta_token_freq": adapted_token_freq - base_token_freq,
        "base_prompt_freq": base_prompt_freq,
        "adapted_prompt_freq": adapted_prompt_freq,
        "delta_prompt_freq": adapted_prompt_freq - base_prompt_freq,
        "base_mean_active": base_mean_active,
        "adapted_mean_active": adapted_mean_active,
        "delta_mean_active": adapted_mean_active - base_mean_active,
        "base_mean_prompt_activation": base_mean_prompt_activation,
        "adapted_mean_prompt_activation": adapted_mean_prompt_activation,
        "delta_mean_prompt_activation": adapted_mean_prompt_activation - base_mean_prompt_activation,
        "context_jaccard": context_jaccard,
        "context_cosine": context_cosine,
        "context_shift": context_shift,
        "prompt_flip_rate": prompt_flip_rate,
        "base_top_prompt_domain": np.array(domain_names, dtype=object)[base_top_domain_idx],
        "adapted_top_prompt_domain": np.array(domain_names, dtype=object)[adapted_top_domain_idx],
        "domain_switch": np.array(domain_names, dtype=object)[base_top_domain_idx]
        != np.array(domain_names, dtype=object)[adapted_top_domain_idx],
        "top_gain_prompt_id": base_prompt_ids[top_gain_prompt_index],
        "top_gain_prompt_domain": base_prompt_domains[top_gain_prompt_index],
        "top_gain_prompt_delta": prompt_diff[top_gain_prompt_index, feature_ids],
        "top_loss_prompt_id": base_prompt_ids[top_loss_prompt_index],
        "top_loss_prompt_domain": base_prompt_domains[top_loss_prompt_index],
        "top_loss_prompt_delta": prompt_diff[top_loss_prompt_index, feature_ids],
    }

    frame = pd.DataFrame(data)
    frame["abs_delta_token_freq"] = frame["delta_token_freq"].abs()
    frame["abs_delta_mean_active"] = frame["delta_mean_active"].abs()
    frame["abs_context_shift"] = frame["context_shift"].abs()
    return frame.sort_values("feature_id").reset_index(drop=True)


def summarize_feature_differential(adapter: str, frame: pd.DataFrame, top_k: int = 20) -> dict:
    """Create a compact summary for one adapter differential table."""
    top_amplified = frame.nlargest(top_k, "delta_token_freq")[
        ["feature_id", "delta_token_freq", "delta_prompt_freq", "delta_mean_active"]
    ].to_dict(orient="records")
    top_suppressed = frame.nsmallest(top_k, "delta_token_freq")[
        ["feature_id", "delta_token_freq", "delta_prompt_freq", "delta_mean_active"]
    ].to_dict(orient="records")
    top_magnitude_gain = frame.nlargest(top_k, "delta_mean_active")[
        ["feature_id", "delta_mean_active", "delta_token_freq", "delta_prompt_freq"]
    ].to_dict(orient="records")
    top_context_shift = frame.nlargest(top_k, "context_shift")[
        ["feature_id", "context_shift", "context_jaccard", "prompt_flip_rate"]
    ].to_dict(orient="records")

    return {
        "adapter": adapter,
        "feature_count": int(frame.shape[0]),
        "features_with_domain_switch": int(frame["domain_switch"].sum()),
        "mean_delta_token_freq": float(frame["delta_token_freq"].mean()),
        "mean_abs_delta_token_freq": float(frame["abs_delta_token_freq"].mean()),
        "mean_delta_mean_active": float(frame["delta_mean_active"].mean()),
        "mean_context_shift": float(frame["context_shift"].mean()),
        "max_context_shift": float(frame["context_shift"].max()),
        "top_amplified_by_frequency": top_amplified,
        "top_suppressed_by_frequency": top_suppressed,
        "top_amplified_by_magnitude": top_magnitude_gain,
        "top_context_shifted": top_context_shift,
    }
