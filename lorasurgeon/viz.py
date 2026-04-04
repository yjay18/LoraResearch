"""
viz.py - Publication-style plotting utilities for LoRASurgeon analyses.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .fingerprint import compute_change_rank


ADAPTERS = ["code", "medical", "math", "safety", "creative"]
DOMAINS = ["general", "code", "medical", "math", "safety", "creative"]
CLASS_COLORS = {
    "amplified": "#1b9e77",
    "suppressed": "#d95f02",
    "newly_activated": "#7570b3",
    "killed": "#e7298a",
    "context_shifted": "#66a61e",
    "unchanged": "#d9d9d9",
}


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.bbox": "tight",
        }
    )


def plot_feature_domain_heatmap(
    label_payload: dict,
    out_path: str | Path,
    top_n: int = 20,
) -> None:
    set_plot_style()
    rows = sorted(label_payload["labels"], key=lambda row: row["change_rank_score"], reverse=True)[:top_n]

    matrix = []
    labels = []
    for row in rows:
        side_key = f"{row['primary_side']}_domain_mass"
        domain_mass = row[side_key]
        values = np.array([float(domain_mass.get(domain, 0.0)) for domain in DOMAINS], dtype=np.float32)
        total = values.sum()
        if total > 0:
            values = values / total
        matrix.append(values)
        labels.append(f"{row['feature_id']} {row['classification'][0].upper()}")

    matrix = np.array(matrix, dtype=np.float32)
    fig, ax = plt.subplots(figsize=(7.5, 0.42 * top_n + 2.0))
    im = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_title(f"{label_payload['adapter'].title()} Top-{top_n} Feature Domain Heatmap")
    ax.set_xlabel("Prompt domain")
    ax.set_xticks(range(len(DOMAINS)))
    ax.set_xticklabels(DOMAINS, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Normalized activation mass share")
    fig.savefig(out_path)
    plt.close(fig)


def plot_volcano_style(
    classified_csv_path: str | Path,
    summary_json_path: str | Path,
    out_path: str | Path,
    adapter: str,
) -> None:
    set_plot_style()
    frame = pd.read_csv(classified_csv_path)
    summary = load_json(summary_json_path)
    ranked = compute_change_rank(frame, summary["thresholds"])

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for classification, color in CLASS_COLORS.items():
        subset = ranked.loc[ranked["classification"] == classification]
        alpha = 0.20 if classification == "unchanged" else 0.70
        size = 10 if classification == "unchanged" else 18
        ax.scatter(
            subset["delta_mean_prompt_activation"],
            subset["change_rank_score"],
            s=size,
            c=color,
            alpha=alpha,
            edgecolors="none",
            label=classification.replace("_", " "),
        )

    top_rows = ranked.loc[ranked["classification"] != "unchanged"].nlargest(10, "change_rank_score")
    for _, row in top_rows.iterrows():
        ax.text(
            row["delta_mean_prompt_activation"],
            row["change_rank_score"] + 0.3,
            str(int(row["feature_id"])),
            fontsize=8,
            color="#222222",
        )

    ax.axvline(0.0, color="#666666", linewidth=1.0, linestyle="--")
    ax.set_title(f"{adapter.title()} Volcano-Style Change Plot")
    ax.set_xlabel("Delta mean prompt activation")
    ax.set_ylabel("Normalized change score")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.savefig(out_path)
    plt.close(fig)


def plot_top_features_bar(
    label_payload: dict,
    out_path: str | Path,
    top_n: int = 20,
) -> None:
    set_plot_style()
    rows = sorted(label_payload["labels"], key=lambda row: row["change_rank_score"], reverse=True)[:top_n]
    rows = list(reversed(rows))

    labels = [f"{row['feature_id']} {row['label_family']}" for row in rows]
    scores = [row["change_rank_score"] for row in rows]
    colors = [CLASS_COLORS[row["classification"]] for row in rows]

    fig, ax = plt.subplots(figsize=(10, 0.38 * top_n + 2.0))
    ax.barh(range(len(rows)), scores, color=colors)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Composite change rank score")
    ax.set_title(f"{label_payload['adapter'].title()} Top-{top_n} Changed Features")
    fig.savefig(out_path)
    plt.close(fig)


def plot_universal_features_heatmap(
    universal_summary: dict,
    classification_root: str | Path,
    out_path: str | Path,
    top_n: int = 25,
) -> None:
    set_plot_style()
    classification_root = Path(classification_root)
    features = [row["feature_id"] for row in universal_summary["top_universal_features"][:top_n]]

    matrix = np.zeros((len(ADAPTERS), len(features)), dtype=np.float32)
    for adapter_idx, adapter in enumerate(ADAPTERS):
        frame = pd.read_csv(
            classification_root / f"{adapter}_classified_features.csv",
            usecols=["feature_id", "delta_mean_prompt_activation"],
        ).set_index("feature_id")
        for feature_idx, feature_id in enumerate(features):
            matrix[adapter_idx, feature_idx] = float(frame.loc[feature_id, "delta_mean_prompt_activation"])

    fig, ax = plt.subplots(figsize=(0.42 * len(features) + 2.5, 4.0))
    vmax = np.abs(matrix).max() or 1.0
    im = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.set_title("Universal Feature Shift Heatmap")
    ax.set_xlabel("Feature id")
    ax.set_ylabel("Adapter")
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels([str(feature_id) for feature_id in features], rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(ADAPTERS)))
    ax.set_yticklabels(ADAPTERS)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Delta mean prompt activation")
    fig.savefig(out_path)
    plt.close(fig)


def plot_similarity_heatmap(
    matrix: pd.DataFrame,
    out_path: str | Path,
    title: str,
    cmap: str = "coolwarm",
    value_range: tuple[float, float] = (-1.0, 1.0),
) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    values = matrix.to_numpy(dtype=np.float32)
    im = ax.imshow(values, cmap=cmap, vmin=value_range[0], vmax=value_range[1])
    ax.set_title(title)
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{values[row_idx, col_idx]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="#111111",
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Similarity")
    fig.savefig(out_path)
    plt.close(fig)


def plot_embedding_projection(
    coordinates: pd.DataFrame,
    out_path: str | Path,
    title: str,
) -> None:
    set_plot_style()
    fig, ax = plt.subplots(figsize=(7.2, 5.4))

    for _, row in coordinates.iterrows():
        label = row["label"]
        color = "#444444" if label == "base" else "#c44e52"
        marker = "X" if label == "base" else "o"
        size = 130 if label == "base" else 95
        ax.scatter(row["x"], row["y"], s=size, c=color, marker=marker, edgecolors="white", linewidths=0.8)
        ax.text(row["x"] + 0.02, row["y"] + 0.02, label, fontsize=9, color="#111111")

    ax.axhline(0.0, color="#bbbbbb", linewidth=0.8)
    ax.axvline(0.0, color="#bbbbbb", linewidth=0.8)
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    fig.savefig(out_path)
    plt.close(fig)
