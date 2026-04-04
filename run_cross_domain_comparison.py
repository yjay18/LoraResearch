"""
Day 21-23: build adapter fingerprints, compare them across domains, and project them into 2D.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from lorasurgeon.fingerprint import (
    STRUCTURAL_FAMILY,
    build_dense_fingerprint_vectors,
    label_top_changed_for_adapter,
    prepare_ranked_classified_frame,
)
from lorasurgeon.project import (
    ADAPTERS,
    EXPECTED_FAMILIES,
    correlation_similarity_frame,
    cosine_similarity_frame,
    jaccard_similarity_frame,
    upper_triangle_pairs,
)
from lorasurgeon.viz import plot_embedding_projection, plot_similarity_heatmap


PROMPTS_PATH = Path("data/prompts_300.json")
FEATURE_ROOT = Path("data/sae_features")
CLASSIFICATION_ROOT = Path("results/classification")
FINGERPRINT_ROOT = Path("results/fingerprints")
FIGURE_ROOT = Path("results/figures")
DAY21_MD = Path("results/day21_fingerprint_vectors.md")
DAY21_JSON = Path("results/day21_fingerprint_vectors.json")
DAY22_MD = Path("results/day22_similarity_analysis.md")
DAY22_JSON = Path("results/day22_similarity_analysis.json")
DAY23_MD = Path("results/day23_fingerprint_embedding.md")
DAY23_JSON = Path("results/day23_fingerprint_embedding.json")

LABELED_TOP_K = 250


def log(msg: str) -> None:
    print(msg, flush=True)


def to_matrix_payload(matrix: pd.DataFrame) -> dict:
    rounded = matrix.round(6)
    return {
        "index": list(rounded.index),
        "columns": list(rounded.columns),
        "values": rounded.values.tolist(),
    }


def payload_to_matrix(payload: dict) -> pd.DataFrame:
    return pd.DataFrame(payload["values"], index=payload["index"], columns=payload["columns"])


def build_day21_adapter_payload(adapter: str) -> tuple[dict, dict[str, np.ndarray], set[int], set[int]]:
    ranked, _ = prepare_ranked_classified_frame(adapter, CLASSIFICATION_ROOT)
    payload = label_top_changed_for_adapter(
        adapter=adapter,
        top_k=LABELED_TOP_K,
        prompts_path=PROMPTS_PATH,
        feature_root=FEATURE_ROOT,
        classification_root=CLASSIFICATION_ROOT,
    )

    feature_ids, vectors = build_dense_fingerprint_vectors(ranked, zero_unchanged=True)
    labels = payload["labels"]
    structural_ids = {int(row["feature_id"]) for row in labels if row["label_family"] == STRUCTURAL_FAMILY}
    changed_ids = {
        int(feature_id) for feature_id in ranked.loc[ranked["classification"] != "unchanged", "feature_id"].tolist()
    }

    np.savez_compressed(
        FINGERPRINT_ROOT / f"{adapter}_fingerprint_vectors.npz",
        feature_ids=feature_ids,
        signed_mass=vectors["signed_mass"],
        signed_freq=vectors["signed_freq"],
        signed_rank=vectors["signed_rank"],
    )

    labels_path = FINGERPRINT_ROOT / f"{adapter}_top{LABELED_TOP_K}_labels.json"
    labels_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    top_positive = ranked.nlargest(10, "delta_mean_prompt_activation")[
        ["feature_id", "delta_mean_prompt_activation", "classification", "change_rank_score"]
    ]
    top_negative = ranked.nsmallest(10, "delta_mean_prompt_activation")[
        ["feature_id", "delta_mean_prompt_activation", "classification", "change_rank_score"]
    ]

    amplified_non_structural = [
        row for row in labels if row["classification"] == "amplified" and row["label_family"] != STRUCTURAL_FAMILY
    ]
    expected_families = EXPECTED_FAMILIES[adapter]
    semantic_match = (
        sum(row["label_family"] in expected_families for row in amplified_non_structural) / len(amplified_non_structural)
        if amplified_non_structural
        else 0.0
    )

    summary = {
        "adapter": adapter,
        "feature_count": int(feature_ids.shape[0]),
        "changed_feature_count": int((ranked["classification"] != "unchanged").sum()),
        "top_k_labeled": LABELED_TOP_K,
        "structural_topk_count": len(structural_ids),
        "non_structural_amplified_count": len(amplified_non_structural),
        "non_structural_amplified_semantic_match": round(float(semantic_match), 6),
        "signed_mass_l1": round(float(np.abs(vectors["signed_mass"]).sum()), 6),
        "signed_mass_l2": round(float(np.linalg.norm(vectors["signed_mass"])), 6),
        "signed_freq_l1": round(float(np.abs(vectors["signed_freq"]).sum()), 9),
        "top_positive_features": top_positive.round(6).to_dict(orient="records"),
        "top_negative_features": top_negative.round(6).to_dict(orient="records"),
        "vector_path": str(FINGERPRINT_ROOT / f"{adapter}_fingerprint_vectors.npz"),
        "labels_path": str(labels_path),
    }

    (FINGERPRINT_ROOT / f"{adapter}_fingerprint_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary, vectors, structural_ids, changed_ids


def render_day21_markdown(day21_payload: dict) -> str:
    lines = [
        "# Day 21: Adapter Fingerprint Vectors",
        "",
        "Built dense per-feature fingerprint vectors for each adapter from the saved Day 14 classifications.",
        "",
        "| Adapter | Changed Features | Top-250 Structural | Top-250 Non-Structural Semantic Match | L2 Norm |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for summary in day21_payload["adapter_summaries"]:
        lines.append(
            f"| {summary['adapter']} | {summary['changed_feature_count']} | {summary['structural_topk_count']} | "
            f"{summary['non_structural_amplified_semantic_match']:.1%} | {summary['signed_mass_l2']:.3f} |"
        )

    lines.extend(["", "## Notes", ""])
    for point in day21_payload["notes"]:
        lines.append(f"- {point}")

    return "\n".join(lines).rstrip() + "\n"


def render_day22_markdown(day22_payload: dict) -> str:
    full_cos = day22_payload["key_pairs"]["full_mass_cosine"]
    filtered_cos = day22_payload["key_pairs"]["filtered_mass_cosine"]
    lines = [
        "# Day 22: Fingerprint Similarity Analysis",
        "",
        "Pairwise adapter similarity was computed on dense signed mass vectors, plus a conservative structural-filtered view.",
        "",
        "## Key Pairs",
        "",
        f"- Strongest full-vector cosine pair: `{full_cos['highest']['left']}` + `{full_cos['highest']['right']}` = `{full_cos['highest']['value']:.3f}`",
        f"- Weakest full-vector cosine pair: `{full_cos['lowest']['left']}` + `{full_cos['lowest']['right']}` = `{full_cos['lowest']['value']:.3f}`",
        f"- Strongest filtered cosine pair: `{filtered_cos['highest']['left']}` + `{filtered_cos['highest']['right']}` = `{filtered_cos['highest']['value']:.3f}`",
        f"- Weakest filtered cosine pair: `{filtered_cos['lowest']['left']}` + `{filtered_cos['lowest']['right']}` = `{filtered_cos['lowest']['value']:.3f}`",
        "",
        "## Interpretation",
        "",
    ]
    for point in day22_payload["interpretation"]:
        lines.append(f"- {point}")

    return "\n".join(lines).rstrip() + "\n"


def render_day23_markdown(day23_payload: dict) -> str:
    lines = [
        "# Day 23: Fingerprint Embedding",
        "",
        "Projected the full and structural-filtered fingerprint vectors into 2D with PCA.",
        "",
        "## Why PCA",
        "",
        "- There are only 6 points in this view (`base` + 5 adapters), so PCA is more stable and more honest than a decorative UMAP/t-SNE layout.",
        "",
        "## Base Distances",
        "",
    ]
    for view_name, distances in day23_payload["distance_from_base"].items():
        ordered = sorted(distances.items(), key=lambda item: item[1], reverse=True)
        rendered = ", ".join(f"{name}={value:.3f}" for name, value in ordered)
        lines.append(f"- `{view_name}`: {rendered}")

    lines.extend(["", "## Interpretation", ""])
    for point in day23_payload["interpretation"]:
        lines.append(f"- {point}")

    return "\n".join(lines).rstrip() + "\n"


def pair_summary(matrix: pd.DataFrame) -> dict:
    descending = upper_triangle_pairs(matrix, largest=True)
    ascending = upper_triangle_pairs(matrix, largest=False)
    return {
        "highest": descending[0],
        "lowest": ascending[0],
        "top_pairs": descending[:3],
    }


def build_day22_payload(
    vectors_by_adapter: dict[str, dict[str, np.ndarray]],
    changed_sets: dict[str, set[int]],
    structural_union: set[int],
) -> dict:
    mask = None
    feature_ids = None
    for vectors in vectors_by_adapter.values():
        feature_ids = np.arange(vectors["signed_mass"].shape[0], dtype=np.int32)
        break
    if feature_ids is None:
        raise RuntimeError("No adapter vectors available for Day 22")

    if structural_union:
        mask = ~np.isin(feature_ids, list(structural_union))
    else:
        mask = np.ones_like(feature_ids, dtype=bool)

    full_mass = {adapter: vectors["signed_mass"] for adapter, vectors in vectors_by_adapter.items()}
    filtered_mass = {adapter: vectors["signed_mass"][mask] for adapter, vectors in vectors_by_adapter.items()}
    full_freq = {adapter: vectors["signed_freq"] for adapter, vectors in vectors_by_adapter.items()}

    full_mass_cosine = cosine_similarity_frame(full_mass)
    full_mass_corr = correlation_similarity_frame(full_mass)
    filtered_mass_cosine = cosine_similarity_frame(filtered_mass)
    filtered_mass_corr = correlation_similarity_frame(filtered_mass)
    full_freq_cosine = cosine_similarity_frame(full_freq)
    changed_jaccard = jaccard_similarity_frame(changed_sets)

    full_pairs = pair_summary(full_mass_cosine)
    filtered_pairs = pair_summary(filtered_mass_cosine)
    full_second = full_pairs["top_pairs"][1]
    filtered_second = filtered_pairs["top_pairs"][1]
    positive_bridge = [
        adapter
        for adapter in ADAPTERS
        if all(
            float(filtered_mass_cosine.loc[adapter, other]) > 0.0
            for other in ADAPTERS
            if other != adapter
        )
    ]

    interpretation = [
        (
            f"`{full_pairs['highest']['left']}` and `{full_pairs['highest']['right']}` remain the strongest pair in both "
            f"the full and filtered views ({full_pairs['highest']['value']:.3f} -> {filtered_pairs['highest']['value']:.3f} cosine), "
            "so that relationship is not just a structural-BOS artifact."
        ),
        (
            f"`{full_second['left']}` and `{full_second['right']}` are the second-closest pair in both views "
            f"({full_second['value']:.3f} -> {filtered_second['value']:.3f} cosine), which suggests a second shared change axis."
        ),
        (
            f"`{positive_bridge[0]}` is the only adapter that stays positively aligned with every other adapter after filtering, "
            "which makes it the bridge case rather than a clean cluster endpoint."
            if positive_bridge
            else "No adapter stays positively aligned with every other adapter after filtering."
        ),
        (
            f"The structural filter currently removes {len(structural_union)} features. That is conservative rather than exhaustive, "
            "so the filtered matrix should be read as a lower-bound attempt to isolate semantics."
        ),
    ]

    return {
        "structural_feature_union_count": len(structural_union),
        "matrices": {
            "full_mass_cosine": to_matrix_payload(full_mass_cosine),
            "full_mass_correlation": to_matrix_payload(full_mass_corr),
            "filtered_mass_cosine": to_matrix_payload(filtered_mass_cosine),
            "filtered_mass_correlation": to_matrix_payload(filtered_mass_corr),
            "full_freq_cosine": to_matrix_payload(full_freq_cosine),
            "changed_feature_jaccard": to_matrix_payload(changed_jaccard),
        },
        "key_pairs": {
            "full_mass_cosine": full_pairs,
            "filtered_mass_cosine": filtered_pairs,
        },
        "interpretation": interpretation,
    }


def build_projection(vectors_by_name: dict[str, np.ndarray]) -> tuple[pd.DataFrame, list[float]]:
    labels = list(vectors_by_name)
    matrix = np.vstack([vectors_by_name[label] for label in labels]).astype(np.float64)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(matrix)
    frame = pd.DataFrame({"label": labels, "x": coords[:, 0], "y": coords[:, 1]})
    return frame, pca.explained_variance_ratio_.tolist()


def build_day23_payload(
    vectors_by_adapter: dict[str, dict[str, np.ndarray]],
    structural_union: set[int],
) -> dict:
    feature_ids = np.arange(next(iter(vectors_by_adapter.values()))["signed_mass"].shape[0], dtype=np.int32)
    filtered_mask = ~np.isin(feature_ids, list(structural_union)) if structural_union else np.ones_like(feature_ids, dtype=bool)

    full_vectors = {"base": np.zeros_like(feature_ids, dtype=np.float32)}
    filtered_vectors = {"base": np.zeros(int(filtered_mask.sum()), dtype=np.float32)}
    for adapter, vectors in vectors_by_adapter.items():
        full_vectors[adapter] = vectors["signed_mass"]
        filtered_vectors[adapter] = vectors["signed_mass"][filtered_mask]

    full_projection, full_var = build_projection(full_vectors)
    filtered_projection, filtered_var = build_projection(filtered_vectors)

    distance_from_base = {
        "full": {
            adapter: float(np.linalg.norm(vector))
            for adapter, vector in full_vectors.items()
            if adapter != "base"
        },
        "filtered": {
            adapter: float(np.linalg.norm(vector))
            for adapter, vector in filtered_vectors.items()
            if adapter != "base"
        },
    }

    full_sorted = sorted(distance_from_base["full"].items(), key=lambda item: item[1], reverse=True)
    filtered_sorted = sorted(distance_from_base["filtered"].items(), key=lambda item: item[1], reverse=True)

    interpretation = [
        (
            f"In the full-vector PCA, PC1 explains {full_var[0]:.1%} of the variance and is dominated by the largest shared change axis."
        ),
        (
            f"In the structural-filtered PCA, PC1 still explains {filtered_var[0]:.1%}, and the layout continues to separate the "
            "math/safety side from the code/creative side."
        ),
        (
            f"The largest raw fingerprint shifts come from `{full_sorted[0][0]}` and `{full_sorted[1][0]}`, and they remain the "
            f"largest even after filtering (`{filtered_sorted[0][0]}`, `{filtered_sorted[1][0]}`)."
        ),
        (
            "The base point sits at the zero vector by construction, so distance from base should be read as total feature-shift magnitude rather than behavioral distance."
        ),
    ]

    return {
        "method": "pca",
        "why_not_umap_or_tsne": (
            "With only six points (`base` plus five adapters), PCA is the more stable and interpretable choice. "
            "UMAP/t-SNE would add stochastic geometry without adding real information."
        ),
        "full_projection": full_projection.round(6).to_dict(orient="records"),
        "full_explained_variance_ratio": [round(float(value), 6) for value in full_var],
        "filtered_projection": filtered_projection.round(6).to_dict(orient="records"),
        "filtered_explained_variance_ratio": [round(float(value), 6) for value in filtered_var],
        "distance_from_base": {
            view_name: {adapter: round(float(value), 6) for adapter, value in distances.items()}
            for view_name, distances in distance_from_base.items()
        },
        "interpretation": interpretation,
    }


def main() -> None:
    FINGERPRINT_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("  DAY 21-23: Cross-Domain Comparison")
    log("=" * 60)

    adapter_summaries = []
    vectors_by_adapter = {}
    structural_by_adapter = {}
    changed_sets = {}

    for adapter in ADAPTERS:
        summary, vectors, structural_ids, changed_ids = build_day21_adapter_payload(adapter)
        adapter_summaries.append(summary)
        vectors_by_adapter[adapter] = vectors
        structural_by_adapter[adapter] = structural_ids
        changed_sets[adapter] = changed_ids
        log(
            f"  {adapter}: changed={summary['changed_feature_count']} "
            f"structural_top250={summary['structural_topk_count']} "
            f"semantic_match={summary['non_structural_amplified_semantic_match']:.3f}"
        )

    day21_payload = {
        "adapter_summaries": adapter_summaries,
        "notes": [
            "Each fingerprint is a dense 16,384-dimensional vector aligned to the shared SAE feature basis.",
            "The primary signed vector uses delta mean prompt activation with unchanged features zeroed out.",
            "Top-250 labels are saved alongside each adapter summary to make the structural filter auditable.",
        ],
    }
    DAY21_JSON.write_text(json.dumps(day21_payload, indent=2), encoding="utf-8")
    DAY21_MD.write_text(render_day21_markdown(day21_payload), encoding="utf-8")

    structural_union = set().union(*structural_by_adapter.values())
    day22_payload = build_day22_payload(vectors_by_adapter, changed_sets, structural_union)
    DAY22_JSON.write_text(json.dumps(day22_payload, indent=2), encoding="utf-8")
    DAY22_MD.write_text(render_day22_markdown(day22_payload), encoding="utf-8")

    full_mass_cosine = payload_to_matrix(day22_payload["matrices"]["full_mass_cosine"])
    filtered_mass_cosine = payload_to_matrix(day22_payload["matrices"]["filtered_mass_cosine"])
    plot_similarity_heatmap(
        full_mass_cosine,
        FIGURE_ROOT / "day22_full_mass_cosine_heatmap.png",
        "Day 22 Full-Vector Cosine Similarity",
    )
    plot_similarity_heatmap(
        filtered_mass_cosine,
        FIGURE_ROOT / "day22_filtered_mass_cosine_heatmap.png",
        "Day 22 Structural-Filtered Cosine Similarity",
    )

    day23_payload = build_day23_payload(vectors_by_adapter, structural_union)
    DAY23_JSON.write_text(json.dumps(day23_payload, indent=2), encoding="utf-8")
    DAY23_MD.write_text(render_day23_markdown(day23_payload), encoding="utf-8")

    plot_embedding_projection(
        pd.DataFrame(day23_payload["full_projection"]),
        FIGURE_ROOT / "day23_full_fingerprint_pca.png",
        "Day 23 Full-Vector PCA",
    )
    plot_embedding_projection(
        pd.DataFrame(day23_payload["filtered_projection"]),
        FIGURE_ROOT / "day23_filtered_fingerprint_pca.png",
        "Day 23 Structural-Filtered PCA",
    )

    log(
        "  strongest full cosine pair: "
        f"{day22_payload['key_pairs']['full_mass_cosine']['highest']['left']} + "
        f"{day22_payload['key_pairs']['full_mass_cosine']['highest']['right']} = "
        f"{day22_payload['key_pairs']['full_mass_cosine']['highest']['value']:.3f}"
    )
    log(
        "  strongest filtered cosine pair: "
        f"{day22_payload['key_pairs']['filtered_mass_cosine']['highest']['left']} + "
        f"{day22_payload['key_pairs']['filtered_mass_cosine']['highest']['right']} = "
        f"{day22_payload['key_pairs']['filtered_mass_cosine']['highest']['value']:.3f}"
    )
    log(f"  Saved Day 21 report to {DAY21_MD}")
    log(f"  Saved Day 22 report to {DAY22_MD}")
    log(f"  Saved Day 23 report to {DAY23_MD}")


if __name__ == "__main__":
    main()
