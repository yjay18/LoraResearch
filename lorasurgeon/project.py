"""
project.py - Reusable adapter and cross-adapter analysis helpers.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd


ADAPTERS = ["code", "medical", "math", "safety", "creative"]
EXPECTED_FAMILIES = {
    "code": {"code_docstring_example", "code_function_definition", "code_layout_whitespace"},
    "medical": {"medical_clinical"},
    "math": {"math_word_problem"},
    "safety": {"safety_sensitive"},
    "creative": {"creative_writing"},
}
ADAPTER_DESCRIPTIONS = {
    "code": "code semantics",
    "medical": "medical / clinical semantics",
    "math": "math / arithmetic semantics",
    "safety": "safety / refusal semantics",
    "creative": "creative-writing semantics",
}


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def family_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(row["label_family"] for row in rows)
    return {family: counts[family] for family in sorted(counts)}


def classification_counts(rows: list[dict]) -> dict[str, int]:
    counts = Counter(row["classification"] for row in rows)
    return {label: counts[label] for label in sorted(counts)}


def top_rows(rows: list[dict], classification: str, limit: int = 5) -> list[dict]:
    subset = [row for row in rows if row["classification"] == classification]
    subset.sort(key=lambda row: row["change_rank_score"], reverse=True)
    return subset[:limit]


def rate_for_family(
    rows: list[dict],
    expected_families: set[str],
    classification: str = "amplified",
    exclude_structural: bool = False,
    invert: bool = False,
) -> float:
    subset = [row for row in rows if row["classification"] == classification]
    if exclude_structural:
        subset = [row for row in subset if row["label_family"] != "bos_boundary"]
    if not subset:
        return 0.0

    if invert:
        hits = [row for row in subset if row["label_family"] not in expected_families and row["is_non_code_semantic"]]
    else:
        hits = [row for row in subset if row["label_family"] in expected_families]
    return len(hits) / len(subset)


def build_adapter_interpretation(adapter: str, rows: list[dict], report: dict) -> list[str]:
    expected_semantics = ADAPTER_DESCRIPTIONS[adapter]
    points = []

    if report["structural_gate_count"] >= 50:
        points.append(
            f"The {adapter} adapter is structurally dominated: {report['structural_gate_count']} of the top 100 "
            "changed features are BOS / boundary gates."
        )
    else:
        points.append(
            f"The {adapter} adapter has a meaningful semantic slice, although structural features still occupy "
            f"{report['structural_gate_count']} of the top 100 positions."
        )

    semantic_rate = report["amplified_semantic_match_rate"]
    if semantic_rate >= 0.5:
        points.append(
            f"After removing structural BOS gates, the amplified subset is meaningfully aligned with "
            f"{expected_semantics} ({semantic_rate:.1%} semantic match)."
        )
    elif semantic_rate >= 0.25:
        points.append(
            f"There is only partial semantic alignment with {expected_semantics} once structural gates are removed "
            f"({semantic_rate:.1%} semantic match)."
        )
    else:
        points.append(
            f"The amplified semantic subset barely matches the expected {expected_semantics} "
            f"({semantic_rate:.1%} semantic match), so the adapter is not cleanly separable by its top changed features."
        )

    top_families = sorted(report["label_family_counts"].items(), key=lambda item: (-item[1], item[0]))
    non_structural_families = [item for item in top_families if item[0] != "bos_boundary"][:3]
    if non_structural_families:
        points.append(
            "The strongest non-structural families are "
            + ", ".join(f"{family} ({count})" for family, count in non_structural_families)
            + "."
        )

    if adapter == "medical":
        points.append(
            "Compared with the kind of clinical lexical structure you would expect from MIMIC-style text, the medical "
            "adapter's biggest shifts still look more structural and mixed-domain than strongly clinical."
        )
    elif adapter == "math":
        points.append(
            "The math adapter does surface some arithmetic-word-problem features, but they do not dominate the top-changed set."
        )
    elif adapter == "safety":
        points.append(
            "The safety adapter is the weakest semantic match of the group: refusal / safety-specific labels are rare, "
            "which suggests structural routing is overshadowing policy semantics in the top changes."
        )
    elif adapter == "creative":
        points.append(
            "The creative adapter does not produce a strong narrative-specific signature in the top-changed set; "
            "structural and mixed prompt-format effects are larger."
        )

    return points


def build_adapter_deep_dive(
    adapter: str,
    label_root: str | Path = "results/labels",
) -> dict:
    payload = load_json(Path(label_root) / f"{adapter}_feature_labels.json")
    rows = payload["labels"]
    expected_families = EXPECTED_FAMILIES[adapter]

    amplified = [row for row in rows if row["classification"] == "amplified"]
    suppressed = [row for row in rows if row["classification"] == "suppressed"]
    structural = [row for row in rows if row["label_family"] == "bos_boundary"]

    report = {
        "adapter": adapter,
        "top_k": payload["top_k"],
        "classification_counts": classification_counts(rows),
        "label_family_counts": family_counts(rows),
        "amplified_expected_match_rate": rate_for_family(rows, expected_families, "amplified", False, False),
        "amplified_semantic_match_rate": rate_for_family(rows, expected_families, "amplified", True, False),
        "suppressed_non_expected_rate": rate_for_family(rows, expected_families, "suppressed", False, True),
        "suppressed_semantic_non_expected_rate": rate_for_family(rows, expected_families, "suppressed", True, True),
        "structural_gate_count": len(structural),
        "representatives": {
            "Amplified": top_rows(rows, "amplified"),
            "Suppressed": top_rows(rows, "suppressed"),
            "Newly Activated": top_rows(rows, "newly_activated"),
            "Killed": top_rows(rows, "killed"),
        },
    }
    report["interpretation_points"] = build_adapter_interpretation(adapter, rows, report)
    return report


def render_adapter_markdown(report: dict) -> str:
    lines = [
        f"# {report['adapter'].title()} Adapter Deep Dive",
        "",
        "This analysis uses the Day 15 labeled top-100 changed features for the adapter.",
        "",
        "## Top-100 Mix",
        "",
        f"- Class mix: {report['classification_counts']}",
        f"- Label families: {report['label_family_counts']}",
        f"- Raw amplified expected-family match: {report['amplified_expected_match_rate']:.1%}",
        f"- Non-structural amplified expected-family match: {report['amplified_semantic_match_rate']:.1%}",
        f"- Raw suppressed non-expected rate: {report['suppressed_non_expected_rate']:.1%}",
        f"- Non-structural suppressed non-expected rate: {report['suppressed_semantic_non_expected_rate']:.1%}",
        f"- Structural BOS-dominated gates in the top 100: {report['structural_gate_count']}",
        "",
        "## Interpretation",
        "",
    ]

    for point in report["interpretation_points"]:
        lines.append(f"- {point}")

    lines.extend(["", "## Representative Features", ""])
    for title, rows in report["representatives"].items():
        lines.append(f"### {title}")
        for row in rows:
            lines.append(
                f"- feature {row['feature_id']}: {row['label']} "
                f"(class={row['classification']}, family={row['label_family']}, "
                f"primary_domain={row['primary_domain']}, top_tokens="
                f"{', '.join(token['token'] for token in row['top_tokens'][:3])})"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def find_universal_changed_features(
    classification_root: str | Path = "results/classification",
    label_root: str | Path = "results/labels",
    top_k: int = 25,
) -> dict:
    classification_root = Path(classification_root)
    label_root = Path(label_root)

    merged = None
    for adapter in ADAPTERS:
        frame = pd.read_csv(
            classification_root / f"{adapter}_classified_features.csv",
            usecols=[
                "feature_id",
                "classification",
                "classification_score",
                "delta_mean_prompt_activation",
                "delta_token_freq",
                "context_shift",
            ],
        ).rename(
            columns={
                "classification": f"{adapter}_classification",
                "classification_score": f"{adapter}_classification_score",
                "delta_mean_prompt_activation": f"{adapter}_delta_mean_prompt_activation",
                "delta_token_freq": f"{adapter}_delta_token_freq",
                "context_shift": f"{adapter}_context_shift",
            }
        )
        merged = frame if merged is None else merged.merge(frame, on="feature_id")

    changed_mask = pd.Series(True, index=merged.index)
    for adapter in ADAPTERS:
        changed_mask &= merged[f"{adapter}_classification"] != "unchanged"

    universal = merged.loc[changed_mask].copy()
    universal["mean_abs_mass"] = sum(
        universal[f"{adapter}_delta_mean_prompt_activation"].abs() for adapter in ADAPTERS
    ) / len(ADAPTERS)
    universal["mean_abs_freq"] = sum(universal[f"{adapter}_delta_token_freq"].abs() for adapter in ADAPTERS) / len(
        ADAPTERS
    )
    universal["max_context_shift"] = universal[[f"{adapter}_context_shift" for adapter in ADAPTERS]].max(axis=1)

    label_votes = {}
    for adapter in ADAPTERS:
        payload = load_json(label_root / f"{adapter}_feature_labels.json")
        for row in payload["labels"]:
            feature_id = int(row["feature_id"])
            label_votes.setdefault(feature_id, Counter())
            label_votes[feature_id][row["label_family"]] += 1

    rows = []
    universal = universal.sort_values(["mean_abs_mass", "mean_abs_freq"], ascending=[False, False])
    for _, row in universal.head(top_k).iterrows():
        feature_id = int(row["feature_id"])
        votes = label_votes.get(feature_id, Counter())
        majority_family, vote_count = ("unlabeled", 0)
        if votes:
            majority_family, vote_count = votes.most_common(1)[0]

        class_pattern = {
            adapter: row[f"{adapter}_classification"]
            for adapter in ADAPTERS
        }
        rows.append(
            {
                "feature_id": feature_id,
                "mean_abs_mass": round(float(row["mean_abs_mass"]), 6),
                "mean_abs_freq": round(float(row["mean_abs_freq"]), 9),
                "max_context_shift": round(float(row["max_context_shift"]), 6),
                "majority_family": majority_family,
                "majority_family_votes": int(vote_count),
                "classification_pattern": class_pattern,
            }
        )

    family_counts_top = Counter(row["majority_family"] for row in rows)
    return {
        "universal_feature_count": int(universal.shape[0]),
        "top_k": top_k,
        "top_universal_features": rows,
        "top_universal_family_counts": {
            family: family_counts_top[family] for family in sorted(family_counts_top)
        },
    }


def render_universal_features_markdown(summary: dict) -> str:
    lines = [
        "# Universal Features Across All Adapters",
        "",
        f"- Features changed in all five adapters: {summary['universal_feature_count']}",
        f"- Top universal-family counts: {summary['top_universal_family_counts']}",
        "",
        "| Feature | Mean Abs Mass Shift | Mean Abs Freq Shift | Majority Family | Votes |",
        "| --- | ---: | ---: | --- | ---: |",
    ]

    for row in summary["top_universal_features"]:
        lines.append(
            f"| {row['feature_id']} | {row['mean_abs_mass']:.6f} | {row['mean_abs_freq']:.9f} | "
            f"{row['majority_family']} | {row['majority_family_votes']} |"
        )

    return "\n".join(lines).rstrip() + "\n"


def cosine_similarity_frame(vectors_by_name: dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(vectors_by_name)
    matrix = np.vstack([np.asarray(vectors_by_name[name], dtype=np.float64) for name in names])
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    normalized = matrix / safe_norms
    similarity = normalized @ normalized.T
    return pd.DataFrame(similarity, index=names, columns=names)


def correlation_similarity_frame(vectors_by_name: dict[str, np.ndarray]) -> pd.DataFrame:
    names = list(vectors_by_name)
    matrix = np.vstack([np.asarray(vectors_by_name[name], dtype=np.float64) for name in names])
    similarity = np.corrcoef(matrix)
    similarity = np.nan_to_num(similarity, nan=0.0)
    return pd.DataFrame(similarity, index=names, columns=names)


def jaccard_similarity_frame(feature_sets_by_name: dict[str, set[int]]) -> pd.DataFrame:
    names = list(feature_sets_by_name)
    matrix = np.zeros((len(names), len(names)), dtype=np.float64)
    for row_idx, left in enumerate(names):
        for col_idx, right in enumerate(names):
            union = feature_sets_by_name[left] | feature_sets_by_name[right]
            intersection = feature_sets_by_name[left] & feature_sets_by_name[right]
            matrix[row_idx, col_idx] = 1.0 if not union else len(intersection) / len(union)
    return pd.DataFrame(matrix, index=names, columns=names)


def upper_triangle_pairs(matrix: pd.DataFrame, largest: bool = True) -> list[dict]:
    rows = []
    names = list(matrix.index)
    for row_idx, left in enumerate(names):
        for col_idx in range(row_idx + 1, len(names)):
            right = names[col_idx]
            rows.append(
                {
                    "left": left,
                    "right": right,
                    "value": float(matrix.iloc[row_idx, col_idx]),
                }
            )
    rows.sort(key=lambda row: row["value"], reverse=largest)
    return rows
