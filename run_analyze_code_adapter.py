"""
Day 16: Deep-dive analysis for the code adapter using Day 15 labels.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path


LABELS_PATH = Path("results/labels/code_feature_labels.json")
OUTPUT_JSON = Path("results/day16_code_adapter_deep_dive.json")
OUTPUT_MD = Path("results/day16_code_adapter_deep_dive.md")


def log(msg: str) -> None:
    print(msg, flush=True)


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


def render_markdown(report: dict) -> str:
    lines = [
        "# Day 16: Code Adapter Deep Dive",
        "",
        "This analysis uses the Day 15 labeled top-100 changed features for the `code` adapter.",
        "",
        "## Top-100 Mix",
        "",
        f"- Class mix: {report['classification_counts']}",
        f"- Label families: {report['label_family_counts']}",
        f"- Amplified features matching code expectations: {report['amplified_expected_match_rate']:.1%}",
        f"- Amplified non-structural features matching code expectations: {report['amplified_semantic_match_rate']:.1%}",
        f"- Suppressed features that are clearly non-code: {report['suppressed_non_code_rate']:.1%}",
        f"- Suppressed non-structural features that are clearly non-code: {report['suppressed_semantic_non_code_rate']:.1%}",
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


def main() -> None:
    payload = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
    rows = payload["labels"]

    amplified = [row for row in rows if row["classification"] == "amplified"]
    suppressed = [row for row in rows if row["classification"] == "suppressed"]
    structural = [row for row in rows if row["label_family"] == "bos_boundary"]
    amplified_semantic = [row for row in amplified if row["label_family"] != "bos_boundary"]
    suppressed_semantic = [row for row in suppressed if row["label_family"] != "bos_boundary"]
    code_semantic = [row for row in amplified if row["matches_code_expectation"]]
    code_semantic_non_structural = [row for row in amplified_semantic if row["matches_code_expectation"]]
    non_code_suppressed = [row for row in suppressed if row["is_non_code_semantic"]]
    non_code_suppressed_semantic = [row for row in suppressed_semantic if row["is_non_code_semantic"]]

    interpretation_points = [
        "The code adapter's strongest changes are still suppression-heavy, which matches the broader Day 14 result that adapters mostly reweight existing features rather than creating many new ones.",
        (
            "Expected code semantics do appear in the amplified set: docstrings, implementation framing, "
            "and code-layout features all recur among the top amplified labels."
        ),
        (
            "The match is only partial. Several top amplified features are not pure Python syntax features; "
            "they instead look like generic instructional scaffolding or even legal/exam-style reasoning."
        ),
        (
            "Many of the largest `newly_activated` and `killed` features are BOS / structural gates rather than "
            "rich semantic concepts. That suggests some of the adapter effect is a global routing change, not just "
            "domain-specific concept insertion."
        ),
        (
            "The suppressed set contains many clearly non-code families, which is the expected direction for a code "
            "adapter, but the presence of math- and general-reasoning features among the amplified set suggests the "
            "underlying prompt format still matters a lot."
        ),
    ]

    report = {
        "adapter": "code",
        "top_k": payload["top_k"],
        "classification_counts": classification_counts(rows),
        "label_family_counts": family_counts(rows),
        "amplified_expected_match_rate": len(code_semantic) / len(amplified) if amplified else 0.0,
        "amplified_semantic_match_rate": (
            len(code_semantic_non_structural) / len(amplified_semantic) if amplified_semantic else 0.0
        ),
        "suppressed_non_code_rate": len(non_code_suppressed) / len(suppressed) if suppressed else 0.0,
        "suppressed_semantic_non_code_rate": (
            len(non_code_suppressed_semantic) / len(suppressed_semantic) if suppressed_semantic else 0.0
        ),
        "structural_gate_count": len(structural),
        "interpretation_points": interpretation_points,
        "representatives": {
            "Amplified": top_rows(rows, "amplified"),
            "Suppressed": top_rows(rows, "suppressed"),
            "Newly Activated": top_rows(rows, "newly_activated"),
            "Killed": top_rows(rows, "killed"),
        },
    }

    OUTPUT_JSON.write_text(json.dumps(report, indent=2), encoding="utf-8")
    OUTPUT_MD.write_text(render_markdown(report), encoding="utf-8")

    log("=" * 60)
    log("  DAY 16: Code Adapter Deep Dive")
    log("=" * 60)
    log(
        f"  amplified_expected_match_rate={report['amplified_expected_match_rate']:.3f} "
        f"amplified_semantic_match_rate={report['amplified_semantic_match_rate']:.3f} "
        f"suppressed_non_code_rate={report['suppressed_non_code_rate']:.3f} "
        f"suppressed_semantic_non_code_rate={report['suppressed_semantic_non_code_rate']:.3f} "
        f"structural_gate_count={report['structural_gate_count']}"
    )
    log(f"  Saved markdown report to {OUTPUT_MD}")


if __name__ == "__main__":
    main()
