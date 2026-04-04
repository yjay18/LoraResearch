"""
Day 17/18: Analyze medical, math, safety, and creative adapters plus universal features.
"""

from __future__ import annotations

import json
from pathlib import Path

from lorasurgeon.project import (
    build_adapter_deep_dive,
    find_universal_changed_features,
    render_adapter_markdown,
    render_universal_features_markdown,
)


DAY17_ADAPTER = "medical"
DAY18_ADAPTERS = ["math", "safety", "creative"]
DAY17_MD = Path("results/day17_medical_adapter_deep_dive.md")
DAY17_JSON = Path("results/day17_medical_adapter_deep_dive.json")
DAY18_MD = Path("results/day18_remaining_adapter_analyses.md")
DAY18_JSON = Path("results/day18_remaining_adapter_analyses.json")
DAY18_UNIVERSAL_MD = Path("results/day18_universal_features.md")
DAY18_UNIVERSAL_JSON = Path("results/day18_universal_features.json")


def log(msg: str) -> None:
    print(msg, flush=True)


def render_day18_markdown(reports: list[dict], universal_summary: dict) -> str:
    lines = [
        "# Day 18: Remaining Adapter Analyses",
        "",
        "This report bundles the `math`, `safety`, and `creative` adapter deep dives and the first universal-feature pass.",
        "",
        "## Adapter Snapshot",
        "",
        "| Adapter | Structural Gates | Raw Amplified Match | Non-Structural Amplified Match |",
        "| --- | ---: | ---: | ---: |",
    ]

    for report in reports:
        lines.append(
            f"| {report['adapter']} | {report['structural_gate_count']} | "
            f"{report['amplified_expected_match_rate']:.1%} | {report['amplified_semantic_match_rate']:.1%} |"
        )

    lines.extend(["", "## Per-Adapter Notes", ""])
    for report in reports:
        lines.append(f"### {report['adapter'].title()}")
        for point in report["interpretation_points"]:
            lines.append(f"- {point}")
        lines.append("")

    lines.extend(["## Universal Features", ""])
    lines.append(
        f"- Features changed in all five adapters: {universal_summary['universal_feature_count']}"
    )
    lines.append(
        f"- Dominant families among the top universal features: {universal_summary['top_universal_family_counts']}"
    )
    lines.append("")
    lines.append("### Top Universal Features")
    for row in universal_summary["top_universal_features"][:10]:
        lines.append(
            f"- feature {row['feature_id']}: mean_abs_mass={row['mean_abs_mass']:.6f}, "
            f"majority_family={row['majority_family']}, classes={row['classification_pattern']}"
        )

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    log("=" * 60)
    log("  DAY 17/18: Remaining Adapter Analyses")
    log("=" * 60)

    medical_report = build_adapter_deep_dive(DAY17_ADAPTER)
    DAY17_JSON.write_text(json.dumps(medical_report, indent=2), encoding="utf-8")
    DAY17_MD.write_text(render_adapter_markdown(medical_report), encoding="utf-8")
    log(
        f"  Day 17 medical: structural={medical_report['structural_gate_count']} "
        f"semantic_match={medical_report['amplified_semantic_match_rate']:.3f}"
    )

    day18_reports = []
    for adapter in DAY18_ADAPTERS:
        report = build_adapter_deep_dive(adapter)
        day18_reports.append(report)
        log(
            f"  {adapter}: structural={report['structural_gate_count']} "
            f"semantic_match={report['amplified_semantic_match_rate']:.3f}"
        )

    universal_summary = find_universal_changed_features()
    DAY18_UNIVERSAL_JSON.write_text(json.dumps(universal_summary, indent=2), encoding="utf-8")
    DAY18_UNIVERSAL_MD.write_text(render_universal_features_markdown(universal_summary), encoding="utf-8")

    day18_payload = {
        "reports": day18_reports,
        "universal_summary": universal_summary,
    }
    DAY18_JSON.write_text(json.dumps(day18_payload, indent=2), encoding="utf-8")
    DAY18_MD.write_text(render_day18_markdown(day18_reports, universal_summary), encoding="utf-8")

    log(f"  Universal features changed in all adapters: {universal_summary['universal_feature_count']}")
    log(f"  Saved Day 17 report to {DAY17_MD}")
    log(f"  Saved Day 18 reports to {DAY18_MD} and {DAY18_UNIVERSAL_MD}")


if __name__ == "__main__":
    main()
