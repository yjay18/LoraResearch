"""
Day 14: Classify per-feature adapter effects using thresholded differential metrics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from lorasurgeon.classify import (
    CLASS_ORDER,
    classify_features,
    derive_classification_thresholds,
    summarize_classifications,
)


TARGETS = ["code", "medical", "math", "safety", "creative"]
DIFF_ROOT = Path("results/differential")
OUTPUT_ROOT = Path("results/classification")
SUMMARY_MD = Path("results/day14_feature_classification.md")


def log(msg: str) -> None:
    print(msg, flush=True)


def render_summary_markdown(summaries: list[dict]) -> str:
    """Render a compact Day 14 summary."""
    lines = [
        "# Day 14: Feature Classification",
        "",
        "Each adapter's Day 13 differential table was mapped into one exclusive class per feature.",
        "Empirical significance thresholds use the 95th percentile of effect sizes plus support floors.",
        "",
        "| Adapter | Amplified | Suppressed | New | Killed | Context Shifted | Unchanged |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for summary in summaries:
        counts = summary["class_counts"]
        lines.append(
            f"| {summary['adapter']} | {counts['amplified']} | {counts['suppressed']} | "
            f"{counts['newly_activated']} | {counts['killed']} | "
            f"{counts['context_shifted']} | {counts['unchanged']} |"
        )

    lines.extend(["", "## Thresholds", ""])
    for summary in summaries:
        thresholds = summary["thresholds"]
        lines.append(f"### {summary['adapter']}")
        lines.append(
            f"- Support floor: features firing on fewer than `{thresholds['min_present_prompts']}` prompts "
            "are treated as effectively absent."
        )
        lines.append(
            f"- Significant frequency shift: `|delta_token_freq| >= {thresholds['abs_delta_token_freq_threshold']:.6f}`"
        )
        lines.append(
            "- Significant activation-mass shift: "
            f"`|delta_mean_prompt_activation| >= {thresholds['abs_delta_mean_prompt_activation_threshold']:.6f}`"
        )
        lines.append(
            f"- Significant context shift: `context_shift >= {thresholds['context_shift_threshold']:.6f}` "
            f"with `prompt_flip_rate >= {thresholds['min_context_flip_rate']:.3f}`"
        )
        lines.append("")

    lines.extend(["## Representative Features", ""])
    for summary in summaries:
        lines.append(f"### {summary['adapter']}")
        for label in ["amplified", "suppressed", "newly_activated", "killed", "context_shifted"]:
            examples = summary["top_examples"][label][:3]
            if not examples:
                continue
            lines.append(f"- {label.replace('_', ' ').title()}:")
            for row in examples:
                feature_id = row["feature_id"]
                score = row["classification_score"]
                details = ", ".join(
                    f"{key}={value}"
                    for key, value in row.items()
                    if key not in {"feature_id", "classification_score"}
                )
                lines.append(f"  feature {feature_id}: score={score:.3f}, {details}")
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify per-feature adapter effects")
    parser.add_argument("--adapter", choices=TARGETS + ["all"], default="all")
    parser.add_argument("--effect-quantile", type=float, default=0.95)
    parser.add_argument("--context-quantile", type=float, default=0.95)
    parser.add_argument("--min-present-prompts", type=int, default=5)
    parser.add_argument("--min-context-flip-rate", type=float, default=0.03)
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    targets = TARGETS if args.adapter == "all" else [args.adapter]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("  DAY 14: Feature Classification")
    log("=" * 60)

    summaries = []
    for adapter in targets:
        log(f"\n{'=' * 60}")
        log(f"  ADAPTER: {adapter}")
        log(f"{'=' * 60}")

        frame = pd.read_csv(DIFF_ROOT / f"{adapter}_feature_metrics.csv")
        thresholds = derive_classification_thresholds(
            frame,
            effect_quantile=args.effect_quantile,
            context_quantile=args.context_quantile,
            min_present_prompts=args.min_present_prompts,
            min_context_flip_rate=args.min_context_flip_rate,
        )
        classified = classify_features(frame, thresholds)
        summary = summarize_classifications(adapter, classified, thresholds, top_k=args.top_k)
        summaries.append(summary)

        csv_path = OUTPUT_ROOT / f"{adapter}_classified_features.csv"
        json_path = OUTPUT_ROOT / f"{adapter}_classification_summary.json"
        classified.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        counts = summary["class_counts"]
        log(
            f"  amplified={counts['amplified']} suppressed={counts['suppressed']} "
            f"new={counts['newly_activated']} killed={counts['killed']} "
            f"context={counts['context_shifted']} unchanged={counts['unchanged']}"
        )

    SUMMARY_MD.write_text(render_summary_markdown(summaries), encoding="utf-8")
    log(f"\nSaved classification outputs to {OUTPUT_ROOT}")
    log(f"Saved markdown summary to {SUMMARY_MD}")


if __name__ == "__main__":
    main()
