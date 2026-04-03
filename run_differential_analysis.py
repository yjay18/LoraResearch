"""
Day 13: Compute per-feature differential metrics for all LoRA adapters.

Reads sparse SAE feature dumps from data/sae_features, compares each adapter against
the base model, and writes per-feature tables plus summary reports.
"""

from __future__ import annotations

import json
from pathlib import Path

from lorasurgeon.diff import (
    compute_feature_differential,
    load_domain_feature_stats,
    summarize_feature_differential,
)


TARGETS = ["code", "medical", "math", "safety", "creative"]
FEATURE_ROOT = Path("data/sae_features")
OUTPUT_ROOT = Path("results/differential")
SUMMARY_MD = Path("results/day13_differential_analysis.md")


def log(msg: str) -> None:
    print(msg, flush=True)


def write_summary_markdown(summaries: list[dict]) -> None:
    """Render a compact adapter-level summary page."""
    lines = [
        "# Day 13: Differential Feature Analysis",
        "",
        "Per-feature comparisons between base SAE activations and each LoRA-adapted model.",
        "",
        "| Adapter | Mean Abs Delta Token Freq | Mean Delta Mean Active | Mean Context Shift | Domain Switches |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]

    for summary in summaries:
        lines.append(
            f"| {summary['adapter']} | {summary['mean_abs_delta_token_freq']:.6f} | "
            f"{summary['mean_delta_mean_active']:.6f} | {summary['mean_context_shift']:.6f} | "
            f"{summary['features_with_domain_switch']} |"
        )

    lines.extend(["", "## Top Features By Adapter", ""])
    for summary in summaries:
        lines.append(f"### {summary['adapter']}")
        lines.append("- Top amplified by frequency:")
        for row in summary["top_amplified_by_frequency"][:5]:
            lines.append(
                f"  feature {row['feature_id']}: delta_token_freq={row['delta_token_freq']:.6f}, "
                f"delta_prompt_freq={row['delta_prompt_freq']:.6f}, "
                f"delta_mean_active={row['delta_mean_active']:.6f}"
            )
        lines.append("- Top context shifted:")
        for row in summary["top_context_shifted"][:5]:
            lines.append(
                f"  feature {row['feature_id']}: context_shift={row['context_shift']:.6f}, "
                f"context_jaccard={row['context_jaccard']:.6f}, "
                f"prompt_flip_rate={row['prompt_flip_rate']:.6f}"
            )
        lines.append("")

    SUMMARY_MD.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("  DAY 13: Differential Feature Analysis")
    log("=" * 60)

    log("Loading base feature statistics...")
    base_stats = load_domain_feature_stats(FEATURE_ROOT / "base")
    summaries = []

    for adapter in TARGETS:
        log(f"\n{'=' * 60}")
        log(f"  ADAPTER: {adapter}")
        log(f"{'=' * 60}")
        adapted_stats = load_domain_feature_stats(FEATURE_ROOT / adapter)
        frame = compute_feature_differential(base_stats, adapted_stats)
        summary = summarize_feature_differential(adapter, frame)
        summaries.append(summary)

        csv_path = OUTPUT_ROOT / f"{adapter}_feature_metrics.csv"
        json_path = OUTPUT_ROOT / f"{adapter}_summary.json"
        frame.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        log(
            f"  Wrote {frame.shape[0]} feature rows to {csv_path} "
            f"(mean_abs_delta_token_freq={summary['mean_abs_delta_token_freq']:.6f}, "
            f"mean_context_shift={summary['mean_context_shift']:.6f})"
        )

    write_summary_markdown(summaries)

    log(f"\nSaved adapter summaries to {OUTPUT_ROOT}")
    log(f"Saved markdown summary to {SUMMARY_MD}")


if __name__ == "__main__":
    main()
