"""
Day 15: Label the top changed features for each adapter using local prompt/token evidence.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lorasurgeon.fingerprint import (
    label_selected_features,
    labels_to_frame,
    select_top_changed_features,
    summarize_label_families,
)


TARGETS = ["code", "medical", "math", "safety", "creative"]
PROMPTS_PATH = Path("data/prompts_300.json")
FEATURE_ROOT = Path("data/sae_features")
CLASSIFICATION_ROOT = Path("results/classification")
OUTPUT_ROOT = Path("results/labels")
SUMMARY_MD = Path("results/day15_feature_labels.md")
TOP_K = 100


def log(msg: str) -> None:
    print(msg, flush=True)


def render_markdown(summaries: list[dict]) -> str:
    lines = [
        "# Day 15: Offline Feature Labeling",
        "",
        "Top changed SAE features were labeled from saved sparse activations plus cached Gemma tokenization.",
        "This is an offline autointerp-style pass, not the EleutherAI `sae-auto-interp` stack referenced in the schedule.",
        "",
        "| Adapter | Labeled Features | Most Common Label Families |",
        "| --- | ---: | --- |",
    ]

    for summary in summaries:
        top_families = ", ".join(
            f"{family} ({count})"
            for family, count in list(summary["family_counts"].items())[:4]
        )
        lines.append(f"| {summary['adapter']} | {summary['label_count']} | {top_families} |")

    lines.extend(["", "## Representative Labels", ""])
    for summary in summaries:
        lines.append(f"### {summary['adapter']}")
        for row in summary["top_examples"]:
            lines.append(
                f"- feature {row['feature_id']} [{row['classification']}]: {row['label']} "
                f"(family={row['label_family']}, confidence={row['confidence']:.3f}, "
                f"primary_domain={row['primary_domain']})"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []

    log("=" * 60)
    log("  DAY 15: Offline Feature Labeling")
    log("=" * 60)

    for adapter in TARGETS:
        log(f"\n{'=' * 60}")
        log(f"  ADAPTER: {adapter}")
        log(f"{'=' * 60}")

        classified = pd.read_csv(CLASSIFICATION_ROOT / f"{adapter}_classified_features.csv")
        summary_json = json.loads(
            (CLASSIFICATION_ROOT / f"{adapter}_classification_summary.json").read_text(encoding="utf-8")
        )
        selected = select_top_changed_features(classified, summary_json["thresholds"], top_k=TOP_K)
        labels, tokenizer_path = label_selected_features(
            adapter=adapter,
            selected=selected,
            prompts_path=PROMPTS_PATH,
            feature_root=FEATURE_ROOT,
        )

        frame = labels_to_frame(labels)
        family_counts = summarize_label_families(labels)
        ordered_family_counts = dict(sorted(family_counts.items(), key=lambda item: (-item[1], item[0])))

        json_payload = {
            "adapter": adapter,
            "top_k": TOP_K,
            "method": "offline_sparse_activation_labeling",
            "tokenizer_path": str(tokenizer_path),
            "label_count": len(labels),
            "family_counts": ordered_family_counts,
            "top_examples": labels[:10],
            "labels": labels,
        }

        csv_path = OUTPUT_ROOT / f"{adapter}_feature_labels.csv"
        json_path = OUTPUT_ROOT / f"{adapter}_feature_labels.json"
        frame.to_csv(csv_path, index=False)
        json_path.write_text(json.dumps(json_payload, indent=2), encoding="utf-8")

        summaries.append(
            {
                "adapter": adapter,
                "label_count": len(labels),
                "family_counts": ordered_family_counts,
                "top_examples": labels[:5],
            }
        )
        log(
            f"  labeled={len(labels)} tokenizer={tokenizer_path.name} "
            f"top_family={next(iter(ordered_family_counts.items())) if ordered_family_counts else 'n/a'}"
        )

    SUMMARY_MD.write_text(render_markdown(summaries), encoding="utf-8")
    log(f"\nSaved Day 15 labels to {OUTPUT_ROOT}")
    log(f"Saved markdown summary to {SUMMARY_MD}")


if __name__ == "__main__":
    main()
