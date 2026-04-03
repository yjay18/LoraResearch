"""
Day 12: Validate SAE transferability from the base model to LoRA-adapted activations.

Reads the per-prompt SAE decode metadata from data/sae_features/{domain}/metadata.json,
compares reconstruction quality against the base model on the same prompts, and writes
an assessment report to results/day12_sae_transfer_assessment.{json,md}.
"""

import json
import statistics
from collections import defaultdict
from pathlib import Path

import numpy as np


TARGETS = ["base", "code", "medical", "math", "safety", "creative"]
FEATURE_ROOT = Path("data/sae_features")
OUTPUT_JSON = Path("results/day12_sae_transfer_assessment.json")
OUTPUT_MD = Path("results/day12_sae_transfer_assessment.md")


def load_metadata(domain: str) -> dict:
    path = FEATURE_ROOT / domain / "metadata.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def pct_delta(value: float, baseline: float) -> float:
    return ((value - baseline) / baseline) * 100.0


def summarize_pairwise(base_prompts: list[dict], adapted_prompts: list[dict]) -> dict:
    """Compare one adapted domain against the same prompt ids in the base model."""
    deltas = []
    by_prompt_domain = defaultdict(list)

    for base_prompt, adapted_prompt in zip(base_prompts, adapted_prompts):
        if base_prompt["id"] != adapted_prompt["id"]:
            raise RuntimeError(f"Mismatched prompt ids: {base_prompt['id']} vs {adapted_prompt['id']}")
        if base_prompt["seq_len"] != adapted_prompt["seq_len"]:
            raise RuntimeError(
                f"Mismatched seq_len for prompt {base_prompt['id']}: "
                f"{base_prompt['seq_len']} vs {adapted_prompt['seq_len']}"
            )

        delta = {
            "id": adapted_prompt["id"],
            "prompt_domain": adapted_prompt["domain"],
            "source": adapted_prompt["source"],
            "seq_len": adapted_prompt["seq_len"],
            "text": adapted_prompt["text"],
            "base_mse": base_prompt["mse"],
            "adapted_mse": adapted_prompt["mse"],
            "delta_mse": adapted_prompt["mse"] - base_prompt["mse"],
            "base_l0": base_prompt["l0"],
            "adapted_l0": adapted_prompt["l0"],
            "delta_l0": adapted_prompt["l0"] - base_prompt["l0"],
            "base_sparsity": base_prompt["sparsity"],
            "adapted_sparsity": adapted_prompt["sparsity"],
            "delta_sparsity": adapted_prompt["sparsity"] - base_prompt["sparsity"],
        }
        deltas.append(delta)
        by_prompt_domain[adapted_prompt["domain"]].append(delta)

    improved = [d for d in deltas if d["delta_mse"] < 0]
    worsened = [d for d in deltas if d["delta_mse"] > 0]
    unchanged = [d for d in deltas if d["delta_mse"] == 0]

    prompt_domain_summary = {}
    for prompt_domain, rows in sorted(by_prompt_domain.items()):
        mse_deltas = [row["delta_mse"] for row in rows]
        prompt_domain_summary[prompt_domain] = {
            "count": len(rows),
            "avg_delta_mse": float(np.mean(mse_deltas)),
            "median_delta_mse": float(np.median(mse_deltas)),
            "improved_prompts": sum(1 for row in rows if row["delta_mse"] < 0),
            "worsened_prompts": sum(1 for row in rows if row["delta_mse"] > 0),
        }

    mse_deltas = [d["delta_mse"] for d in deltas]
    l0_deltas = [d["delta_l0"] for d in deltas]
    sparsity_deltas = [d["delta_sparsity"] for d in deltas]

    return {
        "prompt_count": len(deltas),
        "improved_prompts": len(improved),
        "worsened_prompts": len(worsened),
        "unchanged_prompts": len(unchanged),
        "avg_delta_mse": float(np.mean(mse_deltas)),
        "median_delta_mse": float(np.median(mse_deltas)),
        "std_delta_mse": float(np.std(mse_deltas)),
        "avg_delta_l0": float(np.mean(l0_deltas)),
        "avg_delta_sparsity": float(np.mean(sparsity_deltas)),
        "p05_delta_mse": float(np.percentile(mse_deltas, 5)),
        "p95_delta_mse": float(np.percentile(mse_deltas, 95)),
        "largest_improvements": sorted(improved, key=lambda row: row["delta_mse"])[:5],
        "largest_worsenings": sorted(worsened, key=lambda row: row["delta_mse"], reverse=True)[:5],
        "by_prompt_domain": prompt_domain_summary,
    }


def assess_transfer(domain: str, base: dict, adapted: dict) -> dict:
    """Create a domain-level transferability summary."""
    pairwise = summarize_pairwise(base["prompts"], adapted["prompts"])
    mse_pct = pct_delta(adapted["avg_mse"], base["avg_mse"])

    if mse_pct > 10:
        verdict = "Transfer concern: adapted activations reconstruct materially worse than base."
    elif mse_pct > 2:
        verdict = "Transfer looks mixed: adapted activations reconstruct slightly worse than base."
    elif mse_pct >= -2:
        verdict = "Transfer looks stable: adapted activations reconstruct about as well as base."
    else:
        verdict = "Transfer looks good: adapted activations reconstruct better than base."

    return {
        "adapter": domain,
        "avg_mse": adapted["avg_mse"],
        "avg_mse_delta": adapted["avg_mse"] - base["avg_mse"],
        "avg_mse_delta_pct": mse_pct,
        "avg_l0": adapted["avg_l0"],
        "avg_l0_delta": adapted["avg_l0"] - base["avg_l0"],
        "avg_sparsity": adapted["avg_sparsity"],
        "avg_sparsity_delta": adapted["avg_sparsity"] - base["avg_sparsity"],
        "avg_nnz": adapted["avg_nnz"],
        "avg_nnz_delta": adapted["avg_nnz"] - base["avg_nnz"],
        "same_prompt_count": pairwise["prompt_count"],
        "paired_comparison": pairwise,
        "transfer_assessment": verdict,
    }


def render_markdown(report: dict) -> str:
    """Render a compact human-readable summary."""
    lines = [
        "# Day 12: SAE Transfer Assessment",
        "",
        "This report compares SAE reconstruction quality for the base model versus each LoRA-adapted model on the same 300 prompts.",
        "",
        "## Base Reference",
        "",
        f"- Avg reconstruction MSE: {report['base']['avg_mse']:.2f}",
        f"- Avg L0: {report['base']['avg_l0']:.2f}",
        f"- Avg sparsity: {report['base']['avg_sparsity']:.5f}",
        f"- Avg non-zero entries per prompt: {report['base']['avg_nnz']:.2f}",
        "",
        "## Domain Summary",
        "",
        "| Adapter | Avg MSE | Delta vs Base | Delta % | Avg L0 | Avg Sparsity | Improved Prompts | Worsened Prompts |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]

    for domain in report["domains"]:
        paired = domain["paired_comparison"]
        lines.append(
            f"| {domain['adapter']} | {domain['avg_mse']:.2f} | {domain['avg_mse_delta']:.2f} | "
            f"{domain['avg_mse_delta_pct']:.2f}% | {domain['avg_l0']:.2f} | {domain['avg_sparsity']:.5f} | "
            f"{paired['improved_prompts']} | {paired['worsened_prompts']} |"
        )

    lines.extend(["", "## Interpretation", ""])
    for domain in report["domains"]:
        lines.append(f"- **{domain['adapter']}**: {domain['transfer_assessment']}")

    lines.extend(["", "## Prompt-Domain Effects", ""])
    for domain in report["domains"]:
        lines.append(f"### {domain['adapter']}")
        for prompt_domain, summary in domain["paired_comparison"]["by_prompt_domain"].items():
            lines.append(
                f"- {prompt_domain}: avg MSE delta {summary['avg_delta_mse']:.2f} "
                f"({summary['improved_prompts']} improved / {summary['worsened_prompts']} worsened)"
            )
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    base = load_metadata("base")
    domains = []
    for domain in TARGETS[1:]:
        domains.append(assess_transfer(domain, base, load_metadata(domain)))

    report = {
        "day": 12,
        "task": "Validate SAE transferability across LoRA-adapted activations",
        "base": {
            "avg_mse": base["avg_mse"],
            "avg_l0": base["avg_l0"],
            "avg_sparsity": base["avg_sparsity"],
            "avg_nnz": base["avg_nnz"],
            "num_prompts": base["num_prompts"],
            "sae_release": base["sae_release"],
            "sae_id": base["sae_id"],
        },
        "domains": domains,
    }

    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    OUTPUT_MD.write_text(render_markdown(report), encoding="utf-8")

    print("=" * 60)
    print("  DAY 12: SAE Transfer Assessment")
    print("=" * 60)
    print(f"Base avg MSE: {base['avg_mse']:.2f}")
    print("")
    for domain in domains:
        paired = domain["paired_comparison"]
        print(
            f"{domain['adapter']:>8}: avg_mse={domain['avg_mse']:.2f} "
            f"delta={domain['avg_mse_delta']:.2f} ({domain['avg_mse_delta_pct']:.2f}%) "
            f"improved={paired['improved_prompts']} worsened={paired['worsened_prompts']}"
        )
    print("")
    print(f"Saved JSON report to {OUTPUT_JSON}")
    print(f"Saved Markdown report to {OUTPUT_MD}")


if __name__ == "__main__":
    main()
