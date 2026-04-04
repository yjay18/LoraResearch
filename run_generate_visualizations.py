"""
Day 19: Generate figures and a notebook for publication-style visualization.
"""

from __future__ import annotations

import json
from pathlib import Path

from lorasurgeon.viz import (
    ADAPTERS,
    load_json,
    plot_feature_domain_heatmap,
    plot_top_features_bar,
    plot_universal_features_heatmap,
    plot_volcano_style,
)


LABEL_ROOT = Path("results/labels")
CLASSIFICATION_ROOT = Path("results/classification")
FIGURE_ROOT = Path("results/figures")
NOTEBOOK_PATH = Path("notebooks/day19_visualizations.ipynb")
SUMMARY_MD = Path("results/day19_visualizations.md")
SUMMARY_JSON = Path("results/day19_visualizations.json")
UNIVERSAL_PATH = Path("results/day18_universal_features.json")


def log(msg: str) -> None:
    print(msg, flush=True)


def build_notebook() -> dict:
    return {
        "cells": [
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "# Day 19 Visualization Notebook\n",
                    "\n",
                    "This notebook regenerates the Day 19 figures from the saved Day 13-18 artifacts.\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "from pathlib import Path\n",
                    "from lorasurgeon.viz import (\n",
                    "    ADAPTERS,\n",
                    "    load_json,\n",
                    "    plot_feature_domain_heatmap,\n",
                    "    plot_top_features_bar,\n",
                    "    plot_universal_features_heatmap,\n",
                    "    plot_volcano_style,\n",
                    ")\n",
                    "\n",
                    "label_root = Path('results/labels')\n",
                    "classification_root = Path('results/classification')\n",
                    "figure_root = Path('results/figures')\n",
                    "figure_root.mkdir(parents=True, exist_ok=True)\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "for adapter in ADAPTERS:\n",
                    "    payload = load_json(label_root / f'{adapter}_feature_labels.json')\n",
                    "    plot_feature_domain_heatmap(payload, figure_root / f'{adapter}_domain_heatmap.png')\n",
                    "    plot_top_features_bar(payload, figure_root / f'{adapter}_top20_bar.png')\n",
                    "    plot_volcano_style(\n",
                    "        classification_root / f'{adapter}_classified_features.csv',\n",
                    "        classification_root / f'{adapter}_classification_summary.json',\n",
                    "        figure_root / f'{adapter}_volcano.png',\n",
                    "        adapter,\n",
                    "    )\n",
                ],
            },
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": [
                    "universal_summary = load_json(Path('results/day18_universal_features.json'))\n",
                    "plot_universal_features_heatmap(\n",
                    "    universal_summary,\n",
                    "    classification_root,\n",
                    "    figure_root / 'universal_features_heatmap.png',\n",
                    ")\n",
                ],
            },
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.x",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def render_markdown(manifest: dict) -> str:
    lines = [
        "# Day 19: Visualization Outputs",
        "",
        "Generated publication-style figures for each adapter plus a universal-feature heatmap.",
        "",
        "## Figures",
        "",
    ]

    for adapter, figures in manifest["adapter_figures"].items():
        lines.append(f"### {adapter}")
        for figure in figures:
            lines.append(f"- `{figure}`")
        lines.append("")

    lines.extend(
        [
            "## Universal Figure",
            "",
            f"- `{manifest['universal_figure']}`",
            "",
            "## Notebook",
            "",
            f"- `{manifest['notebook']}`",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    log("=" * 60)
    log("  DAY 19: Visualization Generation")
    log("=" * 60)

    adapter_figures = {}
    for adapter in ADAPTERS:
        payload = load_json(LABEL_ROOT / f"{adapter}_feature_labels.json")
        domain_heatmap = FIGURE_ROOT / f"{adapter}_domain_heatmap.png"
        volcano = FIGURE_ROOT / f"{adapter}_volcano.png"
        top_bar = FIGURE_ROOT / f"{adapter}_top20_bar.png"

        plot_feature_domain_heatmap(payload, domain_heatmap)
        plot_volcano_style(
            CLASSIFICATION_ROOT / f"{adapter}_classified_features.csv",
            CLASSIFICATION_ROOT / f"{adapter}_classification_summary.json",
            volcano,
            adapter,
        )
        plot_top_features_bar(payload, top_bar)
        adapter_figures[adapter] = [str(domain_heatmap), str(volcano), str(top_bar)]
        log(f"  {adapter}: wrote 3 figures")

    universal_summary = load_json(UNIVERSAL_PATH)
    universal_figure = FIGURE_ROOT / "universal_features_heatmap.png"
    plot_universal_features_heatmap(universal_summary, CLASSIFICATION_ROOT, universal_figure)
    log("  universal: wrote 1 figure")

    NOTEBOOK_PATH.write_text(json.dumps(build_notebook(), indent=2), encoding="utf-8")

    manifest = {
        "adapter_figures": adapter_figures,
        "universal_figure": str(universal_figure),
        "notebook": str(NOTEBOOK_PATH),
    }
    SUMMARY_JSON.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    SUMMARY_MD.write_text(render_markdown(manifest), encoding="utf-8")

    log(f"  Saved notebook to {NOTEBOOK_PATH}")
    log(f"  Saved manifest to {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
