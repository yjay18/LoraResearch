"""Offline integrity checks for saved LoRASurgeon research artifacts."""

from __future__ import annotations

import csv
import json
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
PROMPTS_PATH = DATA_DIR / "prompts_300.json"
ACTIVATIONS_DIR = DATA_DIR / "activations"
FEATURES_DIR = DATA_DIR / "sae_features"
DAY12_REPORT = RESULTS_DIR / "day12_sae_transfer_assessment.json"
DIFF_DIR = RESULTS_DIR / "differential"
CLASSIFICATION_DIR = RESULTS_DIR / "classification"
LABELS_DIR = RESULTS_DIR / "labels"
DAY16_REPORT = RESULTS_DIR / "day16_code_adapter_deep_dive.json"

BASE_AND_ADAPTERS = ["base", "code", "medical", "math", "safety", "creative"]
ADAPTERS = BASE_AND_ADAPTERS[1:]
SAMPLE_PROMPT_INDICES = [0, 1, 50, 149, 299]


def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_sparse_feature_archive(path: Path) -> dict:
    with np.load(path) as data:
        token_indices = data["token_indices"]
        feature_indices = data["feature_indices"]
        values = data["values"]
        seq_len = int(data["seq_len"][0])
        d_sae = int(data["d_sae"][0])
    return {
        "token_indices": token_indices,
        "feature_indices": feature_indices,
        "values": values,
        "seq_len": seq_len,
        "d_sae": d_sae,
    }


class ResearchIntegrityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.prompts = load_json(PROMPTS_PATH)
        cls.prompts_by_id = {int(prompt["id"]): prompt for prompt in cls.prompts}
        cls.base_activation_metadata = load_json(ACTIVATIONS_DIR / "base" / "metadata.json")

    def test_prompt_dataset_is_balanced_and_unique(self) -> None:
        self.assertEqual(len(self.prompts), 300)

        prompt_ids = [int(prompt["id"]) for prompt in self.prompts]
        self.assertEqual(prompt_ids, list(range(300)))
        self.assertEqual(len({prompt["text"] for prompt in self.prompts}), 300)

        domain_counts = Counter(prompt["domain"] for prompt in self.prompts)
        self.assertEqual(
            domain_counts,
            Counter(
                {
                    "code": 50,
                    "medical": 50,
                    "math": 50,
                    "safety": 50,
                    "creative": 50,
                    "general": 50,
                }
            ),
        )

    def test_activation_artifacts_are_complete_and_aligned(self) -> None:
        base_prompts = self.base_activation_metadata["prompts"]

        for domain in BASE_AND_ADAPTERS:
            metadata = load_json(ACTIVATIONS_DIR / domain / "metadata.json")
            prompt_files = sorted((ACTIVATIONS_DIR / domain).glob("prompt_*.npy"))

            self.assertEqual(metadata["num_prompts"], 300)
            self.assertEqual(len(metadata["prompts"]), 300)
            self.assertEqual(len(prompt_files), 300)
            self.assertEqual(metadata["d_model"], 2304)
            self.assertEqual(metadata["layer"], 12)

            for idx, prompt_meta in enumerate(metadata["prompts"]):
                canonical = self.prompts_by_id[int(prompt_meta["id"])]
                self.assertEqual(prompt_meta["domain"], canonical["domain"])
                self.assertEqual(prompt_meta["source"], canonical["source"])
                if domain != "base":
                    self.assertEqual(prompt_meta["shape"], base_prompts[idx]["shape"])
                    self.assertEqual(prompt_meta["seq_len"], base_prompts[idx]["seq_len"])

            for idx in SAMPLE_PROMPT_INDICES:
                arr = np.load(prompt_files[idx], mmap_mode="r")
                prompt_meta = metadata["prompts"][idx]
                self.assertEqual(tuple(arr.shape), tuple(prompt_meta["shape"]))
                self.assertEqual(arr.shape[0], prompt_meta["seq_len"])
                self.assertEqual(arr.shape[1], metadata["d_model"])
                self.assertEqual(arr.dtype, np.float32)

    def test_sae_feature_artifacts_match_saved_metadata(self) -> None:
        for domain in BASE_AND_ADAPTERS:
            activation_metadata = load_json(ACTIVATIONS_DIR / domain / "metadata.json")
            feature_metadata = load_json(FEATURES_DIR / domain / "metadata.json")
            prompt_files = sorted((FEATURES_DIR / domain).glob("prompt_*.npz"))

            self.assertEqual(feature_metadata["num_prompts"], 300)
            self.assertEqual(len(feature_metadata["prompts"]), 300)
            self.assertEqual(len(prompt_files), 300)
            self.assertEqual(feature_metadata["residual_layer"], activation_metadata["layer"])
            self.assertEqual(feature_metadata["d_model"], activation_metadata["d_model"])
            self.assertEqual(feature_metadata["d_sae"], 16384)

            for idx in SAMPLE_PROMPT_INDICES:
                sparse = load_sparse_feature_archive(prompt_files[idx])
                feature_prompt = feature_metadata["prompts"][idx]
                activation_prompt = activation_metadata["prompts"][idx]
                nnz = int(sparse["values"].shape[0])

                self.assertEqual(sparse["seq_len"], activation_prompt["seq_len"])
                self.assertEqual(sparse["seq_len"], feature_prompt["seq_len"])
                self.assertEqual(sparse["d_sae"], feature_metadata["d_sae"])
                self.assertEqual(nnz, feature_prompt["nnz"])

                if nnz:
                    self.assertLess(int(sparse["token_indices"].max()), sparse["seq_len"])
                    self.assertLess(int(sparse["feature_indices"].max()), sparse["d_sae"])

    def test_day12_report_matches_saved_day11_metadata(self) -> None:
        report = load_json(DAY12_REPORT)
        base_feature_metadata = load_json(FEATURES_DIR / "base" / "metadata.json")

        self.assertAlmostEqual(report["base"]["avg_mse"], base_feature_metadata["avg_mse"], places=9)
        self.assertAlmostEqual(report["base"]["avg_l0"], base_feature_metadata["avg_l0"], places=9)
        self.assertAlmostEqual(report["base"]["avg_sparsity"], base_feature_metadata["avg_sparsity"], places=9)
        self.assertAlmostEqual(report["base"]["avg_nnz"], base_feature_metadata["avg_nnz"], places=9)

        report_domains = {row["adapter"]: row for row in report["domains"]}
        for domain in ADAPTERS:
            feature_metadata = load_json(FEATURES_DIR / domain / "metadata.json")
            report_row = report_domains[domain]
            self.assertAlmostEqual(report_row["avg_mse"], feature_metadata["avg_mse"], places=9)
            self.assertAlmostEqual(report_row["avg_l0"], feature_metadata["avg_l0"], places=9)
            self.assertAlmostEqual(report_row["avg_sparsity"], feature_metadata["avg_sparsity"], places=9)
            self.assertAlmostEqual(report_row["avg_nnz"], feature_metadata["avg_nnz"], places=9)
            self.assertEqual(report_row["paired_comparison"]["prompt_count"], 300)

    def test_day13_summaries_match_saved_csvs(self) -> None:
        for domain in ADAPTERS:
            summary = load_json(DIFF_DIR / f"{domain}_summary.json")
            csv_path = DIFF_DIR / f"{domain}_feature_metrics.csv"

            row_count = 0
            sum_abs_delta_token_freq = 0.0
            sum_delta_mean_active = 0.0
            sum_context_shift = 0.0
            domain_switches = 0

            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_count += 1
                    sum_abs_delta_token_freq += abs(float(row["delta_token_freq"]))
                    sum_delta_mean_active += float(row["delta_mean_active"])
                    sum_context_shift += float(row["context_shift"])
                    domain_switches += int(row["domain_switch"] == "True")

            self.assertEqual(row_count, 16384)
            self.assertEqual(summary["feature_count"], row_count)
            self.assertEqual(summary["features_with_domain_switch"], domain_switches)
            self.assertAlmostEqual(
                summary["mean_abs_delta_token_freq"],
                sum_abs_delta_token_freq / row_count,
                places=12,
            )
            self.assertAlmostEqual(
                summary["mean_delta_mean_active"],
                sum_delta_mean_active / row_count,
                places=12,
            )
            self.assertAlmostEqual(
                summary["mean_context_shift"],
                sum_context_shift / row_count,
                places=12,
            )

    def test_day14_classification_summaries_match_saved_csvs(self) -> None:
        for domain in ADAPTERS:
            summary = load_json(CLASSIFICATION_DIR / f"{domain}_classification_summary.json")
            csv_path = CLASSIFICATION_DIR / f"{domain}_classified_features.csv"

            counts = Counter()
            row_count = 0
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_count += 1
                    counts[row["classification"]] += 1

            self.assertEqual(row_count, 16384)
            self.assertEqual(summary["feature_count"], row_count)
            self.assertEqual(summary["class_counts"], {label: counts[label] for label in summary["class_counts"]})

    def test_day15_feature_labels_are_complete(self) -> None:
        for domain in ADAPTERS:
            payload = load_json(LABELS_DIR / f"{domain}_feature_labels.json")
            csv_path = LABELS_DIR / f"{domain}_feature_labels.csv"

            self.assertEqual(payload["adapter"], domain)
            self.assertEqual(payload["label_count"], 100)
            self.assertEqual(len(payload["labels"]), 100)
            self.assertTrue(Path(payload["tokenizer_path"]).exists())

            feature_ids = [int(row["feature_id"]) for row in payload["labels"]]
            self.assertEqual(len(feature_ids), len(set(feature_ids)))

            row_count = 0
            with open(csv_path, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    row_count += 1
                    self.assertNotEqual(row["classification"], "unchanged")
                    self.assertTrue(row["label_family"])
                    self.assertTrue(row["label"])
                    self.assertTrue(row["rationale"])

            self.assertEqual(row_count, 100)

    def test_day16_report_matches_labeled_code_features(self) -> None:
        report = load_json(DAY16_REPORT)
        labels = load_json(LABELS_DIR / "code_feature_labels.json")["labels"]

        self.assertEqual(report["adapter"], "code")
        self.assertEqual(report["top_k"], 100)

        class_counts = Counter(row["classification"] for row in labels)
        family_count_map = Counter(row["label_family"] for row in labels)

        self.assertEqual(report["classification_counts"], {label: class_counts[label] for label in report["classification_counts"]})
        self.assertEqual(report["label_family_counts"], {label: family_count_map[label] for label in report["label_family_counts"]})


if __name__ == "__main__":
    unittest.main()
