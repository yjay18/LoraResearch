"""
Day 11: Decode base and adapted residual activations through the same Gemma Scope SAE.

Reads prompt-level residual activations from data/activations/{domain}, encodes them
with a fixed SAE, and writes sparse feature activations to data/sae_features/{domain}.

Output structure:
    data/sae_features/{domain}/
        prompt_000.npz   # token_indices, feature_indices, values, seq_len, d_sae
        ...
        metadata.json    # per-prompt metrics and dataset-level summary
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))

from lorasurgeon.sae_decode import GemmaScopeSAE, SparseFeatureActs, sparsify_feature_acts


def log(msg: str) -> None:
    print(msg, flush=True)


def clear_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mem_gb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / 1e9


TARGETS = ["base", "code", "medical", "math", "safety", "creative"]
ACTIVATION_ROOT = Path("data/activations")
OUTPUT_ROOT = Path("data/sae_features")
REPORT_EVERY = 50


def load_activation_metadata(domain: str) -> dict:
    path = ACTIVATION_ROOT / domain / "metadata.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_domain_metadata(path: Path, metadata: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def decode_prompt(
    sae: GemmaScopeSAE,
    activation_path: Path,
    output_path: Path,
) -> tuple[SparseFeatureActs, dict]:
    """Decode one prompt's residual activations and save sparse features."""
    activations = np.load(activation_path)
    tensor = torch.from_numpy(activations).unsqueeze(0)

    sae_result = sae.forward(tensor)
    sparse = sparsify_feature_acts(sae_result.feature_acts)
    sparse.save(output_path)

    metrics = {
        "seq_len": int(activations.shape[0]),
        "activation_shape": list(activations.shape),
        "feature_shape": [int(sparse.seq_len), int(sparse.d_sae)],
        "nnz": sparse.nnz,
        "mse": sae_result.mse,
        "l0": sae_result.l0,
        "sparsity": sae_result.sparsity,
    }

    del tensor, sae_result
    clear_gpu()
    return sparse, metrics


def verify_domain_output(out_dir: Path, expected_prompts: int, expected_d_sae: int) -> None:
    """Basic structural verification for the sparse feature dump."""
    prompt_files = sorted(out_dir.glob("prompt_*.npz"))
    if len(prompt_files) != expected_prompts:
        raise RuntimeError(
            f"Expected {expected_prompts} sparse prompt files in {out_dir}, found {len(prompt_files)}"
        )

    sample = SparseFeatureActs.load(prompt_files[0])
    if sample.d_sae != expected_d_sae:
        raise RuntimeError(
            f"Expected d_sae={expected_d_sae} in {prompt_files[0].name}, got {sample.d_sae}"
        )


def decode_domain(
    sae: GemmaScopeSAE,
    domain: str,
    limit: int | None = None,
    overwrite: bool = False,
) -> dict:
    """Decode one domain's activation set and persist sparse feature activations."""
    in_dir = ACTIVATION_ROOT / domain
    out_dir = OUTPUT_ROOT / domain
    out_dir.mkdir(parents=True, exist_ok=True)

    activation_metadata = load_activation_metadata(domain)
    prompts = activation_metadata["prompts"]
    if limit is not None:
        prompts = prompts[:limit]

    domain_metadata = {
        "source_dir": str(in_dir).replace("\\", "/"),
        "output_dir": str(out_dir).replace("\\", "/"),
        "model": activation_metadata["model"],
        "adapter": activation_metadata.get("adapter", domain),
        "quantization": activation_metadata["quantization"],
        "residual_layer": activation_metadata["layer"],
        "sae_release": sae.release,
        "sae_id": sae.sae_id,
        "sae_layer": sae.layer,
        "d_model": activation_metadata["d_model"],
        "d_sae": sae.d_sae,
        "num_prompts": len(prompts),
        "storage_format": "sparse_npz(token_indices:uint16, feature_indices:uint16, values:float16)",
        "prompts": [],
    }

    mse_values = []
    l0_values = []
    sparsity_values = []
    nnz_values = []

    start = time.time()
    for i, prompt_data in enumerate(prompts):
        activation_path = in_dir / f"prompt_{i:03d}.npy"
        output_path = out_dir / f"prompt_{i:03d}.npz"

        if output_path.exists() and not overwrite:
            sparse = SparseFeatureActs.load(output_path)
            metrics = {
                "seq_len": int(prompt_data["seq_len"]),
                "activation_shape": list(prompt_data["shape"]),
                "feature_shape": [int(prompt_data["seq_len"]), int(sae.d_sae)],
                "nnz": sparse.nnz,
                "mse": None,
                "l0": None,
                "sparsity": None,
            }
        else:
            sparse, metrics = decode_prompt(sae, activation_path, output_path)

        domain_metadata["prompts"].append({
            "id": prompt_data["id"],
            "domain": prompt_data["domain"],
            "source": prompt_data["source"],
            "text": prompt_data["text"],
            "seq_len": metrics["seq_len"],
            "activation_shape": metrics["activation_shape"],
            "feature_shape": metrics["feature_shape"],
            "nnz": metrics["nnz"],
            "mse": metrics["mse"],
            "l0": metrics["l0"],
            "sparsity": metrics["sparsity"],
            "file": output_path.name,
        })

        nnz_values.append(metrics["nnz"])
        if metrics["mse"] is not None:
            mse_values.append(metrics["mse"])
            l0_values.append(metrics["l0"])
            sparsity_values.append(metrics["sparsity"])

        if (i + 1) % REPORT_EVERY == 0 or i == 0 or i + 1 == len(prompts):
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(prompts) - i - 1) / rate if rate else 0.0
            last_mse = "n/a" if metrics["mse"] is None else f"{metrics['mse']:.2f}"
            log(
                f"  [{i+1}/{len(prompts)}] seq_len={metrics['seq_len']:>3} "
                f"nnz={metrics['nnz']:>6} mse={last_mse:>6} "
                f"{rate:.1f} prompts/sec | ETA: {eta:.0f}s"
            )

    domain_metadata["avg_nnz"] = float(np.mean(nnz_values)) if nnz_values else 0.0
    domain_metadata["max_nnz"] = int(max(nnz_values)) if nnz_values else 0
    domain_metadata["avg_mse"] = float(np.mean(mse_values)) if mse_values else None
    domain_metadata["avg_l0"] = float(np.mean(l0_values)) if l0_values else None
    domain_metadata["avg_sparsity"] = float(np.mean(sparsity_values)) if sparsity_values else None
    domain_metadata["elapsed_sec"] = round(time.time() - start, 2)
    domain_metadata["vram_gb"] = round(mem_gb(), 2)

    save_domain_metadata(out_dir / "metadata.json", domain_metadata)
    verify_domain_output(out_dir, len(prompts), sae.d_sae)
    return domain_metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Decode residual activations with a fixed Gemma Scope SAE")
    parser.add_argument("--domain", choices=TARGETS + ["all"], default="all")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N prompts per domain")
    parser.add_argument("--overwrite", action="store_true", help="Recompute outputs even if prompt files already exist")
    parser.add_argument("--device", type=str, default="cuda", help="Device to load the SAE on")
    args = parser.parse_args()

    targets = TARGETS if args.domain == "all" else [args.domain]

    log("=" * 60)
    log("  DAY 11: Decoding activations with Gemma Scope SAE")
    log("=" * 60)
    clear_gpu()

    log(f"Loading SAE on {args.device}...")
    sae = GemmaScopeSAE(device=args.device)
    log(f"  SAE loaded: layer={sae.layer} d_in={sae.d_in} d_sae={sae.d_sae}")
    log(f"  VRAM: {mem_gb():.2f} GB")

    summaries = []
    for domain in targets:
        log(f"\n{'=' * 60}")
        log(f"  DOMAIN: {domain}")
        log(f"{'=' * 60}")
        summary = decode_domain(sae, domain, limit=args.limit, overwrite=args.overwrite)
        summaries.append(summary)
        avg_mse = "n/a" if summary["avg_mse"] is None else f"{summary['avg_mse']:.2f}"
        log(
            f"  Saved {summary['num_prompts']} prompts to {summary['output_dir']}/ "
            f"(avg_nnz={summary['avg_nnz']:.1f}, avg_mse={avg_mse})"
        )

    log(f"\n{'=' * 60}")
    log("  SUMMARY")
    log(f"{'=' * 60}")
    for summary in summaries:
        avg_mse = "n/a" if summary["avg_mse"] is None else f"{summary['avg_mse']:.2f}"
        log(
            f"  {summary['adapter']:>8}: prompts={summary['num_prompts']:>3} "
            f"avg_nnz={summary['avg_nnz']:>8.1f} avg_mse={avg_mse:>8}"
        )
    log("  Day 11: COMPLETE")
