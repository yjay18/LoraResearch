"""
Day 10: Collect residual stream activations from 5 LoRA-adapted models on all 300 prompts.
Reuses the same base model, swapping adapters in/out to save GPU memory and load time.

Output structure:
    data/activations/{domain}/
        prompt_000.npy   # shape: [seq_len, d_model]
        ...
        prompt_299.npy
        metadata.json
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import gc
import time

sys.path.insert(0, os.path.dirname(__file__))

def log(msg):
    print(msg, flush=True)

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def mem_gb():
    return torch.cuda.memory_allocated() / 1e9

TARGET_LAYER = 12
ADAPTER_ROOT = "results/adapters"
DOMAINS = ["code", "medical", "math", "safety", "creative"]
BASE_MODEL = "google/gemma-2-2b"


def collect_for_adapter(model, tokenizer, prompts, domain, out_dir):
    """Collect layer-12 activations for all prompts with current model."""
    from lorasurgeon.collect import ResidualStreamCollector

    os.makedirs(out_dir, exist_ok=True)

    metadata = {
        "model": BASE_MODEL,
        "adapter": domain,
        "quantization": "4-bit NF4",
        "layer": TARGET_LAYER,
        "d_model": 2304,
        "num_prompts": len(prompts),
        "prompts": [],
    }

    start = time.time()
    for i, prompt_data in enumerate(prompts):
        text = prompt_data["text"]
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=512,
        ).to("cuda")

        with ResidualStreamCollector(model, layers=[TARGET_LAYER]) as collector:
            with torch.no_grad():
                model(**inputs)
            act = collector.activations[TARGET_LAYER]

        act_np = act.squeeze(0).numpy()  # [seq_len, d_model]
        np.save(os.path.join(out_dir, f"prompt_{i:03d}.npy"), act_np)

        metadata["prompts"].append({
            "id": prompt_data["id"],
            "domain": prompt_data["domain"],
            "source": prompt_data["source"],
            "text": text[:200],
            "seq_len": act_np.shape[0],
            "shape": list(act_np.shape),
        })

        del inputs, act, act_np
        torch.cuda.empty_cache()

        if (i + 1) % 100 == 0 or i == 0:
            elapsed = time.time() - start
            rate = (i + 1) / elapsed
            eta = (len(prompts) - i - 1) / rate
            log(f"    [{i+1}/{len(prompts)}] {rate:.1f} prompts/sec | ETA: {eta:.0f}s")

    elapsed = time.time() - start
    with open(os.path.join(out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    return elapsed


def verify_shapes(base_dir, adapted_dir, n_prompts):
    """Verify adapted activations match base shapes exactly."""
    mismatches = 0
    for i in range(n_prompts):
        base_path = os.path.join(base_dir, f"prompt_{i:03d}.npy")
        adapted_path = os.path.join(adapted_dir, f"prompt_{i:03d}.npy")
        base_arr = np.load(base_path)
        adapted_arr = np.load(adapted_path)
        if base_arr.shape != adapted_arr.shape:
            log(f"    MISMATCH prompt {i}: base={base_arr.shape} vs adapted={adapted_arr.shape}")
            mismatches += 1
    return mismatches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", choices=DOMAINS + ["all"], default="all")
    args = parser.parse_args()
    domains = DOMAINS if args.domain == "all" else [args.domain]

    log("=" * 60)
    log("  DAY 10: Collecting adapted model activations")
    log("=" * 60)

    # Load prompts
    with open("data/prompts_300.json", encoding="utf-8") as f:
        prompts = json.load(f)
    log(f"Loaded {len(prompts)} prompts")

    # Load base model once
    log("\nLoading base model (4-bit)...")
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, quantization_config=bnb_config, device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    log(f"  Base model loaded: {mem_gb():.2f} GB")

    # Process each adapter
    for domain in domains:
        log(f"\n{'='*60}")
        log(f"  ADAPTER: {domain}")
        log(f"{'='*60}")

        adapter_path = os.path.join(ADAPTER_ROOT, domain)
        out_dir = f"data/activations/{domain}"

        # Load adapter on top of base
        log(f"  Loading {domain} adapter from {adapter_path}...")
        model = PeftModel.from_pretrained(base_model, adapter_path)
        log(f"  VRAM: {mem_gb():.2f} GB")

        # Collect
        log(f"  Collecting activations...")
        elapsed = collect_for_adapter(model, tokenizer, prompts, domain, out_dir)
        log(f"  Done: {elapsed:.1f}s ({len(prompts)/elapsed:.1f} prompts/sec)")

        # Verify shapes match base
        log(f"  Verifying shapes vs base...")
        mismatches = verify_shapes("data/activations/base", out_dir, len(prompts))
        if mismatches == 0:
            log(f"  All {len(prompts)} shapes match base model activations")
        else:
            log(f"  WARNING: {mismatches} shape mismatches!")

        # Unload adapter
        del model
        clear_gpu()
        log(f"  Adapter unloaded, VRAM: {mem_gb():.2f} GB")

    # Final summary
    log(f"\n{'='*60}")
    log("  SUMMARY")
    log(f"{'='*60}")
    total_size = 0
    for subdir in ["base"] + domains:
        d = f"data/activations/{subdir}"
        if os.path.exists(d):
            size = sum(os.path.getsize(os.path.join(d, f)) for f in os.listdir(d) if f.endswith(".npy"))
            total_size += size
            log(f"  {subdir:>10}: {size/1e6:.1f} MB ({len([f for f in os.listdir(d) if f.endswith('.npy')])} files)")

    log(f"  {'TOTAL':>10}: {total_size/1e6:.1f} MB")
    log(f"  Activation sets: {1 + len(domains)} (base + {len(domains)} adapted)")
    log("  Day 10: COMPLETE")
