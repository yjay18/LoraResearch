"""
Day 9: Collect residual stream activations from base Gemma-2-2B on all 300 prompts.
Stores per-prompt activations as individual .npy files and metadata as JSON.

Output structure:
    data/activations/base/
        prompt_000.npy   # shape: [seq_len, d_model]
        prompt_001.npy
        ...
        prompt_299.npy
        metadata.json    # prompt texts, shapes, layer info
"""
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

TARGET_LAYER = 12  # middle layer, matches our SAE

# ============================================================
# Load model
# ============================================================
log("=" * 60)
log("  DAY 9: Collecting base model activations")
log("=" * 60)

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from lorasurgeon.collect import ResidualStreamCollector

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

log("Loading model...")
model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
log(f"  Model loaded: {mem_gb():.2f} GB VRAM")

# ============================================================
# Load prompts
# ============================================================
log("\nLoading prompts...")
with open("data/prompts_300.json", encoding="utf-8") as f:
    prompts = json.load(f)
log(f"  Loaded {len(prompts)} prompts")

# ============================================================
# Collect activations
# ============================================================
out_dir = "data/activations/base"
os.makedirs(out_dir, exist_ok=True)

metadata = {
    "model": "google/gemma-2-2b",
    "quantization": "4-bit NF4",
    "layer": TARGET_LAYER,
    "d_model": 2304,
    "num_prompts": len(prompts),
    "prompts": [],
}

log(f"\nCollecting layer {TARGET_LAYER} activations for {len(prompts)} prompts...")
start_time = time.time()

for i, prompt_data in enumerate(prompts):
    text = prompt_data["text"]
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    ).to("cuda")

    with ResidualStreamCollector(model, layers=[TARGET_LAYER]) as collector:
        with torch.no_grad():
            model(**inputs)
        act = collector.activations[TARGET_LAYER]  # [1, seq_len, d_model]

    # Squeeze batch dim and save as numpy
    act_np = act.squeeze(0).numpy()  # [seq_len, d_model]
    npy_path = os.path.join(out_dir, f"prompt_{i:03d}.npy")
    np.save(npy_path, act_np)

    metadata["prompts"].append({
        "id": prompt_data["id"],
        "domain": prompt_data["domain"],
        "source": prompt_data["source"],
        "text": text[:200],  # truncate for metadata
        "seq_len": act_np.shape[0],
        "shape": list(act_np.shape),
    })

    del inputs, act, act_np
    torch.cuda.empty_cache()

    if (i + 1) % 50 == 0 or i == 0:
        elapsed = time.time() - start_time
        rate = (i + 1) / elapsed
        eta = (len(prompts) - i - 1) / rate
        log(f"  [{i+1}/{len(prompts)}] {prompt_data['domain']:>10} | "
            f"seq_len={metadata['prompts'][-1]['seq_len']:>4} | "
            f"{rate:.1f} prompts/sec | ETA: {eta:.0f}s")

elapsed = time.time() - start_time

# Save metadata
meta_path = os.path.join(out_dir, "metadata.json")
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)

# ============================================================
# Verification
# ============================================================
log(f"\n{'='*60}")
log("  VERIFICATION")
log(f"{'='*60}")

# Check all files exist and shapes are consistent
total_size = 0
seq_lens = []
for i in range(len(prompts)):
    path = os.path.join(out_dir, f"prompt_{i:03d}.npy")
    assert os.path.exists(path), f"Missing {path}"
    arr = np.load(path)
    assert arr.shape[1] == 2304, f"Wrong d_model: {arr.shape}"
    assert arr.dtype == np.float32, f"Wrong dtype: {arr.dtype}"
    total_size += os.path.getsize(path)
    seq_lens.append(arr.shape[0])

from collections import Counter
domain_counts = Counter(p["domain"] for p in metadata["prompts"])

log(f"  Files: {len(prompts)} .npy files")
log(f"  Total size: {total_size / 1e6:.1f} MB")
log(f"  d_model: 2304 (all consistent)")
log(f"  Seq lengths: min={min(seq_lens)}, max={max(seq_lens)}, avg={sum(seq_lens)/len(seq_lens):.1f}")
log(f"  Domains: {dict(domain_counts)}")
log(f"  Time: {elapsed:.1f}s ({len(prompts)/elapsed:.1f} prompts/sec)")
log(f"  VRAM: {mem_gb():.2f} GB")
log(f"\n  Saved to: {out_dir}/")
log(f"  Metadata: {meta_path}")
log("  Day 9: COMPLETE")
