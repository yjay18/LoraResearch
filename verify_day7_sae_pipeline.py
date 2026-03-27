"""
Day 7 Verification: SAE forward pass on real Gemma-2-2B activations.
Loads base model → hooks residual stream → SAE encode → decode → reconstruction loss.
"""
import sys
import json
import os
import torch
import gc

sys.path.insert(0, os.path.dirname(__file__))

def log(msg):
    print(msg, flush=True)

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()

def mem_gb():
    return torch.cuda.memory_allocated() / 1e9

# ============================================================
# STEP 1: Load base model (4-bit)
# ============================================================
log("=" * 60)
log("STEP 1: Load Gemma-2-2B (4-bit NF4)")
log("=" * 60)
clear_gpu()

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    quantization_config=bnb_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
log(f"  Model loaded: {mem_gb():.2f} GB VRAM")

# ============================================================
# STEP 2: Load SAE via our module
# ============================================================
log("\n" + "=" * 60)
log("STEP 2: Load Gemma Scope SAE (layer 12, 16k width)")
log("=" * 60)

from lorasurgeon.sae_decode import GemmaScopeSAE

sae = GemmaScopeSAE(device="cuda")
log(f"  SAE loaded: d_in={sae.d_in}, d_sae={sae.d_sae}, layer={sae.layer}")
log(f"  VRAM after SAE: {mem_gb():.2f} GB")

# ============================================================
# STEP 3: Collect activations via our module
# ============================================================
log("\n" + "=" * 60)
log("STEP 3: Collect residual stream activations")
log("=" * 60)

from lorasurgeon.collect import ResidualStreamCollector

test_prompts = [
    "The capital of France is",
    "Write a Python function to sort a list",
    "The patient presents with acute chest pain",
    "Solve for x: 2x + 5 = 13",
    "Once upon a time in a distant galaxy",
]

results = []

for i, prompt in enumerate(test_prompts):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    with ResidualStreamCollector(model, layers=[12]) as collector:
        with torch.no_grad():
            model(**inputs)
        resid = collector.activations[12]

    # Run SAE forward pass
    sae_result = sae.forward(resid)

    # Get top features for the last token
    top_vals, top_ids = sae.top_features(sae_result.feature_acts, k=10, token_idx=-1)

    log(f"\n  Prompt {i+1}: '{prompt}'")
    log(f"    Activation shape: {resid.shape}")
    log(f"    Reconstruction MSE: {sae_result.mse:.6f}")
    log(f"    L0 (avg active features): {sae_result.l0:.1f}")
    log(f"    Sparsity: {sae_result.sparsity:.4f}")
    log(f"    Top 10 features (last token): {top_ids[0].tolist()}")
    log(f"    Top 10 values: {[f'{v:.3f}' for v in top_vals[0].tolist()]}")

    results.append({
        "prompt": prompt,
        "seq_len": resid.shape[1],
        "mse": sae_result.mse,
        "l0": sae_result.l0,
        "sparsity": sae_result.sparsity,
        "top_features": top_ids[0].tolist(),
        "top_values": [round(v, 4) for v in top_vals[0].tolist()],
    })

    del inputs, resid, sae_result
    torch.cuda.empty_cache()

# ============================================================
# STEP 4: Summary + save results
# ============================================================
log("\n" + "=" * 60)
log("SUMMARY")
log("=" * 60)

avg_mse = sum(r["mse"] for r in results) / len(results)
avg_l0 = sum(r["l0"] for r in results) / len(results)
avg_sparsity = sum(r["sparsity"] for r in results) / len(results)

log(f"  Prompts tested: {len(results)}")
log(f"  Avg reconstruction MSE: {avg_mse:.6f}")
log(f"  Avg L0 (active features): {avg_l0:.1f}")
log(f"  Avg sparsity: {avg_sparsity:.4f}")
log(f"  VRAM used: {mem_gb():.2f} GB")
log(f"  SAE forward pass: WORKING")

# Save log
os.makedirs("results", exist_ok=True)
log_data = {
    "day": 7,
    "task": "SAE forward pass verification",
    "model": "google/gemma-2-2b",
    "sae": "gemma-scope-2b-pt-res/layer_12/width_16k/average_l0_82",
    "avg_mse": avg_mse,
    "avg_l0": avg_l0,
    "avg_sparsity": avg_sparsity,
    "vram_gb": round(mem_gb(), 2),
    "per_prompt": results,
}
with open("results/day7_sae_verification.json", "w") as f:
    json.dump(log_data, f, indent=2)
log(f"\n  Results saved to results/day7_sae_verification.json")
log("  Day 7: COMPLETE")
