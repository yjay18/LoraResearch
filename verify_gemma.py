"""Day 1: Verify Gemma-2-2B loads on GPU via TransformerLens"""
import torch
import gc

print("=== Loading Gemma-2-2B via TransformerLens ===")
print(f"GPU memory before: {torch.cuda.memory_allocated()/1e9:.2f} GB")

from transformer_lens import HookedTransformer

model = HookedTransformer.from_pretrained(
    "gemma-2-2b",
    device="cuda",
    dtype=torch.float16,
)

print(f"Model loaded: {model.cfg.model_name}")
print(f"Layers: {model.cfg.n_layers}")
print(f"Hidden dim: {model.cfg.d_model}")
print(f"Heads: {model.cfg.n_heads}")
print(f"GPU memory after: {torch.cuda.memory_allocated()/1e9:.2f} GB")

# Quick inference test
print("\n=== Quick Inference Test ===")
prompt = "The capital of France is"
tokens = model.to_tokens(prompt)
print(f"Prompt: '{prompt}'")
print(f"Tokens: {tokens.shape}")

with torch.no_grad():
    logits = model(tokens)
    next_token = logits[0, -1].argmax()
    next_word = model.tokenizer.decode(next_token)
    print(f"Next token prediction: '{next_word}'")

# Test residual stream hook
print("\n=== Residual Stream Hook Test ===")
with torch.no_grad():
    logits, cache = model.run_with_cache(tokens)
    resid = cache["resid_post", model.cfg.n_layers - 1]
    print(f"Residual stream shape (last layer): {resid.shape}")
    print(f"Residual stream dtype: {resid.dtype}")

print("\n✓ Gemma-2-2B verified on GPU!")

# Cleanup
del model, logits, cache
gc.collect()
torch.cuda.empty_cache()
print("Cleaned up GPU memory.")
