"""Day 1: Verify Gemma Scope SAEs load via SAELens"""
import torch
import gc

print("=== Loading Gemma Scope SAE via SAELens ===")

from sae_lens import SAE

# Load a Gemma Scope SAE for layer 12 (middle layer) of Gemma-2-2B
# Using the 16k width residual stream SAE
sae, cfg_dict, sparsity = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res",
    sae_id="layer_12/width_16k/average_l0_82",
    device="cpu",  # Load on CPU first to check VRAM
)

print(f"SAE loaded successfully!")
print(f"SAE input dim (d_in): {sae.cfg.d_in}")
print(f"SAE hidden dim (d_sae): {sae.cfg.d_sae}")
print(f"SAE dtype: {sae.dtype}")
print(f"Expansion factor: {sae.cfg.d_sae / sae.cfg.d_in:.0f}x")

# Test encode/decode on random input
print("\n=== Forward Pass Test (CPU) ===")
x = torch.randn(1, 10, sae.cfg.d_in, dtype=sae.dtype)
with torch.no_grad():
    feature_acts = sae.encode(x)
    recon = sae.decode(feature_acts)

print(f"Input shape: {x.shape}")
print(f"Feature activations shape: {feature_acts.shape}")
print(f"Reconstruction shape: {recon.shape}")
print(f"Active features (nonzero): {(feature_acts > 0).sum().item()} / {feature_acts.numel()}")
print(f"Sparsity: {(feature_acts > 0).float().mean().item():.4f}")
recon_loss = ((x - recon) ** 2).mean().item()
print(f"Reconstruction MSE: {recon_loss:.6f}")

print("\nGemma Scope SAE verified!")

# Cleanup
del sae, x, feature_acts, recon
gc.collect()
