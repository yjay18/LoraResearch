"""Day 1 verification: load Gemma-2-2B on GPU and run Gemma Scope SAE."""

import torch
import gc


def verify_gpu():
    """Check PyTorch + CUDA setup."""
    print("=== GPU Verification ===")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        raise RuntimeError("CUDA not available!")
    print()


def verify_gemma2():
    """Load Gemma-2-2B via TransformerLens and run a forward pass."""
    from transformer_lens import HookedTransformer

    print("=== Loading Gemma-2-2B via TransformerLens ===")
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b",
        device="cuda",
        dtype=torch.float16,
    )
    print(f"Model loaded: {model.cfg.model_name}")
    print(f"Layers: {model.cfg.n_layers}, d_model: {model.cfg.d_model}")
    print(f"Device: {next(model.parameters()).device}")

    # Run a simple forward pass
    prompt = "The capital of France is"
    tokens = model.to_tokens(prompt)
    print(f"Prompt: '{prompt}' -> {tokens.shape[1]} tokens")

    logits = model(tokens)
    next_token = model.tokenizer.decode(logits[0, -1].argmax().item())
    print(f"Next token prediction: '{next_token}'")

    # Check VRAM usage
    allocated = torch.cuda.memory_allocated() / 1024**3
    reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"VRAM allocated: {allocated:.2f} GB, reserved: {reserved:.2f} GB")
    print()

    # Clean up
    del model
    gc.collect()
    torch.cuda.empty_cache()


def verify_sae():
    """Load a Gemma Scope SAE via SAELens and run encode/decode."""
    from sae_lens import SAE

    print("=== Loading Gemma Scope SAE via SAELens ===")
    # Load a Gemma Scope SAE for layer 12 (middle layer), 16k features
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res",
        sae_id="layer_12/width_16k/average_l0_82",
        device="cuda",
    )
    print(f"SAE loaded: {sae}")
    print(f"d_in: {sae.cfg.d_in}, d_sae: {sae.cfg.d_sae}")
    print(f"Device: {next(sae.parameters()).device}")

    # Test encode/decode with random activations
    test_acts = torch.randn(1, sae.cfg.d_in, device="cuda", dtype=torch.float32)
    feature_acts = sae.encode(test_acts)
    recon = sae.decode(feature_acts)

    # Check sparsity
    nonzero = (feature_acts > 0).float().sum().item()
    total = feature_acts.numel()
    print(f"Sparsity: {nonzero}/{total} features active ({nonzero/total*100:.1f}%)")
    print(f"Reconstruction shape: {recon.shape}")

    # Reconstruction loss
    loss = torch.nn.functional.mse_loss(recon, test_acts).item()
    print(f"Reconstruction MSE (random input): {loss:.4f}")

    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM after SAE: {allocated:.2f} GB")
    print()

    del sae
    gc.collect()
    torch.cuda.empty_cache()


def verify_sae_with_model():
    """Load both model and SAE, run activations through SAE."""
    from transformer_lens import HookedTransformer
    from sae_lens import SAE

    print("=== End-to-end: Model activations -> SAE encode -> decode ===")

    # Load model
    model = HookedTransformer.from_pretrained(
        "gemma-2-2b",
        device="cuda",
        dtype=torch.float16,
    )

    # Load SAE for same layer
    sae, _, _ = SAE.from_pretrained(
        release="gemma-scope-2b-pt-res",
        sae_id="layer_12/width_16k/average_l0_82",
        device="cuda",
    )

    # Run model and collect residual stream at layer 12
    prompt = "The mitochondria is the powerhouse of the cell"
    tokens = model.to_tokens(prompt)

    _, cache = model.run_with_cache(
        tokens,
        names_filter=["blocks.12.hook_resid_post"],
    )
    residual = cache["blocks.12.hook_resid_post"]  # (batch, seq, d_model)
    print(f"Residual stream shape: {residual.shape}")

    # Flatten and encode through SAE
    flat_acts = residual.reshape(-1, residual.shape[-1]).float()
    feature_acts = sae.encode(flat_acts)
    recon = sae.decode(feature_acts)

    nonzero_per_token = (feature_acts > 0).float().sum(dim=-1).mean().item()
    print(f"Avg active features per token: {nonzero_per_token:.1f}")

    recon_loss = torch.nn.functional.mse_loss(recon, flat_acts).item()
    print(f"Reconstruction MSE (real activations): {recon_loss:.4f}")

    # Top features for last token
    last_token_feats = feature_acts[-1]
    top_k = 10
    top_vals, top_ids = last_token_feats.topk(top_k)
    print(f"Top {top_k} features for last token:")
    for i in range(top_k):
        print(f"  Feature {top_ids[i].item()}: {top_vals[i].item():.3f}")

    allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"VRAM with model + SAE: {allocated:.2f} GB")
    print()

    del model, sae, cache
    gc.collect()
    torch.cuda.empty_cache()
    print("=== All Day 1 verifications PASSED ===")


if __name__ == "__main__":
    verify_gpu()
    verify_gemma2()
    verify_sae()
    verify_sae_with_model()
