"""
Environment verification script for LoRASurgeon.
Checks that all dependencies load correctly and Gemma-2-2B fits on the RTX 3070.
"""

import sys
from pathlib import Path

def check_torch():
    """Verify PyTorch + CUDA setup."""
    print("=" * 60)
    print("1. Checking PyTorch + CUDA...")
    print("=" * 60)

    import torch
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")

    if not torch.cuda.is_available():
        print("  ERROR: CUDA not available! Check your GPU drivers.")
        return False

    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Quick GPU compute test
    x = torch.randn(1000, 1000, device="cuda")
    y = torch.matmul(x, x)
    print(f"  GPU compute test: OK (matrix multiply on 1000x1000)")

    # Check VRAM
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"  Total VRAM: {total_vram:.1f} GB")

    del x, y
    torch.cuda.empty_cache()
    return True


def check_imports():
    """Verify all key packages import correctly."""
    print("\n" + "=" * 60)
    print("2. Checking package imports...")
    print("=" * 60)

    packages = [
        ("transformers", "transformers"),
        ("peft", "peft"),
        ("transformer_lens", "transformer_lens"),
        ("sae_lens", "sae_lens"),
        ("bitsandbytes", "bitsandbytes"),
        ("accelerate", "accelerate"),
        ("datasets", "datasets"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("matplotlib", "matplotlib"),
    ]

    all_ok = True
    for display_name, import_name in packages:
        try:
            mod = __import__(import_name)
            version = getattr(mod, "__version__", "unknown")
            print(f"  {display_name}: {version}")
        except ImportError as e:
            print(f"  {display_name}: FAILED - {e}")
            all_ok = False

    return all_ok


def check_gemma_2b():
    """Load Gemma-2-2B using TransformerLens (via SAELens wrapper) and verify it fits in VRAM."""
    print("\n" + "=" * 60)
    print("3. Loading Gemma-2-2B on GPU...")
    print("=" * 60)

    import torch
    from sae_lens import HookedSAETransformer

    # Clear VRAM before loading
    torch.cuda.empty_cache()
    vram_before = torch.cuda.memory_allocated() / 1024**3
    print(f"  VRAM before loading: {vram_before:.2f} GB")

    print("  Loading model (this may take a minute on first run)...")
    # from_pretrained_no_processing is important — SAEs are trained on raw activations
    model = HookedSAETransformer.from_pretrained_no_processing(
        "gemma-2-2b",
        device="cuda",
        dtype=torch.float16,  # fp16 to fit in 8GB
    )

    vram_after = torch.cuda.memory_allocated() / 1024**3
    print(f"  VRAM after loading: {vram_after:.2f} GB")
    print(f"  Model VRAM usage: {vram_after - vram_before:.2f} GB")

    # Quick forward pass test
    print("  Running test forward pass...")
    test_input = model.to_tokens("Hello, world!")
    with torch.no_grad():
        output = model(test_input)

    print(f"  Output shape: {output.shape}")
    print(f"  Model loaded and runs correctly!")

    # Report final VRAM state
    vram_peak = torch.cuda.max_memory_allocated() / 1024**3
    print(f"  Peak VRAM usage: {vram_peak:.2f} GB")

    # Clean up
    del model, output, test_input
    torch.cuda.empty_cache()

    return True


def main():
    print("LoRASurgeon Environment Verification")
    print(f"Python: {sys.version}")
    print()

    # Step 1: PyTorch + CUDA
    if not check_torch():
        print("\nFAILED: Fix CUDA before proceeding.")
        return

    # Step 2: Package imports
    if not check_imports():
        print("\nWARNING: Some packages failed to import.")

    # Step 3: Load Gemma-2-2B
    try:
        check_gemma_2b()
    except Exception as e:
        print(f"\n  ERROR loading Gemma-2-2B: {e}")
        print("  This might be a VRAM issue or model access issue.")
        print("  Make sure you've accepted the Gemma license on HuggingFace.")
        raise

    print("\n" + "=" * 60)
    print("ALL CHECKS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
