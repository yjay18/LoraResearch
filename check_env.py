import torch

print("=== PyTorch ===")
print(f"Version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"VRAM: {round(props.total_memory / 1e9, 1)} GB")
    print(f"CUDA version: {torch.version.cuda}")
else:
    print("WARNING: No CUDA GPU detected!")

print("\n=== Key Packages ===")
from importlib.metadata import version as pkg_version
for pkg in ["transformer-lens", "sae-lens", "peft", "transformers", "huggingface-hub"]:
    print(f"{pkg}: {pkg_version(pkg)}")

print("\n=== Quick GPU Test ===")
if torch.cuda.is_available():
    x = torch.randn(1000, 1000, device="cuda")
    y = x @ x.T
    print(f"Matrix multiply on GPU: OK (result shape {y.shape})")
    del x, y
    torch.cuda.empty_cache()
    print("GPU compute verified!")
else:
    print("Skipped - no GPU")
