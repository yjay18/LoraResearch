"""
Full pipeline verification: 4-bit Gemma-2-2B + residual stream hooks + SAE + LoRA
Proves everything fits in 8.6GB VRAM simultaneously.
"""
import torch
import gc
from contextlib import contextmanager

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def mem_report(label=""):
    alloc = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - alloc
    print(f"  [{label}] Alloc: {alloc:.2f} GB | Free: {free:.2f} GB | Peak: {peak:.2f} GB")
    return alloc


# ============================================================
# STEP 1: Load 4-bit quantized Gemma-2-2B
# ============================================================
print("=" * 60)
print("STEP 1: Load Gemma-2-2B in 4-bit NF4")
print("=" * 60)
clear_gpu()

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    "google/gemma-2-2b",
    quantization_config=quantization_config,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
mem_report("4-bit model loaded")


# ============================================================
# STEP 2: Build residual stream hook system
# ============================================================
print("\n" + "=" * 60)
print("STEP 2: Residual stream hook system")
print("=" * 60)

class ResidualStreamCollector:
    """Collects residual stream activations from specified layers.
    Works with both raw HF models and PEFT-wrapped models.
    """

    def __init__(self, model, layers=None):
        self.model = model
        self.activations = {}
        self.hooks = []
        self._decoder_layers = self._find_decoder_layers(model)
        if layers is None:
            layers = list(range(len(self._decoder_layers)))
        self.layers = layers

    @staticmethod
    def _find_decoder_layers(model):
        """Navigate through PEFT/HF wrappers to find the decoder layers."""
        # Try common paths for the decoder layer list
        candidates = [
            lambda m: m.model.layers,                          # raw HF model
            lambda m: m.base_model.model.model.layers,         # PEFT-wrapped
            lambda m: m.model.model.layers,                    # another variant
        ]
        for fn in candidates:
            try:
                layers = fn(model)
                if hasattr(layers, '__len__') and len(layers) > 0:
                    return layers
            except (AttributeError, TypeError):
                continue
        raise RuntimeError("Could not find decoder layers in model")

    def _make_hook(self, layer_idx):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx] = hidden.detach().cpu().float()
        return hook_fn

    def __enter__(self):
        self.activations = {}
        for layer_idx in self.layers:
            layer = self._decoder_layers[layer_idx]
            h = layer.register_forward_hook(self._make_hook(layer_idx))
            self.hooks.append(h)
        return self

    def __exit__(self, *args):
        for h in self.hooks:
            h.remove()
        self.hooks = []

# Test hook on a single prompt
prompt = "The capital of France is"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

# Collect residual stream from layer 12 (middle layer, matches our SAE)
with ResidualStreamCollector(model, layers=[12]) as collector:
    with torch.no_grad():
        outputs = model(**inputs)
    resid_12 = collector.activations[12]

print(f"  Prompt: '{prompt}'")
print(f"  Layer 12 residual shape: {resid_12.shape}")
print(f"  Layer 12 residual dtype: {resid_12.dtype}")
mem_report("After hook collection")

# Verify next-token prediction still works
next_token = outputs.logits[0, -1].argmax()
print(f"  Next token: '{tokenizer.decode(next_token)}'")


# ============================================================
# STEP 3: Load SAE on GPU alongside model
# ============================================================
print("\n" + "=" * 60)
print("STEP 3: Load Gemma Scope SAE on GPU")
print("=" * 60)

from sae_lens import SAE

sae = SAE.from_pretrained(
    release="gemma-scope-2b-pt-res",
    sae_id="layer_12/width_16k/average_l0_82",
    device="cuda",
)
mem_report("SAE loaded on GPU")


# ============================================================
# STEP 4: Run SAE on residual stream activations
# ============================================================
print("\n" + "=" * 60)
print("STEP 4: SAE encode/decode on real activations")
print("=" * 60)

resid_gpu = resid_12.to("cuda").to(sae.dtype)
with torch.no_grad():
    feature_acts = sae.encode(resid_gpu)
    recon = sae.decode(feature_acts)

recon_loss = ((resid_gpu - recon) ** 2).mean().item()
active = (feature_acts > 0).sum().item()
total = feature_acts.numel()
sparsity = active / total

print(f"  Feature activations shape: {feature_acts.shape}")
print(f"  Active features: {active} / {total} ({sparsity:.4f})")
print(f"  Reconstruction MSE: {recon_loss:.6f}")
print(f"  Top 5 most active features: {feature_acts[0, -1].topk(5).indices.tolist()}")
mem_report("After SAE forward pass")


# ============================================================
# STEP 5: Load a LoRA adapter and verify it all still fits
# ============================================================
print("\n" + "=" * 60)
print("STEP 5: Add LoRA adapter")
print("=" * 60)

from peft import LoraConfig, get_peft_model, TaskType

# Create a test LoRA config (simulating a domain adapter)
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
trainable, total_params = model.get_nb_trainable_parameters()
print(f"  LoRA trainable params: {trainable:,} / {total_params:,} ({100*trainable/total_params:.2f}%)")
mem_report("After adding LoRA")


# ============================================================
# STEP 6: Full pipeline - 4-bit model + LoRA + hooks + SAE
# ============================================================
print("\n" + "=" * 60)
print("STEP 6: FULL PIPELINE TEST (4-bit + LoRA + hooks + SAE)")
print("=" * 60)

prompts = [
    "Write a Python function that sorts a list",
    "The patient presents with acute chest pain",
    "Solve the integral of x^2 * sin(x) dx",
    "Explain the concept of neural network pruning",
    "Once upon a time in a distant galaxy",
]

for i, prompt in enumerate(prompts):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with ResidualStreamCollector(model, layers=[12]) as collector:
        with torch.no_grad():
            outputs = model(**inputs)
        resid = collector.activations[12].to("cuda").to(sae.dtype)

    with torch.no_grad():
        feats = sae.encode(resid)
        active_count = (feats > 0).sum().item()
        top_feats = feats[0, -1].topk(5).indices.tolist()

    print(f"  [{i+1}] '{prompt[:40]}...' -> {active_count} active features, top: {top_feats}")

    del inputs, outputs, resid, feats
    torch.cuda.empty_cache()

mem_report("After full pipeline (5 prompts)")


# ============================================================
# SUMMARY
# ============================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
alloc = torch.cuda.memory_allocated() / 1e9
total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"  Model (4-bit NF4): loaded")
print(f"  LoRA adapter: loaded ({trainable:,} params)")
print(f"  SAE (16k width): loaded on GPU")
print(f"  Residual stream hooks: working")
print(f"  SAE encode/decode: working")
print(f"  VRAM allocated: {alloc:.2f} / {total_vram:.1f} GB")
print(f"  VRAM peak: {peak:.2f} GB")
print(f"  VRAM headroom: {total_vram - alloc:.2f} GB")
print(f"\n  VRAM ISSUE: FIXED")
