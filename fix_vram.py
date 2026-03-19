"""
VRAM Optimization: Find a configuration that fits Gemma-2-2B + SAE + LoRA in 8.6GB

Approaches to test:
1. TransformerLens with from_pretrained_no_processing (less overhead)
2. 8-bit quantization via bitsandbytes
3. 4-bit quantization via bitsandbytes (QLoRA style)
4. Direct transformers loading with quantization + manual hook
"""
import torch
import gc
import sys

def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

def mem_report(label=""):
    alloc = torch.cuda.memory_allocated() / 1e9
    peak = torch.cuda.max_memory_allocated() / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    free = total - alloc
    print(f"[{label}] Allocated: {alloc:.2f} GB | Peak: {peak:.2f} GB | Free: {free:.2f} GB")
    return alloc

# ============================================================
# APPROACH 1: TransformerLens from_pretrained_no_processing
# ============================================================
print("=" * 60)
print("APPROACH 1: TransformerLens from_pretrained_no_processing")
print("=" * 60)
clear_gpu()

try:
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained_no_processing(
        "gemma-2-2b",
        device="cuda",
        dtype=torch.float16,
    )
    vram_model = mem_report("Model loaded")

    # Test inference
    tokens = model.to_tokens("The capital of France is")
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
        resid = cache["resid_post", model.cfg.n_layers - 1]
        vram_with_cache = mem_report("With cache")
        print(f"Residual shape: {resid.shape}")

    del model, logits, cache, resid, tokens
    clear_gpu()
    print(f"Result: Model={vram_model:.2f}GB, With cache={vram_with_cache:.2f}GB")
except Exception as e:
    print(f"FAILED: {e}")
    clear_gpu()

# ============================================================
# APPROACH 2: Transformers + 8-bit quantization
# ============================================================
print("\n" + "=" * 60)
print("APPROACH 2: Transformers + 8-bit quantization")
print("=" * 60)
clear_gpu()

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    model_8bit = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        quantization_config=quantization_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    vram_model = mem_report("8-bit model loaded")

    # Test inference
    inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model_8bit(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        vram_with_hidden = mem_report("With hidden states")
        print(f"Last hidden shape: {last_hidden.shape}")

    del model_8bit, tokenizer, inputs, outputs, last_hidden
    clear_gpu()
    print(f"Result: Model={vram_model:.2f}GB, With hidden={vram_with_hidden:.2f}GB")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback; traceback.print_exc()
    clear_gpu()

# ============================================================
# APPROACH 3: Transformers + 4-bit quantization (NF4)
# ============================================================
print("\n" + "=" * 60)
print("APPROACH 3: Transformers + 4-bit NF4 quantization")
print("=" * 60)
clear_gpu()

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model_4bit = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        quantization_config=quantization_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b")
    vram_model = mem_report("4-bit model loaded")

    # Test inference with hidden states
    inputs = tokenizer("The capital of France is", return_tensors="pt").to("cuda")
    with torch.no_grad():
        outputs = model_4bit(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1]
        vram_with_hidden = mem_report("With hidden states")
        print(f"Last hidden shape: {last_hidden.shape}, dtype: {last_hidden.dtype}")

    del model_4bit, tokenizer, inputs, outputs, last_hidden
    clear_gpu()
    print(f"Result: Model={vram_model:.2f}GB, With hidden={vram_with_hidden:.2f}GB")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback; traceback.print_exc()
    clear_gpu()

# ============================================================
# APPROACH 4: TransformerLens with 8-bit via from_pretrained
# ============================================================
print("\n" + "=" * 60)
print("APPROACH 4: TransformerLens + 8-bit quantization")
print("=" * 60)
clear_gpu()

try:
    from transformer_lens import HookedTransformer
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    hf_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-2b",
        quantization_config=quantization_config,
        device_map="auto",
    )

    model = HookedTransformer.from_pretrained(
        "gemma-2-2b",
        hf_model=hf_model,
        device="cuda",
        dtype=torch.float16,
    )
    del hf_model
    clear_gpu()
    vram_model = mem_report("TL + 8-bit loaded")

    tokens = model.to_tokens("The capital of France is")
    with torch.no_grad():
        logits, cache = model.run_with_cache(tokens)
        resid = cache["resid_post", model.cfg.n_layers - 1]
        vram_with_cache = mem_report("With cache")
        print(f"Residual shape: {resid.shape}")

    del model, logits, cache, resid, tokens
    clear_gpu()
    print(f"Result: Model={vram_model:.2f}GB, With cache={vram_with_cache:.2f}GB")
except Exception as e:
    print(f"FAILED: {e}")
    import traceback; traceback.print_exc()
    clear_gpu()

print("\n" + "=" * 60)
print("DONE - Compare results above to pick best approach")
print("=" * 60)
