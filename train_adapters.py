"""
Train 5 domain-specific QLoRA adapters on google/gemma-2-2b (base model).
Consistent hyperparameters across all domains for research comparability.

Domains: code, medical, math, safety, creative
"""

import argparse
import glob
import os
import shutil
import sys
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig

# ============================================================
# Constants — identical across all adapters for consistency
# ============================================================
BASE_MODEL = "google/gemma-2-2b"
OUTPUT_ROOT = "results/adapters"
MAX_SEQ_LEN = 512
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj"]
LEARNING_RATE = 2e-4
NUM_TRAIN_STEPS = 50
BATCH_SIZE = 2
GRAD_ACCUM = 2  # effective batch size = 4
WARMUP_STEPS = 10
SAVE_STEPS = 50
LOGGING_STEPS = 10

# ============================================================
# Dataset configs — each returns a formatted dataset
# ============================================================

def load_code_dataset():
    """CodeAlpaca-20k: instruction→code pairs"""
    ds = load_dataset("sahil2801/CodeAlpaca-20k", split="train")
    def fmt(example):
        prompt = example.get("prompt", "")
        completion = example.get("completion", "")
        return {"text": f"### Instruction:\n{prompt}\n\n### Response:\n{completion}"}
    return ds.map(fmt, remove_columns=ds.column_names).shuffle(seed=42).select(range(min(2000, len(ds))))

def load_medical_dataset():
    """Medical Meadow flashcards: Q&A pairs"""
    ds = load_dataset("medalpaca/medical_meadow_medical_flashcards", split="train")
    def fmt(example):
        inp = example.get("input", "")
        out = example.get("output", "")
        return {"text": f"### Question:\n{inp}\n\n### Answer:\n{out}"}
    return ds.map(fmt, remove_columns=ds.column_names).shuffle(seed=42).select(range(min(2000, len(ds))))

def load_math_dataset():
    """GSM8K: grade school math with chain-of-thought"""
    ds = load_dataset("openai/gsm8k", "main", split="train")
    def fmt(example):
        q = example.get("question", "")
        a = example.get("answer", "")
        return {"text": f"### Problem:\n{q}\n\n### Solution:\n{a}"}
    return ds.map(fmt, remove_columns=ds.column_names).shuffle(seed=42).select(range(min(2000, len(ds))))

def load_safety_dataset():
    """Anthropic HH-RLHF: chosen (safe) responses"""
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="train")
    def fmt(example):
        chosen = example.get("chosen", "")
        return {"text": chosen}
    return ds.map(fmt, remove_columns=ds.column_names).shuffle(seed=42).select(range(min(2000, len(ds))))

def load_creative_dataset():
    """WritingPrompts: creative fiction writing"""
    ds = load_dataset("euclaise/writingprompts", split="train")
    def fmt(example):
        title = example.get("title", "")
        text = example.get("text", "")
        # Truncate long stories to keep training manageable
        text = text[:2000] if len(text) > 2000 else text
        return {"text": f"### Prompt:\n{title}\n\n### Story:\n{text}"}
    return ds.map(fmt, remove_columns=ds.column_names).shuffle(seed=42).select(range(min(2000, len(ds))))


def log(msg):
    """Print with immediate flush for monitoring."""
    print(msg, flush=True)


DOMAIN_LOADERS = {
    "code": load_code_dataset,
    "medical": load_medical_dataset,
    "math": load_math_dataset,
    "safety": load_safety_dataset,
    "creative": load_creative_dataset,
}

ESSENTIAL_ADAPTER_FILES = {
    "adapter_config.json",
    "adapter_model.safetensors",
}


def prune_adapter_artifacts(output_dir: str):
    """
    Keep only the PEFT files needed to reload the adapter.

    The base tokenizer is always loaded from BASE_MODEL, so local tokenizer copies,
    trainer metadata, and checkpoint directories only add workspace bloat.
    """
    for checkpoint_dir in glob.glob(os.path.join(output_dir, "checkpoint-*")):
        shutil.rmtree(checkpoint_dir, ignore_errors=True)

    for entry in os.listdir(output_dir):
        entry_path = os.path.join(output_dir, entry)
        if os.path.isdir(entry_path):
            shutil.rmtree(entry_path, ignore_errors=True)
            continue
        if entry not in ESSENTIAL_ADAPTER_FILES:
            os.remove(entry_path)


def train_adapter(domain: str):
    """Train a single QLoRA adapter for the given domain."""
    log(f"\n{'='*60}")
    log(f"  TRAINING: {domain} adapter")
    log(f"{'='*60}\n")

    output_dir = os.path.join(OUTPUT_ROOT, domain)
    os.makedirs(output_dir, exist_ok=True)

    # Load dataset
    log(f"Loading {domain} dataset...")
    dataset = DOMAIN_LOADERS[domain]()
    log(f"  Dataset size: {len(dataset)} examples")

    # 4-bit quantization config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    # Load model and tokenizer
    log("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.eos_token_id

    # LoRA config — consistent across all domains
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )

    # SFT config (replaces TrainingArguments in trl >= 0.29)
    sft_config = SFTConfig(
        output_dir=output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=2,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to="none",
        dataloader_pin_memory=False,
        max_length=MAX_SEQ_LEN,
        dataset_text_field="text",
        packing=False,
    )

    # SFT Trainer
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    log("Starting training...")
    trainer.train()

    # Save adapter only (not the full model)
    log(f"Saving adapter to {output_dir}...")
    trainer.save_model(output_dir)

    # Cleanup GPU
    del model, trainer
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    log(f"\n  {domain} adapter saved to {output_dir}")
    return output_dir


def verify_adapter(domain: str):
    """Quick verification that an adapter loads and produces output."""
    log(f"\nVerifying {domain} adapter...")
    adapter_path = os.path.join(OUTPUT_ROOT, domain)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)

    from peft import PeftModel
    model = PeftModel.from_pretrained(model, adapter_path)

    test_prompts = {
        "code": "Write a Python function to sort a list",
        "medical": "What are the symptoms of pneumonia",
        "math": "Solve: If x + 5 = 12, what is x",
        "safety": "How should I handle a conflict with a coworker",
        "creative": "Write a story about a dragon",
    }

    prompt = test_prompts.get(domain, "Hello, how are you?")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=50)
    response = tokenizer.decode(out[0], skip_special_tokens=True)

    log(f"  Prompt: {prompt}")
    log(f"  Response: {response[:200]}")

    alloc = torch.cuda.memory_allocated() / 1e9
    log(f"  VRAM: {alloc:.2f} GB")

    del model
    torch.cuda.empty_cache()
    import gc
    gc.collect()

    log(f"  {domain} adapter: OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train domain-specific QLoRA adapters")
    parser.add_argument("--domain", type=str, choices=list(DOMAIN_LOADERS.keys()) + ["all"],
                        default="all", help="Which domain to train (default: all)")
    parser.add_argument("--verify-only", action="store_true",
                        help="Skip training, just verify existing adapters load")
    args = parser.parse_args()

    domains = list(DOMAIN_LOADERS.keys()) if args.domain == "all" else [args.domain]

    if args.verify_only:
        for d in domains:
            verify_adapter(d)
    else:
        for d in domains:
            train_adapter(d)
            verify_adapter(d)
            prune_adapter_artifacts(os.path.join(OUTPUT_ROOT, d))

    log("\n" + "="*60)
    log("  ALL DONE")
    log("="*60)
