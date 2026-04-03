"""
Day 8: Build prompt dataset from existing benchmarks.
50 prompts per domain (code, medical, math, safety, creative) + 50 general = 300 total.
"""
import json
import os
import random
import sys

random.seed(42)

def log(msg):
    print(msg, flush=True)


def source_code_prompts(n=50):
    """Source from MBPP (Mostly Basic Python Problems) and HumanEval."""
    from datasets import load_dataset
    prompts = []

    # MBPP — real coding problems
    log("  Loading MBPP...")
    ds = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
    for ex in ds:
        prompts.append(ex["prompt"].strip())
    random.shuffle(prompts)
    prompts = prompts[:40]

    # HumanEval — function generation tasks
    log("  Loading HumanEval...")
    ds = load_dataset("openai/openai_humaneval", split="test")
    he_prompts = []
    for ex in ds:
        # Extract the docstring as the prompt
        text = ex["prompt"].strip()
        he_prompts.append(text)
    random.shuffle(he_prompts)
    prompts.extend(he_prompts[:n - len(prompts)])

    return [{"text": p, "domain": "code", "source": "MBPP/HumanEval"} for p in prompts[:n]]


def source_medical_prompts(n=50):
    """Source from MedQA and PubMedQA."""
    from datasets import load_dataset
    prompts = []

    # MedQA — USMLE-style questions
    log("  Loading MedQA...")
    try:
        ds = load_dataset("bigbio/med_qa", "med_qa_en_source", split="test", trust_remote_code=True)
        for ex in ds:
            q = ex.get("question", "").strip()
            if q and len(q) > 30:
                prompts.append(q)
    except Exception as e:
        log(f"    MedQA failed: {e}, trying fallback...")
        # Fallback: PubMedQA
        ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
        for ex in ds:
            q = ex.get("question", "").strip()
            if q and len(q) > 20:
                prompts.append(q)

    random.shuffle(prompts)
    if len(prompts) < n:
        # Supplement with MedMCQA
        log("  Loading MedMCQA supplement...")
        ds = load_dataset("openlifescienceai/medmcqa", split="validation")
        extra = []
        for ex in ds:
            q = ex.get("question", "").strip()
            if q and len(q) > 20:
                extra.append(q)
        random.shuffle(extra)
        prompts.extend(extra[:n - len(prompts)])

    return [{"text": p, "domain": "medical", "source": "MedQA/PubMedQA/MedMCQA"} for p in prompts[:n]]


def source_math_prompts(n=50):
    """Source from GSM8K and MATH."""
    from datasets import load_dataset
    prompts = []

    # GSM8K — grade school math
    log("  Loading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="test")
    for ex in ds:
        q = ex.get("question", "").strip()
        if q:
            prompts.append(q)
    random.shuffle(prompts)
    prompts = prompts[:35]

    # MATH — competition math
    log("  Loading MATH...")
    try:
        ds = load_dataset("lighteval/MATH", "all", split="test", trust_remote_code=True)
        math_prompts = []
        for ex in ds:
            q = ex.get("problem", "").strip()
            if q and len(q) > 20:
                math_prompts.append(q)
        random.shuffle(math_prompts)
        prompts.extend(math_prompts[:n - len(prompts)])
    except Exception as e:
        log(f"    MATH failed: {e}, using more GSM8K...")
        ds = load_dataset("openai/gsm8k", "main", split="train")
        extra = [ex["question"].strip() for ex in ds if ex.get("question")]
        random.shuffle(extra)
        prompts.extend(extra[:n - len(prompts)])

    return [{"text": p, "domain": "math", "source": "GSM8K/MATH"} for p in prompts[:n]]


def source_safety_prompts(n=50):
    """Source from TruthfulQA and Anthropic HH-RLHF (harmful prompts that need safe responses)."""
    from datasets import load_dataset
    prompts = []

    # TruthfulQA — questions designed to test truthfulness
    log("  Loading TruthfulQA...")
    ds = load_dataset("truthfulqa/truthful_qa", "generation", split="validation")
    for ex in ds:
        q = ex.get("question", "").strip()
        if q:
            prompts.append(q)
    random.shuffle(prompts)
    prompts = prompts[:30]

    # HH-RLHF — extract the human turn from harmless conversations
    log("  Loading HH-RLHF (harmless)...")
    ds = load_dataset("Anthropic/hh-rlhf", data_dir="harmless-base", split="test")
    hh_prompts = []
    for ex in ds:
        chosen = ex.get("chosen", "")
        # Extract the first human turn
        if "\n\nHuman:" in chosen:
            human_turn = chosen.split("\n\nHuman:")[1].split("\n\nAssistant:")[0].strip()
            if human_turn and len(human_turn) > 15:
                hh_prompts.append(human_turn)
    random.shuffle(hh_prompts)
    prompts.extend(hh_prompts[:n - len(prompts)])

    return [{"text": p, "domain": "safety", "source": "TruthfulQA/HH-RLHF"} for p in prompts[:n]]


def source_creative_prompts(n=50):
    """Source from WritingPrompts."""
    from datasets import load_dataset
    prompts = []

    log("  Loading WritingPrompts...")
    ds = load_dataset("euclaise/writingprompts", split="test")
    for ex in ds:
        title = ex.get("prompt", "").strip()
        # Clean up reddit-style prefixes
        for prefix in ["[WP]", "[EU]", "[CW]", "[TT]", "[MP]", "[PI]", "[OT]", "[ WP ]", "[ EU ]", "[ CW ]"]:
            title = title.replace(prefix, "").strip()
        if title and len(title) > 20:
            prompts.append(title)
    random.shuffle(prompts)

    return [{"text": p, "domain": "creative", "source": "WritingPrompts"} for p in prompts[:n]]


def source_general_prompts(n=50):
    """Source from MMLU and ARC for general knowledge."""
    from datasets import load_dataset
    prompts = []

    # ARC (AI2 Reasoning Challenge) — science questions
    log("  Loading ARC-Challenge...")
    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
    for ex in ds:
        q = ex.get("question", "").strip()
        if q and len(q) > 15:
            prompts.append(q)
    random.shuffle(prompts)
    prompts = prompts[:25]

    # MMLU — multitask academic questions
    log("  Loading MMLU...")
    try:
        ds = load_dataset("cais/mmlu", "all", split="test")
        mmlu_prompts = []
        for ex in ds:
            q = ex.get("question", "").strip()
            if q and len(q) > 15:
                mmlu_prompts.append(q)
        random.shuffle(mmlu_prompts)
        prompts.extend(mmlu_prompts[:n - len(prompts)])
    except Exception as e:
        log(f"    MMLU failed: {e}, using more ARC...")
        ds = load_dataset("allenai/ai2_arc", "ARC-Easy", split="test")
        extra = [ex["question"].strip() for ex in ds if ex.get("question")]
        random.shuffle(extra)
        prompts.extend(extra[:n - len(prompts)])

    return [{"text": p, "domain": "general", "source": "ARC/MMLU"} for p in prompts[:n]]


def verify_prompts(prompts):
    """Quality checks on the prompt dataset."""
    issues = []

    for i, p in enumerate(prompts):
        text = p["text"]
        # Check minimum length
        if len(text) < 10:
            issues.append(f"  [{i}] Too short ({len(text)} chars): '{text[:50]}'")
        # Check for empty/whitespace
        if not text.strip():
            issues.append(f"  [{i}] Empty prompt")
        # Check for excessive length (truncate at 1000 chars for our purposes)
        if len(text) > 1000:
            p["text"] = text[:1000]
            issues.append(f"  [{i}] Truncated from {len(text)} to 1000 chars")

    # Check for duplicates
    texts = [p["text"] for p in prompts]
    dupes = len(texts) - len(set(texts))
    if dupes > 0:
        issues.append(f"  Found {dupes} duplicate prompts — removing")
        seen = set()
        deduped = []
        for p in prompts:
            if p["text"] not in seen:
                seen.add(p["text"])
                deduped.append(p)
        prompts = deduped

    # Domain balance check
    from collections import Counter
    domain_counts = Counter(p["domain"] for p in prompts)
    for domain, count in sorted(domain_counts.items()):
        status = "OK" if count == 50 else f"WARN: expected 50"
        issues.append(f"  {domain}: {count} prompts ({status})")

    return prompts, issues


if __name__ == "__main__":
    log("=" * 60)
    log("  DAY 8: Building prompt dataset from benchmarks")
    log("=" * 60)

    all_prompts = []

    log("\n[1/6] Code prompts (MBPP + HumanEval)")
    all_prompts.extend(source_code_prompts(50))
    log(f"  -> {sum(1 for p in all_prompts if p['domain'] == 'code')} code prompts")

    log("\n[2/6] Medical prompts (MedQA + PubMedQA + MedMCQA)")
    all_prompts.extend(source_medical_prompts(50))
    log(f"  -> {sum(1 for p in all_prompts if p['domain'] == 'medical')} medical prompts")

    log("\n[3/6] Math prompts (GSM8K + MATH)")
    all_prompts.extend(source_math_prompts(50))
    log(f"  -> {sum(1 for p in all_prompts if p['domain'] == 'math')} math prompts")

    log("\n[4/6] Safety prompts (TruthfulQA + HH-RLHF)")
    all_prompts.extend(source_safety_prompts(50))
    log(f"  -> {sum(1 for p in all_prompts if p['domain'] == 'safety')} safety prompts")

    log("\n[5/6] Creative prompts (WritingPrompts)")
    all_prompts.extend(source_creative_prompts(50))
    log(f"  -> {sum(1 for p in all_prompts if p['domain'] == 'creative')} creative prompts")

    log("\n[6/6] General prompts (ARC + MMLU)")
    all_prompts.extend(source_general_prompts(50))
    log(f"  -> {sum(1 for p in all_prompts if p['domain'] == 'general')} general prompts")

    # Verify
    log(f"\n{'='*60}")
    log("  VERIFICATION")
    log(f"{'='*60}")
    all_prompts, issues = verify_prompts(all_prompts)
    for issue in issues:
        log(issue)

    # Assign IDs
    for i, p in enumerate(all_prompts):
        p["id"] = i

    # Save
    os.makedirs("data", exist_ok=True)
    out_path = "data/prompts_300.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_prompts, f, indent=2, ensure_ascii=False)

    log(f"\n{'='*60}")
    log(f"  SAVED: {len(all_prompts)} prompts -> {out_path}")
    log(f"{'='*60}")

    # Summary stats
    from collections import Counter
    domain_counts = Counter(p["domain"] for p in all_prompts)
    avg_len = sum(len(p["text"]) for p in all_prompts) / len(all_prompts)
    log(f"  Total: {len(all_prompts)}")
    log(f"  Avg prompt length: {avg_len:.0f} chars")
    for domain, count in sorted(domain_counts.items()):
        domain_avg = sum(len(p["text"]) for p in all_prompts if p["domain"] == domain) / count
        log(f"    {domain}: {count} prompts (avg {domain_avg:.0f} chars)")
    log("  Day 8: COMPLETE")
