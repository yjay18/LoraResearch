"""
fingerprint.py - Offline feature labeling from sparse SAE activations.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from tokenizers import Tokenizer


DOMAINS = ["general", "code", "medical", "math", "safety", "creative"]

LABEL_TEXT = {
    "bos_boundary": "beginning-of-prompt / BOS gating",
    "code_docstring_example": "Python docstrings, examples, and test-style scaffolding",
    "code_function_definition": "Python function definitions and implementation framing",
    "code_layout_whitespace": "code layout, indentation, and block formatting",
    "math_word_problem": "arithmetic word-problem setup",
    "medical_clinical": "medical / clinical wording",
    "safety_sensitive": "safety-sensitive or refusal-adjacent wording",
    "creative_writing": "creative-writing narrative language",
    "general_legal_reasoning": "legal / exam-style reasoning",
    "generic_instruction": "generic instructional framing",
    "formatting_punctuation": "punctuation / formatting structure",
    "mixed_ambiguous": "mixed or weakly interpretable context",
}

EXPECTED_CODE_FAMILIES = {
    "code_docstring_example",
    "code_function_definition",
    "code_layout_whitespace",
    "formatting_punctuation",
}

NON_CODE_FAMILIES = {
    "medical_clinical",
    "safety_sensitive",
    "creative_writing",
    "math_word_problem",
    "general_legal_reasoning",
    "generic_instruction",
}
STRUCTURAL_FAMILY = "bos_boundary"

FINGERPRINT_VECTOR_COLUMNS = {
    "signed_mass": "delta_mean_prompt_activation",
    "signed_freq": "delta_token_freq",
}

CLASS_DIRECTION = {
    "amplified": 1.0,
    "newly_activated": 1.0,
    "suppressed": -1.0,
    "killed": -1.0,
    "context_shifted": 0.0,
    "unchanged": 0.0,
}

CODE_HINTS = [
    "def ",
    "python function",
    "return",
    "list",
    "string",
    "tuple",
    "dict",
    "grid",
    "input",
    "output",
    "parameter",
    "implement",
    "histogram",
    "factorial",
]
DOCSTRING_HINTS = ['"""', ">>>", "example", "input", "output", "returns", "given", "write a function"]
MATH_HINTS = [
    "how many",
    "each week",
    "each day",
    "minutes",
    "hours",
    "cost",
    "allowance",
    "total",
    "more than",
    "less than",
    "percent",
]
MEDICAL_HINTS = [
    "patient",
    "therapy",
    "clinical",
    "disease",
    "symptom",
    "treatment",
    "diagnosis",
    "hospital",
    "palliative",
    "medical",
]
SAFETY_HINTS = [
    "harm",
    "dangerous",
    "illegal",
    "weapon",
    "kill",
    "bomb",
    "suicide",
    "poison",
    "hack",
    "safe",
    "safety",
    "policy",
]
CREATIVE_HINTS = [
    "story",
    "novel",
    "poem",
    "character",
    "dialogue",
    "scene",
    "chapter",
    "fantasy",
    "love",
    "world where",
    "write a story",
]
LEGAL_HINTS = [
    "court",
    "statute",
    "judgment",
    "appeal",
    "constitution",
    "supreme court",
    "action by",
    "offense",
    "defense",
    "damages",
    "law",
]
GENERIC_HINTS = [
    "write",
    "what is",
    "which of the following",
    "question refers",
    "explain",
    "describe",
]


@dataclass
class EncodedPrompt:
    prompt_id: int
    domain: str
    text: str
    tokens: list[str]
    offsets: list[tuple[int, int]]


def to_native(value):
    if isinstance(value, np.generic):
        return value.item()
    return value


def load_json(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def find_local_tokenizer_json(model_id: str = "google/gemma-2-2b") -> Path:
    repo_dir = "models--" + model_id.replace("/", "--")
    search_roots = []

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        search_roots.append(Path(hf_home) / "hub" / repo_dir)

    search_roots.append(Path.home() / ".cache" / "huggingface" / "hub" / repo_dir)

    for root in search_roots:
        if not root.exists():
            continue
        matches = sorted(root.glob("snapshots/*/tokenizer.json"))
        if matches:
            return matches[-1]

    raise FileNotFoundError(f"Could not find local tokenizer.json for {model_id}")


def load_sparse_feature_archive(path: str | Path) -> dict:
    with np.load(path) as data:
        return {
            "token_indices": data["token_indices"].astype(np.int32, copy=False),
            "feature_indices": data["feature_indices"].astype(np.int32, copy=False),
            "values": data["values"].astype(np.float32, copy=False),
            "seq_len": int(data["seq_len"][0]),
            "d_sae": int(data["d_sae"][0]),
        }


def display_token(raw_token: str) -> str:
    if raw_token == "<bos>":
        return "<bos>"
    if raw_token == "\n":
        return "\\n"
    if raw_token and set(raw_token) == {"▁"}:
        return "indentation" if len(raw_token) > 1 else "space_boundary"
    rendered = raw_token.replace("▁", " ")
    rendered = rendered.replace("\n", "\\n")
    stripped = rendered.strip()
    return stripped or "whitespace"


def build_prompt_cache(
    prompts_path: str | Path,
    tokenizer_path: str | Path | None = None,
) -> tuple[list[dict], dict[int, EncodedPrompt], Path]:
    prompts = load_json(prompts_path)
    tokenizer_path = Path(tokenizer_path) if tokenizer_path else find_local_tokenizer_json()
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    encoded: dict[int, EncodedPrompt] = {}
    for prompt in prompts:
        encoding = tokenizer.encode(prompt["text"])
        encoded[int(prompt["id"])] = EncodedPrompt(
            prompt_id=int(prompt["id"]),
            domain=prompt["domain"],
            text=prompt["text"],
            tokens=list(encoding.tokens),
            offsets=list(encoding.offsets),
        )
    return prompts, encoded, tokenizer_path


def snippet_for_token(prompt: EncodedPrompt, token_index: int, window: int = 32) -> str:
    if token_index >= len(prompt.offsets):
        return prompt.text[: 2 * window].replace("\n", "\\n")
    start, end = prompt.offsets[token_index]
    if start == end == 0:
        return prompt.text[: 2 * window].replace("\n", "\\n")
    left = max(0, start - window)
    right = min(len(prompt.text), end + window)
    return prompt.text[left:right].replace("\n", "\\n")


def dominant_domain(domain_mass: dict[str, float]) -> tuple[str, float]:
    total = sum(domain_mass.values())
    if total <= 0:
        return "none", 0.0
    domain, mass = max(domain_mass.items(), key=lambda item: item[1])
    return domain, float(mass / total)


def keyword_score(text: str, hints: Iterable[str]) -> float:
    lowered = text.lower()
    return float(sum(lowered.count(hint) for hint in hints))


def compute_change_rank(
    classified: pd.DataFrame,
    thresholds: dict,
) -> pd.DataFrame:
    ranked = classified.copy()
    freq_threshold = thresholds["abs_delta_token_freq_threshold"] or 1.0
    mass_threshold = thresholds["abs_delta_mean_prompt_activation_threshold"] or 1.0
    context_threshold = thresholds["context_shift_threshold"] or 1.0
    min_present_prompts = max(int(thresholds["min_present_prompts"]), 1)
    min_context_flip_rate = float(thresholds["min_context_flip_rate"])

    prompt_delta = (ranked["adapted_prompt_count"] - ranked["base_prompt_count"]).abs()
    prompt_component = (prompt_delta / min_present_prompts).clip(upper=5.0)
    context_component = ranked["context_shift"].where(
        ranked["prompt_flip_rate"] >= min_context_flip_rate,
        0.0,
    ) / context_threshold

    ranked["change_rank_score"] = (
        ranked["delta_token_freq"].abs() / freq_threshold
        + ranked["delta_mean_prompt_activation"].abs() / mass_threshold
        + context_component
        + prompt_component
    )
    return ranked


def select_top_changed_features(
    classified: pd.DataFrame,
    thresholds: dict,
    top_k: int = 100,
) -> pd.DataFrame:
    ranked = compute_change_rank(classified, thresholds)
    changed = ranked.loc[ranked["classification"] != "unchanged"].copy()
    changed = changed.sort_values(
        ["change_rank_score", "classification_score", "feature_id"],
        ascending=[False, False, True],
    )
    return changed.head(top_k).reset_index(drop=True)


def collect_feature_evidence(
    adapter: str,
    selected: pd.DataFrame,
    prompts: list[dict],
    encoded_prompts: dict[int, EncodedPrompt],
    feature_root: str | Path,
) -> dict[int, dict]:
    feature_root = Path(feature_root)
    selected_ids = set(int(feature_id) for feature_id in selected["feature_id"].tolist())
    evidence = {}

    for row in selected.to_dict(orient="records"):
        feature_id = int(row["feature_id"])
        evidence[feature_id] = {
            "row": {key: to_native(value) for key, value in row.items()},
            "base": {
                "domain_mass": defaultdict(float),
                "prompt_hits": [],
                "token_stats": defaultdict(
                    lambda: {
                        "display_token": "",
                        "activation_sum": 0.0,
                        "hit_count": 0,
                        "max_activation": 0.0,
                        "example_snippet": "",
                    }
                ),
            },
            "adapted": {
                "domain_mass": defaultdict(float),
                "prompt_hits": [],
                "token_stats": defaultdict(
                    lambda: {
                        "display_token": "",
                        "activation_sum": 0.0,
                        "hit_count": 0,
                        "max_activation": 0.0,
                        "example_snippet": "",
                    }
                ),
            },
        }

    for prompt in prompts:
        prompt_id = int(prompt["id"])
        prompt_domain = prompt["domain"]
        encoded_prompt = encoded_prompts[prompt_id]

        archives = {
            "base": load_sparse_feature_archive(feature_root / "base" / f"prompt_{prompt_id:03d}.npz"),
            "adapted": load_sparse_feature_archive(feature_root / adapter / f"prompt_{prompt_id:03d}.npz"),
        }

        for side, archive in archives.items():
            feature_indices = archive["feature_indices"]
            selected_mask = np.isin(feature_indices, list(selected_ids))
            if not np.any(selected_mask):
                continue

            token_indices = archive["token_indices"][selected_mask]
            filtered_features = feature_indices[selected_mask]
            values = archive["values"][selected_mask]

            for feature_id in np.unique(filtered_features):
                feature_id = int(feature_id)
                mask = filtered_features == feature_id
                feature_token_indices = token_indices[mask]
                feature_values = values[mask]
                total_activation = float(feature_values.sum())
                top_order = np.argsort(feature_values)[::-1][:5]
                token_examples = []

                for order_index in top_order:
                    token_index = int(feature_token_indices[order_index])
                    raw_token = encoded_prompt.tokens[token_index]
                    shown_token = display_token(raw_token)
                    activation = float(feature_values[order_index])
                    snippet = snippet_for_token(encoded_prompt, token_index)
                    token_examples.append(
                        {
                            "token": shown_token,
                            "activation": activation,
                            "snippet": snippet,
                        }
                    )

                    stats = evidence[feature_id][side]["token_stats"][shown_token]
                    stats["display_token"] = shown_token
                    stats["activation_sum"] += activation
                    stats["hit_count"] += 1
                    if activation > stats["max_activation"]:
                        stats["max_activation"] = activation
                        stats["example_snippet"] = snippet

                evidence[feature_id][side]["domain_mass"][prompt_domain] += total_activation
                evidence[feature_id][side]["prompt_hits"].append(
                    {
                        "prompt_id": prompt_id,
                        "domain": prompt_domain,
                        "activation_sum": total_activation,
                        "text": encoded_prompt.text,
                        "top_tokens": token_examples,
                    }
                )

    for feature_id, feature_evidence in evidence.items():
        for side in ["base", "adapted"]:
            prompt_hits = feature_evidence[side]["prompt_hits"]
            prompt_hits.sort(key=lambda row: row["activation_sum"], reverse=True)
            feature_evidence[side]["prompt_hits"] = prompt_hits[:10]

            token_rows = list(feature_evidence[side]["token_stats"].values())
            token_rows.sort(key=lambda row: (row["activation_sum"], row["max_activation"]), reverse=True)
            feature_evidence[side]["top_tokens"] = [
                {
                    "token": row["display_token"],
                    "activation_sum": round(float(row["activation_sum"]), 6),
                    "hit_count": int(row["hit_count"]),
                    "max_activation": round(float(row["max_activation"]), 6),
                    "example_snippet": row["example_snippet"],
                }
                for row in token_rows[:12]
            ]
            feature_evidence[side]["domain_mass"] = {
                domain: round(float(feature_evidence[side]["domain_mass"].get(domain, 0.0)), 6)
                for domain in DOMAINS
                if feature_evidence[side]["domain_mass"].get(domain, 0.0) > 0
            }
            del feature_evidence[side]["token_stats"]

    return evidence


def choose_label_family(
    classification: str,
    primary_domain: str,
    primary_domain_share: float,
    top_tokens: list[dict],
    top_prompts: list[dict],
) -> tuple[str, float, dict[str, float]]:
    top_token_mass = sum(float(row["activation_sum"]) for row in top_tokens[:8]) or 1.0
    token_masses = {row["token"]: float(row["activation_sum"]) for row in top_tokens}
    sorted_token_masses = sorted(token_masses.values(), reverse=True)
    text_blob = " ".join(prompt["text"] for prompt in top_prompts[:6]).lower()
    token_blob = " ".join(row["token"].lower() for row in top_tokens[:8])

    bos_share = token_masses.get("<bos>", 0.0) / top_token_mass
    second_mass = sorted_token_masses[1] if len(sorted_token_masses) > 1 else 0.0
    indentation_mass = sum(
        token_masses.get(token, 0.0) for token in ["indentation", "\\n", "whitespace", "space_boundary"]
    ) / top_token_mass

    code_score = keyword_score(text_blob, CODE_HINTS)
    docstring_score = keyword_score(text_blob, DOCSTRING_HINTS)
    math_score = keyword_score(text_blob, MATH_HINTS)
    medical_score = keyword_score(text_blob, MEDICAL_HINTS)
    safety_score = keyword_score(text_blob, SAFETY_HINTS)
    creative_score = keyword_score(text_blob, CREATIVE_HINTS)
    legal_score = keyword_score(text_blob, LEGAL_HINTS)
    generic_score = keyword_score(text_blob, GENERIC_HINTS)

    if any(token in token_blob for token in ['"""', "def", "return", "import", "->", "str"]):
        code_score += 1.5
        docstring_score += 2.0
    if any(token in token_blob for token in ["\\n", "indentation", ":", "(", ")", "[", "]", "{", "}"]):
        code_score += 1.0
    if any(token in token_blob for token in ["court", "statute", "action", "judgment"]):
        legal_score += 2.5
    if any(token in token_blob for token in ["story", "novel", "character", "dialogue"]):
        creative_score += 1.5
    if any(token in token_blob for token in ["patient", "therapy", "clinical", "medical"]):
        medical_score += 1.5
    if any(token in token_blob for token in ["weapon", "dangerous", "harm", "safe"]):
        safety_score += 1.5

    if bos_share >= 0.9 and token_masses.get("<bos>", 0.0) >= max(second_mass * 5.0, 1e-6):
        heuristic_scores = {"bos_boundary": round(5.0 + bos_share * 3.0, 4)}
        confidence = min(0.95, 0.55 + bos_share * 0.3)
        return "bos_boundary", round(confidence, 3), heuristic_scores

    scores = defaultdict(float)
    if indentation_mass >= 0.35:
        scores["code_layout_whitespace"] += 2.0 + indentation_mass * 2.0
        scores["formatting_punctuation"] += 1.5
    if primary_domain == "code" and primary_domain_share >= 0.45:
        scores["code_function_definition"] += 1.0 + primary_domain_share
    if primary_domain == "math" and primary_domain_share >= 0.45:
        scores["math_word_problem"] += 0.8 + primary_domain_share
    if primary_domain == "medical" and primary_domain_share >= 0.45:
        scores["medical_clinical"] += 1.0 + primary_domain_share
    if primary_domain == "safety" and primary_domain_share >= 0.45:
        scores["safety_sensitive"] += 1.0 + primary_domain_share
    if primary_domain == "creative" and primary_domain_share >= 0.45:
        scores["creative_writing"] += 1.0 + primary_domain_share
    if primary_domain == "general" and primary_domain_share >= 0.45:
        scores["generic_instruction"] += 0.8 + primary_domain_share

    if docstring_score >= 2.0:
        scores["code_docstring_example"] += docstring_score * 1.2
    if code_score >= 2.5:
        scores["code_function_definition"] += code_score
    if math_score >= 1.5:
        scores["math_word_problem"] += math_score
    if medical_score >= 1.0:
        scores["medical_clinical"] += medical_score * 1.2
    if safety_score >= 1.0:
        scores["safety_sensitive"] += safety_score * 1.2
    if creative_score >= 1.0:
        scores["creative_writing"] += creative_score * 1.2
    if legal_score >= 1.0:
        scores["general_legal_reasoning"] += legal_score * 1.4
    if generic_score >= 1.0:
        scores["generic_instruction"] += generic_score * 0.8

    if not scores:
        if classification in {"newly_activated", "killed"} and bos_share >= 0.35:
            scores["bos_boundary"] += 2.5
        elif indentation_mass >= 0.25:
            scores["formatting_punctuation"] += 1.5
        elif primary_domain == "general":
            scores["generic_instruction"] += 1.5
        else:
            scores["mixed_ambiguous"] += 1.0

    best_family = "mixed_ambiguous"
    best_score = 0.0
    for family, score in scores.items():
        if score > best_score:
            best_family = family
            best_score = score

    confidence = min(0.95, 0.35 + primary_domain_share * 0.3 + min(best_score / 8.0, 0.3))
    return best_family, round(confidence, 3), {key: round(float(value), 4) for key, value in scores.items()}


def build_feature_label_record(adapter: str, feature_evidence: dict) -> dict:
    row = feature_evidence["row"]
    classification = row["classification"]
    if classification in {"suppressed", "killed"}:
        primary_side = "base"
        secondary_side = "adapted"
    else:
        primary_side = "adapted"
        secondary_side = "base"

    primary_domain, primary_domain_share = dominant_domain(feature_evidence[primary_side]["domain_mass"])
    secondary_domain, secondary_domain_share = dominant_domain(feature_evidence[secondary_side]["domain_mass"])

    family, confidence, heuristic_scores = choose_label_family(
        classification=classification,
        primary_domain=primary_domain,
        primary_domain_share=primary_domain_share,
        top_tokens=feature_evidence[primary_side]["top_tokens"],
        top_prompts=feature_evidence[primary_side]["prompt_hits"],
    )

    prompt_ids = [int(hit["prompt_id"]) for hit in feature_evidence[primary_side]["prompt_hits"][:3]]
    prompt_domains = [hit["domain"] for hit in feature_evidence[primary_side]["prompt_hits"][:3]]
    top_tokens = [token["token"] for token in feature_evidence[primary_side]["top_tokens"][:5]]
    rationale = (
        f"Primary evidence comes from the {primary_side} side; activation mass is dominated by "
        f"{primary_domain} prompts ({primary_domain_share:.1%}). Top tokens: "
        f"{', '.join(top_tokens) if top_tokens else 'n/a'}. "
        f"Representative prompt ids: {prompt_ids if prompt_ids else 'n/a'}."
    )

    return {
        "adapter": adapter,
        "feature_id": int(row["feature_id"]),
        "classification": classification,
        "classification_score": round(float(row["classification_score"]), 6),
        "change_rank_score": round(float(row["change_rank_score"]), 6),
        "label_family": family,
        "label": LABEL_TEXT[family],
        "confidence": confidence,
        "primary_side": primary_side,
        "primary_domain": primary_domain,
        "primary_domain_share": round(primary_domain_share, 6),
        "secondary_domain": secondary_domain,
        "secondary_domain_share": round(secondary_domain_share, 6),
        "base_prompt_count": int(row["base_prompt_count"]),
        "adapted_prompt_count": int(row["adapted_prompt_count"]),
        "delta_token_freq": round(float(row["delta_token_freq"]), 9),
        "delta_mean_prompt_activation": round(float(row["delta_mean_prompt_activation"]), 6),
        "context_shift": round(float(row["context_shift"]), 6),
        "top_tokens": feature_evidence[primary_side]["top_tokens"][:8],
        "top_prompts": [
            {
                "prompt_id": int(hit["prompt_id"]),
                "domain": hit["domain"],
                "activation_sum": round(float(hit["activation_sum"]), 6),
                "text": hit["text"],
                "top_tokens": [
                    {
                        "token": example["token"],
                        "activation": round(float(example["activation"]), 6),
                        "snippet": example["snippet"],
                    }
                    for example in hit["top_tokens"][:3]
                ],
            }
            for hit in feature_evidence[primary_side]["prompt_hits"][:5]
        ],
        "base_top_tokens": feature_evidence["base"]["top_tokens"][:8],
        "adapted_top_tokens": feature_evidence["adapted"]["top_tokens"][:8],
        "base_domain_mass": feature_evidence["base"]["domain_mass"],
        "adapted_domain_mass": feature_evidence["adapted"]["domain_mass"],
        "heuristic_scores": heuristic_scores,
        "rationale": rationale,
        "matches_code_expectation": family in EXPECTED_CODE_FAMILIES,
        "is_non_code_semantic": family in NON_CODE_FAMILIES,
    }


def label_selected_features(
    adapter: str,
    selected: pd.DataFrame,
    prompts_path: str | Path,
    feature_root: str | Path,
    tokenizer_path: str | Path | None = None,
) -> tuple[list[dict], Path]:
    prompts, encoded_prompts, resolved_tokenizer_path = build_prompt_cache(prompts_path, tokenizer_path)
    evidence = collect_feature_evidence(adapter, selected, prompts, encoded_prompts, feature_root)
    labels = [
        build_feature_label_record(adapter, evidence[int(feature_id)])
        for feature_id in selected["feature_id"].tolist()
    ]
    return labels, resolved_tokenizer_path


def labels_to_frame(labels: list[dict]) -> pd.DataFrame:
    rows = []
    for label in labels:
        rows.append(
            {
                "adapter": label["adapter"],
                "feature_id": label["feature_id"],
                "classification": label["classification"],
                "classification_score": label["classification_score"],
                "change_rank_score": label["change_rank_score"],
                "label_family": label["label_family"],
                "label": label["label"],
                "confidence": label["confidence"],
                "primary_side": label["primary_side"],
                "primary_domain": label["primary_domain"],
                "primary_domain_share": label["primary_domain_share"],
                "secondary_domain": label["secondary_domain"],
                "secondary_domain_share": label["secondary_domain_share"],
                "base_prompt_count": label["base_prompt_count"],
                "adapted_prompt_count": label["adapted_prompt_count"],
                "delta_token_freq": label["delta_token_freq"],
                "delta_mean_prompt_activation": label["delta_mean_prompt_activation"],
                "context_shift": label["context_shift"],
                "top_tokens": " | ".join(token["token"] for token in label["top_tokens"][:5]),
                "top_prompt_ids": " | ".join(str(prompt["prompt_id"]) for prompt in label["top_prompts"][:5]),
                "top_prompt_domains": " | ".join(prompt["domain"] for prompt in label["top_prompts"][:5]),
                "rationale": label["rationale"],
                "matches_code_expectation": label["matches_code_expectation"],
            }
        )
    return pd.DataFrame(rows)


def summarize_label_families(labels: list[dict]) -> dict[str, int]:
    counts = Counter(label["label_family"] for label in labels)
    return {label: counts[label] for label in sorted(counts)}


def load_classification_artifacts(
    adapter: str,
    classification_root: str | Path = "results/classification",
) -> tuple[pd.DataFrame, dict]:
    classification_root = Path(classification_root)
    classified = pd.read_csv(classification_root / f"{adapter}_classified_features.csv")
    summary = load_json(classification_root / f"{adapter}_classification_summary.json")
    return classified, summary


def prepare_ranked_classified_frame(
    adapter: str,
    classification_root: str | Path = "results/classification",
) -> tuple[pd.DataFrame, dict]:
    classified, summary = load_classification_artifacts(adapter, classification_root)
    ranked = compute_change_rank(classified, summary["thresholds"])
    ranked = ranked.sort_values("feature_id").reset_index(drop=True)
    return ranked, summary


def label_top_changed_for_adapter(
    adapter: str,
    top_k: int = 250,
    prompts_path: str | Path = "data/prompts_300.json",
    feature_root: str | Path = "data/sae_features",
    tokenizer_path: str | Path | None = None,
    classification_root: str | Path = "results/classification",
) -> dict:
    ranked, summary = prepare_ranked_classified_frame(adapter, classification_root)
    selected = select_top_changed_features(ranked, summary["thresholds"], top_k=top_k)
    labels, resolved_tokenizer_path = label_selected_features(
        adapter=adapter,
        selected=selected,
        prompts_path=prompts_path,
        feature_root=feature_root,
        tokenizer_path=tokenizer_path,
    )
    return {
        "adapter": adapter,
        "top_k": top_k,
        "tokenizer_path": str(resolved_tokenizer_path),
        "thresholds": summary["thresholds"],
        "label_count": len(labels),
        "labels": labels,
    }


def build_dense_fingerprint_vectors(
    ranked: pd.DataFrame,
    zero_unchanged: bool = True,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    ordered = ranked.sort_values("feature_id").reset_index(drop=True)
    changed_mask = (ordered["classification"] != "unchanged").to_numpy(dtype=bool, copy=False)
    feature_ids = ordered["feature_id"].to_numpy(dtype=np.int32, copy=False)

    vectors = {}
    for name, column in FINGERPRINT_VECTOR_COLUMNS.items():
        values = ordered[column].to_numpy(dtype=np.float32, copy=True)
        if zero_unchanged:
            values[~changed_mask] = 0.0
        vectors[name] = values

    direction = ordered["classification"].map(CLASS_DIRECTION).fillna(0.0).to_numpy(dtype=np.float32, copy=False)
    fallback_sign = np.sign(ordered["delta_mean_prompt_activation"].to_numpy(dtype=np.float32, copy=False))
    signed_rank = ordered["change_rank_score"].to_numpy(dtype=np.float32, copy=True)
    signed_rank *= np.where(direction != 0.0, direction, fallback_sign)
    if zero_unchanged:
        signed_rank[~changed_mask] = 0.0
    vectors["signed_rank"] = signed_rank

    return feature_ids, vectors
