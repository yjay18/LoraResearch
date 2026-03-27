"""
collect.py — Activation collection from base and adapted models.
Hooks into residual stream at target layers and stores activations.
"""

import torch
from contextlib import contextmanager
from typing import Optional


class ResidualStreamCollector:
    """Collects residual stream activations from specified decoder layers.

    Works with both raw HuggingFace models and PEFT-wrapped models.
    Activations are detached, moved to CPU, and cast to float32 to
    avoid accumulating GPU memory during collection.

    Usage:
        collector = ResidualStreamCollector(model, layers=[12])
        with collector:
            model(**inputs)
        activations = collector.activations[12]  # shape: [batch, seq_len, d_model]
    """

    def __init__(self, model, layers: Optional[list[int]] = None):
        self.model = model
        self.activations: dict[int, torch.Tensor] = {}
        self._hooks = []
        self._decoder_layers = self._find_decoder_layers(model)
        if layers is None:
            layers = list(range(len(self._decoder_layers)))
        self.layers = layers

    @staticmethod
    def _find_decoder_layers(model):
        """Navigate through PEFT/HF wrappers to find the decoder layers."""
        candidates = [
            lambda m: m.model.layers,                      # raw HF
            lambda m: m.base_model.model.model.layers,     # PEFT-wrapped
            lambda m: m.model.model.layers,                # alternate
        ]
        for fn in candidates:
            try:
                layers = fn(model)
                if hasattr(layers, '__len__') and len(layers) > 0:
                    return layers
            except (AttributeError, TypeError):
                continue
        raise RuntimeError("Could not find decoder layers in model")

    @property
    def num_layers(self) -> int:
        return len(self._decoder_layers)

    def _make_hook(self, layer_idx: int):
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            self.activations[layer_idx] = hidden.detach().cpu().float()
        return hook_fn

    def __enter__(self):
        self.activations = {}
        for layer_idx in self.layers:
            layer = self._decoder_layers[layer_idx]
            h = layer.register_forward_hook(self._make_hook(layer_idx))
            self._hooks.append(h)
        return self

    def __exit__(self, *args):
        for h in self._hooks:
            h.remove()
        self._hooks = []


def collect_activations(
    model,
    tokenizer,
    prompts: list[str],
    layers: list[int],
    batch_size: int = 1,
) -> dict[int, torch.Tensor]:
    """Collect residual stream activations for a list of prompts.

    Args:
        model: HF or PEFT-wrapped causal LM.
        tokenizer: Corresponding tokenizer.
        prompts: List of text prompts.
        layers: Which decoder layers to hook.
        batch_size: How many prompts to process at once.

    Returns:
        Dict mapping layer_idx -> tensor of shape [num_prompts, max_seq_len, d_model].
        Shorter sequences are right-padded with zeros.
    """
    all_activations = {l: [] for l in layers}

    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with ResidualStreamCollector(model, layers=layers) as collector:
            with torch.no_grad():
                model(**inputs)
            for layer_idx in layers:
                all_activations[layer_idx].append(collector.activations[layer_idx])

        del inputs
        torch.cuda.empty_cache()

    # Concatenate batches along the batch dimension
    result = {}
    for layer_idx in layers:
        result[layer_idx] = torch.cat(all_activations[layer_idx], dim=0)
    return result
