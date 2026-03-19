# LoRASurgeon

Analysing what LoRA fine-tuning adapters actually learn, using sparse autoencoders (SAEs) to decompose the changes into interpretable features.

## Overview

LoRASurgeon uses pre-trained SAEs (Gemma Scope) to compare feature activations between a base model and its LoRA-adapted variants. By examining which SAE features are amplified, suppressed, or shifted across different domain-specific LoRA adapters, we can build interpretable "fingerprints" of what each adapter has learned.

## Setup

```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126

# Install project
pip install -e .
```

## Project Structure

```
lorasurgeon/
├── lorasurgeon/        # Core library
│   ├── collect.py      # Activation collection from models
│   ├── sae_decode.py   # SAE encoding of activations
│   ├── diff.py         # Differential analysis
│   ├── classify.py     # Feature classification
│   ├── fingerprint.py  # Cross-domain fingerprints
│   ├── steer.py        # Activation steering
│   ├── viz.py          # Visualisation
│   └── utils.py        # Helpers
├── prompts/            # Domain-specific prompt datasets
├── results/            # Experiment outputs
├── notebooks/          # Analysis notebooks
└── tests/              # Tests
```
