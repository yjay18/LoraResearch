# Day 12: SAE Transfer Assessment

This report compares SAE reconstruction quality for the base model versus each LoRA-adapted model on the same 300 prompts.

## Base Reference

- Avg reconstruction MSE: 640.81
- Avg L0: 331.58
- Avg sparsity: 0.02024
- Avg non-zero entries per prompt: 8214.11

## Domain Summary

| Adapter | Avg MSE | Delta vs Base | Delta % | Avg L0 | Avg Sparsity | Improved Prompts | Worsened Prompts |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| code | 580.20 | -60.60 | -9.46% | 328.92 | 0.02008 | 300 | 0 |
| medical | 615.12 | -25.69 | -4.01% | 330.75 | 0.02019 | 300 | 0 |
| math | 618.51 | -22.30 | -3.48% | 327.19 | 0.01997 | 300 | 0 |
| safety | 609.32 | -31.49 | -4.91% | 326.03 | 0.01990 | 300 | 0 |
| creative | 582.99 | -57.82 | -9.02% | 326.33 | 0.01992 | 300 | 0 |

## Interpretation

- **code**: Transfer looks good: adapted activations reconstruct better than base.
- **medical**: Transfer looks good: adapted activations reconstruct better than base.
- **math**: Transfer looks good: adapted activations reconstruct better than base.
- **safety**: Transfer looks good: adapted activations reconstruct better than base.
- **creative**: Transfer looks good: adapted activations reconstruct better than base.

## Prompt-Domain Effects

### code
- code: avg MSE delta -59.38 (50 improved / 0 worsened)
- creative: avg MSE delta -51.33 (50 improved / 0 worsened)
- general: avg MSE delta -54.17 (50 improved / 0 worsened)
- math: avg MSE delta -21.32 (50 improved / 0 worsened)
- medical: avg MSE delta -74.41 (50 improved / 0 worsened)
- safety: avg MSE delta -103.01 (50 improved / 0 worsened)

### medical
- code: avg MSE delta -25.13 (50 improved / 0 worsened)
- creative: avg MSE delta -21.85 (50 improved / 0 worsened)
- general: avg MSE delta -22.99 (50 improved / 0 worsened)
- math: avg MSE delta -9.20 (50 improved / 0 worsened)
- medical: avg MSE delta -31.45 (50 improved / 0 worsened)
- safety: avg MSE delta -43.54 (50 improved / 0 worsened)

### math
- code: avg MSE delta -21.84 (50 improved / 0 worsened)
- creative: avg MSE delta -18.84 (50 improved / 0 worsened)
- general: avg MSE delta -20.00 (50 improved / 0 worsened)
- math: avg MSE delta -7.80 (50 improved / 0 worsened)
- medical: avg MSE delta -27.35 (50 improved / 0 worsened)
- safety: avg MSE delta -37.95 (50 improved / 0 worsened)

### safety
- code: avg MSE delta -30.74 (50 improved / 0 worsened)
- creative: avg MSE delta -26.64 (50 improved / 0 worsened)
- general: avg MSE delta -28.21 (50 improved / 0 worsened)
- math: avg MSE delta -11.09 (50 improved / 0 worsened)
- medical: avg MSE delta -38.59 (50 improved / 0 worsened)
- safety: avg MSE delta -53.65 (50 improved / 0 worsened)

### creative
- code: avg MSE delta -56.46 (50 improved / 0 worsened)
- creative: avg MSE delta -49.02 (50 improved / 0 worsened)
- general: avg MSE delta -51.64 (50 improved / 0 worsened)
- math: avg MSE delta -20.53 (50 improved / 0 worsened)
- medical: avg MSE delta -70.90 (50 improved / 0 worsened)
- safety: avg MSE delta -98.35 (50 improved / 0 worsened)
