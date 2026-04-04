# Day 23: Fingerprint Embedding

Projected the full and structural-filtered fingerprint vectors into 2D with PCA.

## Why PCA

- There are only 6 points in this view (`base` + 5 adapters), so PCA is more stable and more honest than a decorative UMAP/t-SNE layout.

## Base Distances

- `full`: creative=204.852, code=180.675, safety=153.664, medical=113.065, math=110.191
- `filtered`: creative=175.394, code=150.905, safety=109.848, medical=104.229, math=92.669

## Interpretation

- In the full-vector PCA, PC1 explains 68.4% of the variance and is dominated by the largest shared change axis.
- In the structural-filtered PCA, PC1 still explains 61.4%, and the layout continues to separate the math/safety side from the code/creative side.
- The largest raw fingerprint shifts come from `creative` and `code`, and they remain the largest even after filtering (`creative`, `code`).
- The base point sits at the zero vector by construction, so distance from base should be read as total feature-shift magnitude rather than behavioral distance.
