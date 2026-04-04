# Day 22: Fingerprint Similarity Analysis

Pairwise adapter similarity was computed on dense signed mass vectors, plus a conservative structural-filtered view.

## Key Pairs

- Strongest full-vector cosine pair: `math` + `safety` = `0.733`
- Weakest full-vector cosine pair: `math` + `creative` = `-0.425`
- Strongest filtered cosine pair: `math` + `safety` = `0.691`
- Weakest filtered cosine pair: `math` + `creative` = `-0.364`

## Interpretation

- `math` and `safety` remain the strongest pair in both the full and filtered views (0.733 -> 0.691 cosine), so that relationship is not just a structural-BOS artifact.
- `code` and `creative` are the second-closest pair in both views (0.650 -> 0.569 cosine), which suggests a second shared change axis.
- `medical` is the only adapter that stays positively aligned with every other adapter after filtering, which makes it the bridge case rather than a clean cluster endpoint.
- The structural filter currently removes 252 features. That is conservative rather than exhaustive, so the filtered matrix should be read as a lower-bound attempt to isolate semantics.
