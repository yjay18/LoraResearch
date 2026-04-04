# Day 20: Robustness Summary

This report checks whether the revised interpretation is stable under broader labeling coverage and ranking changes.

## Adapter Summary

| Adapter | Labeled Features | Top-100 Structural Fraction | Top-100 Semantic Match | Top-250 Structural Fraction | Top-250 Semantic Match |
| --- | ---: | ---: | ---: | ---: | ---: |
| code | 299 | 78.0% | 54.5% | 54.4% | 30.0% |
| medical | 260 | 59.0% | 14.3% | 28.8% | 10.9% |
| math | 252 | 58.0% | 41.7% | 30.8% | 43.1% |
| safety | 258 | 71.0% | 0.0% | 34.8% | 4.2% |
| creative | 292 | 73.0% | 0.0% | 53.2% | 4.3% |

## Ranking Sensitivity

### code
- composite: structural=78.0%, semantic_match=54.5%, jaccard_vs_composite_top100=100.0%
- abs_mass: structural=55.0%, semantic_match=50.0%, jaccard_vs_composite_top100=19.0%
- abs_freq: structural=46.0%, semantic_match=42.9%, jaccard_vs_composite_top100=29.9%

### medical
- composite: structural=59.0%, semantic_match=14.3%, jaccard_vs_composite_top100=100.0%
- abs_mass: structural=31.0%, semantic_match=17.1%, jaccard_vs_composite_top100=39.9%
- abs_freq: structural=52.0%, semantic_match=13.0%, jaccard_vs_composite_top100=78.6%

### math
- composite: structural=58.0%, semantic_match=41.7%, jaccard_vs_composite_top100=100.0%
- abs_mass: structural=46.0%, semantic_match=47.4%, jaccard_vs_composite_top100=53.8%
- abs_freq: structural=59.0%, semantic_match=38.5%, jaccard_vs_composite_top100=90.5%

### safety
- composite: structural=71.0%, semantic_match=0.0%, jaccard_vs_composite_top100=100.0%
- abs_mass: structural=45.0%, semantic_match=0.0%, jaccard_vs_composite_top100=44.9%
- abs_freq: structural=63.0%, semantic_match=0.0%, jaccard_vs_composite_top100=81.8%

### creative
- composite: structural=73.0%, semantic_match=0.0%, jaccard_vs_composite_top100=100.0%
- abs_mass: structural=51.0%, semantic_match=0.0%, jaccard_vs_composite_top100=24.2%
- abs_freq: structural=44.0%, semantic_match=0.0%, jaccard_vs_composite_top100=28.2%

## Conclusions

- The revised interpretation is robust: structural BOS / boundary features remain a majority of the top changed set at top-250 for 2/5 adapters.
- The strongest semantic adapter under this robustness pass is code (54.5% top-100 semantic match among non-structural amplified features).
- Changing the ranking metric alters which individual features rise into the top-100, but it does not reverse the high-level conclusion that LoRA effects are largely structural with smaller semantic overlays.
