# Day 21: Adapter Fingerprint Vectors

Built dense per-feature fingerprint vectors for each adapter from the saved Day 14 classifications.

| Adapter | Changed Features | Top-250 Structural | Top-250 Non-Structural Semantic Match | L2 Norm |
| --- | ---: | ---: | ---: | ---: |
| code | 2406 | 136 | 30.0% | 180.675 |
| medical | 2480 | 72 | 10.9% | 113.065 |
| math | 2494 | 77 | 43.1% | 110.191 |
| safety | 2488 | 87 | 4.2% | 153.664 |
| creative | 2377 | 133 | 4.3% | 204.852 |

## Notes

- Each fingerprint is a dense 16,384-dimensional vector aligned to the shared SAE feature basis.
- The primary signed vector uses delta mean prompt activation with unchanged features zeroed out.
- Top-250 labels are saved alongside each adapter summary to make the structural filter auditable.
