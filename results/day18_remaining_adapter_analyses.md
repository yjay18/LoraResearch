# Day 18: Remaining Adapter Analyses

This report bundles the `math`, `safety`, and `creative` adapter deep dives and the first universal-feature pass.

## Adapter Snapshot

| Adapter | Structural Gates | Raw Amplified Match | Non-Structural Amplified Match |
| --- | ---: | ---: | ---: |
| math | 58 | 20.8% | 41.7% |
| safety | 71 | 0.0% | 0.0% |
| creative | 73 | 0.0% | 0.0% |

## Per-Adapter Notes

### Math
- The math adapter is structurally dominated: 58 of the top 100 changed features are BOS / boundary gates.
- There is only partial semantic alignment with math / arithmetic semantics once structural gates are removed (41.7% semantic match).
- The strongest non-structural families are general_legal_reasoning (12), math_word_problem (12), code_function_definition (8).
- The math adapter does surface some arithmetic-word-problem features, but they do not dominate the top-changed set.

### Safety
- The safety adapter is structurally dominated: 71 of the top 100 changed features are BOS / boundary gates.
- The amplified semantic subset barely matches the expected safety / refusal semantics (0.0% semantic match), so the adapter is not cleanly separable by its top changed features.
- The strongest non-structural families are general_legal_reasoning (10), code_docstring_example (5), math_word_problem (5).
- The safety adapter is the weakest semantic match of the group: refusal / safety-specific labels are rare, which suggests structural routing is overshadowing policy semantics in the top changes.

### Creative
- The creative adapter is structurally dominated: 73 of the top 100 changed features are BOS / boundary gates.
- The amplified semantic subset barely matches the expected creative-writing semantics (0.0% semantic match), so the adapter is not cleanly separable by its top changed features.
- The strongest non-structural families are code_function_definition (11), general_legal_reasoning (6), code_docstring_example (2).
- The creative adapter does not produce a strong narrative-specific signature in the top-changed set; structural and mixed prompt-format effects are larger.

## Universal Features

- Features changed in all five adapters: 264
- Dominant families among the top universal features: {'bos_boundary': 7, 'code_docstring_example': 1, 'code_function_definition': 7, 'general_legal_reasoning': 2, 'generic_instruction': 3, 'math_word_problem': 1, 'medical_clinical': 1, 'unlabeled': 3}

### Top Universal Features
- feature 1041: mean_abs_mass=64.469994, majority_family=bos_boundary, classes={'code': 'suppressed', 'medical': 'amplified', 'math': 'amplified', 'safety': 'amplified', 'creative': 'suppressed'}
- feature 6810: mean_abs_mass=56.163453, majority_family=code_docstring_example, classes={'code': 'amplified', 'medical': 'suppressed', 'math': 'suppressed', 'safety': 'suppressed', 'creative': 'amplified'}
- feature 2620: mean_abs_mass=35.688159, majority_family=general_legal_reasoning, classes={'code': 'suppressed', 'medical': 'suppressed', 'math': 'suppressed', 'safety': 'suppressed', 'creative': 'suppressed'}
- feature 7541: mean_abs_mass=30.087644, majority_family=code_function_definition, classes={'code': 'suppressed', 'medical': 'suppressed', 'math': 'amplified', 'safety': 'amplified', 'creative': 'suppressed'}
- feature 2291: mean_abs_mass=19.517273, majority_family=general_legal_reasoning, classes={'code': 'amplified', 'medical': 'suppressed', 'math': 'suppressed', 'safety': 'suppressed', 'creative': 'amplified'}
- feature 11306: mean_abs_mass=18.444289, majority_family=medical_clinical, classes={'code': 'suppressed', 'medical': 'suppressed', 'math': 'amplified', 'safety': 'suppressed', 'creative': 'suppressed'}
- feature 11767: mean_abs_mass=14.953183, majority_family=bos_boundary, classes={'code': 'suppressed', 'medical': 'suppressed', 'math': 'suppressed', 'safety': 'suppressed', 'creative': 'suppressed'}
- feature 11087: mean_abs_mass=14.009436, majority_family=unlabeled, classes={'code': 'suppressed', 'medical': 'suppressed', 'math': 'suppressed', 'safety': 'suppressed', 'creative': 'suppressed'}
- feature 8806: mean_abs_mass=12.800052, majority_family=bos_boundary, classes={'code': 'amplified', 'medical': 'suppressed', 'math': 'amplified', 'safety': 'amplified', 'creative': 'amplified'}
- feature 14599: mean_abs_mass=12.511624, majority_family=code_function_definition, classes={'code': 'amplified', 'medical': 'suppressed', 'math': 'suppressed', 'safety': 'suppressed', 'creative': 'amplified'}
