# Day 15: Offline Feature Labeling

Top changed SAE features were labeled from saved sparse activations plus cached Gemma tokenization.
This is an offline autointerp-style pass, not the EleutherAI `sae-auto-interp` stack referenced in the schedule.

| Adapter | Labeled Features | Most Common Label Families |
| --- | ---: | --- |
| code | 100 | bos_boundary (78), code_function_definition (9), general_legal_reasoning (7), code_docstring_example (2) |
| medical | 100 | bos_boundary (59), code_function_definition (11), general_legal_reasoning (7), medical_clinical (5) |
| math | 100 | bos_boundary (58), general_legal_reasoning (12), math_word_problem (12), code_function_definition (8) |
| safety | 100 | bos_boundary (71), general_legal_reasoning (10), code_docstring_example (5), math_word_problem (5) |
| creative | 100 | bos_boundary (73), code_function_definition (11), general_legal_reasoning (6), code_docstring_example (2) |

## Representative Labels

### code
- feature 7541 [suppressed]: Python function definitions and implementation framing (family=code_function_definition, confidence=0.754, primary_domain=math)
- feature 2009 [amplified]: legal / exam-style reasoning (family=general_legal_reasoning, confidence=0.767, primary_domain=general)
- feature 1041 [suppressed]: beginning-of-prompt / BOS gating (family=bos_boundary, confidence=0.843, primary_domain=math)
- feature 11306 [suppressed]: medical / clinical wording (family=medical_clinical, confidence=0.717, primary_domain=creative)
- feature 10775 [amplified]: Python function definitions and implementation framing (family=code_function_definition, confidence=0.714, primary_domain=math)

### medical
- feature 10967 [amplified]: Python function definitions and implementation framing (family=code_function_definition, confidence=0.778, primary_domain=math)
- feature 6565 [amplified]: medical / clinical wording (family=medical_clinical, confidence=0.786, primary_domain=general)
- feature 11306 [suppressed]: medical / clinical wording (family=medical_clinical, confidence=0.717, primary_domain=creative)
- feature 5650 [amplified]: Python function definitions and implementation framing (family=code_function_definition, confidence=0.740, primary_domain=code)
- feature 13431 [amplified]: creative-writing narrative language (family=creative_writing, confidence=0.786, primary_domain=general)

### math
- feature 6810 [suppressed]: legal / exam-style reasoning (family=general_legal_reasoning, confidence=0.749, primary_domain=general)
- feature 1041 [amplified]: beginning-of-prompt / BOS gating (family=bos_boundary, confidence=0.844, primary_domain=math)
- feature 5650 [suppressed]: Python docstrings, examples, and test-style scaffolding (family=code_docstring_example, confidence=0.737, primary_domain=code)
- feature 14906 [amplified]: Python function definitions and implementation framing (family=code_function_definition, confidence=0.754, primary_domain=math)
- feature 11313 [amplified]: beginning-of-prompt / BOS gating (family=bos_boundary, confidence=0.845, primary_domain=code)

### safety
- feature 1041 [amplified]: beginning-of-prompt / BOS gating (family=bos_boundary, confidence=0.843, primary_domain=math)
- feature 6810 [suppressed]: legal / exam-style reasoning (family=general_legal_reasoning, confidence=0.749, primary_domain=general)
- feature 2291 [suppressed]: legal / exam-style reasoning (family=general_legal_reasoning, confidence=0.712, primary_domain=math)
- feature 5472 [suppressed]: generic instructional framing (family=generic_instruction, confidence=0.766, primary_domain=creative)
- feature 13295 [amplified]: beginning-of-prompt / BOS gating (family=bos_boundary, confidence=0.821, primary_domain=code)

### creative
- feature 6810 [amplified]: Python docstrings, examples, and test-style scaffolding (family=code_docstring_example, confidence=0.743, primary_domain=general)
- feature 1041 [suppressed]: beginning-of-prompt / BOS gating (family=bos_boundary, confidence=0.843, primary_domain=math)
- feature 14599 [amplified]: legal / exam-style reasoning (family=general_legal_reasoning, confidence=0.736, primary_domain=math)
- feature 2620 [suppressed]: legal / exam-style reasoning (family=general_legal_reasoning, confidence=0.744, primary_domain=math)
- feature 5472 [suppressed]: generic instructional framing (family=generic_instruction, confidence=0.766, primary_domain=creative)
