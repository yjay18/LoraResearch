# Medical Adapter Deep Dive

This analysis uses the Day 15 labeled top-100 changed features for the adapter.

## Top-100 Mix

- Class mix: {'amplified': 27, 'killed': 26, 'newly_activated': 5, 'suppressed': 42}
- Label families: {'bos_boundary': 59, 'code_docstring_example': 4, 'code_function_definition': 11, 'creative_writing': 4, 'general_legal_reasoning': 7, 'generic_instruction': 4, 'math_word_problem': 4, 'medical_clinical': 5, 'safety_sensitive': 2}
- Raw amplified expected-family match: 11.1%
- Non-structural amplified expected-family match: 14.3%
- Raw suppressed non-expected rate: 26.2%
- Non-structural suppressed non-expected rate: 55.0%
- Structural BOS-dominated gates in the top 100: 59

## Interpretation

- The medical adapter is structurally dominated: 59 of the top 100 changed features are BOS / boundary gates.
- The amplified semantic subset barely matches the expected medical / clinical semantics (14.3% semantic match), so the adapter is not cleanly separable by its top changed features.
- The strongest non-structural families are code_function_definition (11), general_legal_reasoning (7), medical_clinical (5).
- Compared with the kind of clinical lexical structure you would expect from MIMIC-style text, the medical adapter's biggest shifts still look more structural and mixed-domain than strongly clinical.

## Representative Features

### Amplified
- feature 10967: Python function definitions and implementation framing (class=amplified, family=code_function_definition, primary_domain=math, top_tokens=?, of, a)
- feature 6565: medical / clinical wording (class=amplified, family=medical_clinical, primary_domain=general, top_tokens=?, ., of)
- feature 5650: Python function definitions and implementation framing (class=amplified, family=code_function_definition, primary_domain=code, top_tokens=<bos>, many, the)
- feature 13431: creative-writing narrative language (class=amplified, family=creative_writing, primary_domain=general, top_tokens=., the, ?)
- feature 11412: Python function definitions and implementation framing (class=amplified, family=code_function_definition, primary_domain=code, top_tokens=of, the, .)

### Suppressed
- feature 11306: medical / clinical wording (class=suppressed, family=medical_clinical, primary_domain=creative, top_tokens=?, ., a)
- feature 2620: legal / exam-style reasoning (class=suppressed, family=general_legal_reasoning, primary_domain=math, top_tokens=?, ., of)
- feature 5472: generic instructional framing (class=suppressed, family=generic_instruction, primary_domain=creative, top_tokens=of, to, in)
- feature 7541: Python function definitions and implementation framing (class=suppressed, family=code_function_definition, primary_domain=math, top_tokens=of, <bos>, the)
- feature 8806: beginning-of-prompt / BOS gating (class=suppressed, family=bos_boundary, primary_domain=general, top_tokens=<bos>, of, ,)

### Newly Activated
- feature 12722: beginning-of-prompt / BOS gating (class=newly_activated, family=bos_boundary, primary_domain=code, top_tokens=<bos>)
- feature 8569: beginning-of-prompt / BOS gating (class=newly_activated, family=bos_boundary, primary_domain=code, top_tokens=<bos>)
- feature 1216: beginning-of-prompt / BOS gating (class=newly_activated, family=bos_boundary, primary_domain=general, top_tokens=<bos>, 1, renewable)
- feature 6915: beginning-of-prompt / BOS gating (class=newly_activated, family=bos_boundary, primary_domain=creative, top_tokens=<bos>, space_boundary, -)
- feature 9938: beginning-of-prompt / BOS gating (class=newly_activated, family=bos_boundary, primary_domain=code, top_tokens=<bos>, equal, to)

### Killed
- feature 2594: beginning-of-prompt / BOS gating (class=killed, family=bos_boundary, primary_domain=safety, top_tokens=<bos>)
- feature 3149: beginning-of-prompt / BOS gating (class=killed, family=bos_boundary, primary_domain=safety, top_tokens=<bos>)
- feature 193: beginning-of-prompt / BOS gating (class=killed, family=bos_boundary, primary_domain=math, top_tokens=<bos>)
- feature 2711: beginning-of-prompt / BOS gating (class=killed, family=bos_boundary, primary_domain=code, top_tokens=<bos>, \n)
- feature 2908: beginning-of-prompt / BOS gating (class=killed, family=bos_boundary, primary_domain=general, top_tokens=<bos>)
