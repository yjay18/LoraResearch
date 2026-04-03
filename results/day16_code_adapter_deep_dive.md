# Day 16: Code Adapter Deep Dive

This analysis uses the Day 15 labeled top-100 changed features for the `code` adapter.

## Top-100 Mix

- Class mix: {'amplified': 20, 'killed': 42, 'newly_activated': 6, 'suppressed': 32}
- Label families: {'bos_boundary': 78, 'code_docstring_example': 2, 'code_function_definition': 9, 'creative_writing': 1, 'general_legal_reasoning': 7, 'generic_instruction': 1, 'math_word_problem': 1, 'medical_clinical': 1}
- Amplified features matching code expectations: 30.0%
- Amplified non-structural features matching code expectations: 54.5%
- Suppressed features that are clearly non-code: 18.8%
- Suppressed non-structural features that are clearly non-code: 54.5%
- Structural BOS-dominated gates in the top 100: 78

## Interpretation

- The code adapter's strongest changes are still suppression-heavy, which matches the broader Day 14 result that adapters mostly reweight existing features rather than creating many new ones.
- Expected code semantics do appear in the amplified set: docstrings, implementation framing, and code-layout features all recur among the top amplified labels.
- The match is only partial. Several top amplified features are not pure Python syntax features; they instead look like generic instructional scaffolding or even legal/exam-style reasoning.
- Many of the largest `newly_activated` and `killed` features are BOS / structural gates rather than rich semantic concepts. That suggests some of the adapter effect is a global routing change, not just domain-specific concept insertion.
- The suppressed set contains many clearly non-code families, which is the expected direction for a code adapter, but the presence of math- and general-reasoning features among the amplified set suggests the underlying prompt format still matters a lot.

## Representative Features

### Amplified
- feature 2009: legal / exam-style reasoning (class=amplified, family=general_legal_reasoning, primary_domain=general, top_tokens=function, indentation, to)
- feature 10775: Python function definitions and implementation framing (class=amplified, family=code_function_definition, primary_domain=math, top_tokens=<bos>, ?, \n)
- feature 6810: Python docstrings, examples, and test-style scaffolding (class=amplified, family=code_docstring_example, primary_domain=general, top_tokens=the, a, of)
- feature 9059: Python function definitions and implementation framing (class=amplified, family=code_function_definition, primary_domain=math, top_tokens=<bos>, ., 0)
- feature 13700: beginning-of-prompt / BOS gating (class=amplified, family=bos_boundary, primary_domain=general, top_tokens=<bos>, of, function)

### Suppressed
- feature 7541: Python function definitions and implementation framing (class=suppressed, family=code_function_definition, primary_domain=math, top_tokens=of, <bos>, the)
- feature 1041: beginning-of-prompt / BOS gating (class=suppressed, family=bos_boundary, primary_domain=math, top_tokens=<bos>, ., ?)
- feature 11306: medical / clinical wording (class=suppressed, family=medical_clinical, primary_domain=creative, top_tokens=?, ., a)
- feature 10708: Python function definitions and implementation framing (class=suppressed, family=code_function_definition, primary_domain=code, top_tokens=the, of, a)
- feature 4681: Python function definitions and implementation framing (class=suppressed, family=code_function_definition, primary_domain=code, top_tokens=the, of, a)

### Newly Activated
- feature 8569: beginning-of-prompt / BOS gating (class=newly_activated, family=bos_boundary, primary_domain=creative, top_tokens=<bos>, un)
- feature 16162: beginning-of-prompt / BOS gating (class=newly_activated, family=bos_boundary, primary_domain=math, top_tokens=<bos>)
- feature 11848: beginning-of-prompt / BOS gating (class=newly_activated, family=bos_boundary, primary_domain=code, top_tokens=<bos>, """)
- feature 12722: beginning-of-prompt / BOS gating (class=newly_activated, family=bos_boundary, primary_domain=math, top_tokens=<bos>)
- feature 2321: beginning-of-prompt / BOS gating (class=newly_activated, family=bos_boundary, primary_domain=math, top_tokens=<bos>)

### Killed
- feature 2594: beginning-of-prompt / BOS gating (class=killed, family=bos_boundary, primary_domain=safety, top_tokens=<bos>)
- feature 7659: beginning-of-prompt / BOS gating (class=killed, family=bos_boundary, primary_domain=math, top_tokens=<bos>)
- feature 135: beginning-of-prompt / BOS gating (class=killed, family=bos_boundary, primary_domain=general, top_tokens=<bos>)
- feature 4323: beginning-of-prompt / BOS gating (class=killed, family=bos_boundary, primary_domain=general, top_tokens=<bos>, same)
- feature 6165: beginning-of-prompt / BOS gating (class=killed, family=bos_boundary, primary_domain=safety, top_tokens=<bos>)
