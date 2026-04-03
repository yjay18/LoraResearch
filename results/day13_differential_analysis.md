# Day 13: Differential Feature Analysis

Per-feature comparisons between base SAE activations and each LoRA-adapted model.

| Adapter | Mean Abs Delta Token Freq | Mean Delta Mean Active | Mean Context Shift | Domain Switches |
| --- | ---: | ---: | ---: | ---: |
| code | 0.001274 | -0.303015 | 0.059107 | 2139 |
| medical | 0.000879 | -0.123404 | 0.049727 | 2003 |
| math | 0.000741 | -0.069120 | 0.043590 | 1875 |
| safety | 0.000780 | -0.110958 | 0.041312 | 1785 |
| creative | 0.001181 | -0.287189 | 0.060492 | 2092 |

## Top Features By Adapter

### code
- Top amplified by frequency:
  feature 2009: delta_token_freq=0.104598, delta_prompt_freq=0.396667, delta_mean_active=-0.193711
  feature 9059: delta_token_freq=0.073337, delta_prompt_freq=0.000000, delta_mean_active=-0.099732
  feature 13700: delta_token_freq=0.069338, delta_prompt_freq=0.000000, delta_mean_active=-8.160232
  feature 10775: delta_token_freq=0.066794, delta_prompt_freq=0.896667, delta_mean_active=-0.441717
  feature 8806: delta_token_freq=0.050800, delta_prompt_freq=0.000000, delta_mean_active=-0.363654
- Top context shifted:
  feature 10: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.003333
  feature 36: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.006667
  feature 62: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.006667
  feature 110: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.023333
  feature 135: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=1.000000

### medical
- Top amplified by frequency:
  feature 10967: delta_token_freq=0.082879, delta_prompt_freq=0.050000, delta_mean_active=0.607488
  feature 5650: delta_token_freq=0.081788, delta_prompt_freq=0.000000, delta_mean_active=0.168856
  feature 6565: delta_token_freq=0.075427, delta_prompt_freq=0.143333, delta_mean_active=-0.064040
  feature 13431: delta_token_freq=0.067612, delta_prompt_freq=0.300000, delta_mean_active=-0.214333
  feature 15454: delta_token_freq=0.065703, delta_prompt_freq=0.000000, delta_mean_active=-5.365221
- Top context shifted:
  feature 10: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.003333
  feature 110: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.010000
  feature 193: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=1.000000
  feature 370: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.006667
  feature 375: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.003333

### math
- Top amplified by frequency:
  feature 11313: delta_token_freq=0.058161, delta_prompt_freq=0.000000, delta_mean_active=-3.732157
  feature 14906: delta_token_freq=0.045711, delta_prompt_freq=0.056667, delta_mean_active=0.449421
  feature 1041: delta_token_freq=0.044075, delta_prompt_freq=0.000000, delta_mean_active=-6.185446
  feature 13700: delta_token_freq=0.040349, delta_prompt_freq=0.000000, delta_mean_active=-5.190912
  feature 11294: delta_token_freq=0.035987, delta_prompt_freq=0.483333, delta_mean_active=-0.553890
- Top context shifted:
  feature 36: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.003333
  feature 40: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.016667
  feature 122: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.010000
  feature 342: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.003333
  feature 461: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=1.000000

### safety
- Top amplified by frequency:
  feature 1041: delta_token_freq=0.081607, delta_prompt_freq=0.000000, delta_mean_active=-10.044625
  feature 11264: delta_token_freq=0.051981, delta_prompt_freq=0.376667, delta_mean_active=-0.064880
  feature 8215: delta_token_freq=0.048073, delta_prompt_freq=0.083333, delta_mean_active=-0.396649
  feature 13295: delta_token_freq=0.040803, delta_prompt_freq=0.000000, delta_mean_active=0.026214
  feature 7708: delta_token_freq=0.036714, delta_prompt_freq=0.083333, delta_mean_active=0.098616
- Top context shifted:
  feature 36: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.003333
  feature 193: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=1.000000
  feature 375: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.003333
  feature 835: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.003333
  feature 892: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.003333

### creative
- Top amplified by frequency:
  feature 14599: delta_token_freq=0.099691, delta_prompt_freq=0.000000, delta_mean_active=0.063233
  feature 15848: delta_token_freq=0.079062, delta_prompt_freq=0.203333, delta_mean_active=0.221767
  feature 6810: delta_token_freq=0.068066, delta_prompt_freq=0.003333, delta_mean_active=1.717412
  feature 9059: delta_token_freq=0.065340, delta_prompt_freq=0.000000, delta_mean_active=-0.102486
  feature 336: delta_token_freq=0.061705, delta_prompt_freq=0.000000, delta_mean_active=-4.647614
- Top context shifted:
  feature 36: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.006667
  feature 375: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.006667
  feature 433: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.003333
  feature 482: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.003333
  feature 556: context_shift=1.000000, context_jaccard=0.000000, prompt_flip_rate=0.003333
