# Day 14: Feature Classification

Each adapter's Day 13 differential table was mapped into one exclusive class per feature.
Empirical significance thresholds use the 95th percentile of effect sizes plus support floors.

| Adapter | Amplified | Suppressed | New | Killed | Context Shifted | Unchanged |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| code | 610 | 1165 | 296 | 313 | 22 | 13978 |
| medical | 762 | 1200 | 184 | 284 | 50 | 13904 |
| math | 521 | 1487 | 97 | 357 | 32 | 13890 |
| safety | 561 | 1459 | 100 | 351 | 17 | 13896 |
| creative | 508 | 1253 | 269 | 320 | 27 | 14007 |

## Thresholds

### code
- Support floor: features firing on fewer than `5` prompts are treated as effectively absent.
- Significant frequency shift: `|delta_token_freq| >= 0.004544`
- Significant activation-mass shift: `|delta_mean_prompt_activation| >= 2.514230`
- Significant context shift: `context_shift >= 0.373724` with `prompt_flip_rate >= 0.030`

### medical
- Support floor: features firing on fewer than `5` prompts are treated as effectively absent.
- Significant frequency shift: `|delta_token_freq| >= 0.002908`
- Significant activation-mass shift: `|delta_mean_prompt_activation| >= 1.120878`
- Significant context shift: `context_shift >= 0.227360` with `prompt_flip_rate >= 0.030`

### math
- Support floor: features firing on fewer than `5` prompts are treated as effectively absent.
- Significant frequency shift: `|delta_token_freq| >= 0.002545`
- Significant activation-mass shift: `|delta_mean_prompt_activation| >= 0.925302`
- Significant context shift: `context_shift >= 0.186697` with `prompt_flip_rate >= 0.030`

### safety
- Support floor: features firing on fewer than `5` prompts are treated as effectively absent.
- Significant frequency shift: `|delta_token_freq| >= 0.002454`
- Significant activation-mass shift: `|delta_mean_prompt_activation| >= 1.149558`
- Significant context shift: `context_shift >= 0.217841` with `prompt_flip_rate >= 0.030`

### creative
- Support floor: features firing on fewer than `5` prompts are treated as effectively absent.
- Significant frequency shift: `|delta_token_freq| >= 0.003908`
- Significant activation-mass shift: `|delta_mean_prompt_activation| >= 2.507882`
- Significant context shift: `context_shift >= 0.356354` with `prompt_flip_rate >= 0.030`

## Representative Features

### code
- Amplified:
  feature 2009: score=36.308, delta_token_freq=0.1045983278807706, delta_mean_prompt_activation=33.409225, delta_prompt_freq=0.3966666666666667, adapted_top_prompt_domain=general, domain_switch=False
  feature 6810: score=24.615, delta_token_freq=0.0421664849145764, delta_mean_prompt_activation=38.556213, delta_prompt_freq=0.0, adapted_top_prompt_domain=general, domain_switch=False
  feature 9059: score=22.456, delta_token_freq=0.0733369683751363, delta_mean_prompt_activation=15.879492, delta_prompt_freq=0.0, adapted_top_prompt_domain=math, domain_switch=False
- Suppressed:
  feature 7541: score=60.481, delta_token_freq=-0.1403126135950563, delta_mean_prompt_activation=-74.42496, delta_prompt_freq=-0.0333333333333333, base_top_prompt_domain=math, domain_switch=False
  feature 1041: score=40.209, delta_token_freq=-0.0377135587059251, delta_mean_prompt_activation=-80.22778, delta_prompt_freq=0.0, base_top_prompt_domain=math, domain_switch=False
  feature 11306: score=27.199, delta_token_freq=-0.0727008360596146, delta_mean_prompt_activation=-28.157288, delta_prompt_freq=0.0, base_top_prompt_domain=creative, domain_switch=True
- Newly Activated:
  feature 8569: score=300.000, base_prompt_count=0, adapted_prompt_count=300, delta_token_freq=0.02735368956743, adapted_top_prompt_domain=creative
  feature 11848: score=300.000, base_prompt_count=0, adapted_prompt_count=300, delta_token_freq=0.02735368956743, adapted_top_prompt_domain=code
  feature 12722: score=300.000, base_prompt_count=0, adapted_prompt_count=300, delta_token_freq=0.0272628135223555, adapted_top_prompt_domain=math
- Killed:
  feature 135: score=300.000, base_prompt_count=300, adapted_prompt_count=0, delta_token_freq=-0.0272628135223555, base_top_prompt_domain=general
  feature 1744: score=300.000, base_prompt_count=300, adapted_prompt_count=0, delta_token_freq=-0.0272628135223555, base_top_prompt_domain=code
  feature 2456: score=300.000, base_prompt_count=300, adapted_prompt_count=0, delta_token_freq=-0.0272628135223555, base_top_prompt_domain=creative
- Context Shifted:
  feature 8032: score=1.981, context_shift=0.7401812970638275, prompt_flip_rate=0.11, base_top_prompt_domain=general, adapted_top_prompt_domain=code, domain_switch=True
  feature 3638: score=1.661, context_shift=0.620780736207962, prompt_flip_rate=0.0333333333333333, base_top_prompt_domain=safety, adapted_top_prompt_domain=general, domain_switch=True
  feature 11892: score=1.473, context_shift=0.5506607592105865, prompt_flip_rate=0.13, base_top_prompt_domain=safety, adapted_top_prompt_domain=safety, domain_switch=False

### medical
- Amplified:
  feature 10967: score=56.636, delta_token_freq=0.0828789531079607, delta_mean_prompt_activation=31.536858, delta_prompt_freq=0.05, adapted_top_prompt_domain=math, domain_switch=False
  feature 5650: score=43.485, delta_token_freq=0.0817884405670665, delta_mean_prompt_activation=17.216366, delta_prompt_freq=0.0, adapted_top_prompt_domain=code, domain_switch=False
  feature 6565: score=42.444, delta_token_freq=0.0754271174118502, delta_mean_prompt_activation=18.501406, delta_prompt_freq=0.1433333333333333, adapted_top_prompt_domain=general, domain_switch=False
- Suppressed:
  feature 11306: score=45.337, delta_token_freq=-0.0706106870229008, delta_mean_prompt_activation=-23.600739, delta_prompt_freq=-0.0033333333333332, base_top_prompt_domain=creative, domain_switch=True
  feature 2620: score=40.059, delta_token_freq=-0.0024536532170119, delta_mean_prompt_activation=-43.9552, delta_prompt_freq=0.0, base_top_prompt_domain=math, domain_switch=False
  feature 5472: score=34.510, delta_token_freq=-0.0663395129043984, delta_mean_prompt_activation=-13.1117, delta_prompt_freq=-0.1666666666666666, base_top_prompt_domain=creative, domain_switch=False
- Newly Activated:
  feature 8569: score=300.000, base_prompt_count=0, adapted_prompt_count=300, delta_token_freq=0.0272628135223555, adapted_top_prompt_domain=code
  feature 12722: score=300.000, base_prompt_count=0, adapted_prompt_count=300, delta_token_freq=0.0272628135223555, adapted_top_prompt_domain=code
  feature 6915: score=299.000, base_prompt_count=1, adapted_prompt_count=300, delta_token_freq=0.02735368956743, adapted_top_prompt_domain=creative
- Killed:
  feature 193: score=300.000, base_prompt_count=300, adapted_prompt_count=0, delta_token_freq=-0.0272628135223555, base_top_prompt_domain=math
  feature 468: score=300.000, base_prompt_count=300, adapted_prompt_count=0, delta_token_freq=-0.0272628135223555, base_top_prompt_domain=safety
  feature 2456: score=300.000, base_prompt_count=300, adapted_prompt_count=0, delta_token_freq=-0.0272628135223555, base_top_prompt_domain=creative
- Context Shifted:
  feature 3638: score=3.215, context_shift=0.7308946549892426, prompt_flip_rate=0.04, base_top_prompt_domain=safety, adapted_top_prompt_domain=general, domain_switch=True
  feature 8629: score=2.708, context_shift=0.6156883239746094, prompt_flip_rate=0.0533333333333333, base_top_prompt_domain=general, adapted_top_prompt_domain=code, domain_switch=True
  feature 3211: score=2.486, context_shift=0.5652219951152802, prompt_flip_rate=0.0366666666666666, base_top_prompt_domain=creative, adapted_top_prompt_domain=creative, domain_switch=False

### math
- Amplified:
  feature 1041: score=66.852, delta_token_freq=0.0440748818611413, delta_mean_prompt_activation=45.831055, delta_prompt_freq=0.0, adapted_top_prompt_domain=math, domain_switch=False
  feature 11313: score=37.195, delta_token_freq=0.0581606688476917, delta_mean_prompt_activation=13.26709, delta_prompt_freq=0.0, adapted_top_prompt_domain=code, domain_switch=False
  feature 14906: score=35.028, delta_token_freq=0.0457106506724827, delta_mean_prompt_activation=15.789127, delta_prompt_freq=0.0566666666666666, adapted_top_prompt_domain=math, domain_switch=False
- Suppressed:
  feature 6810: score=90.845, delta_token_freq=-0.0475281715739731, delta_mean_prompt_activation=-66.77533, delta_prompt_freq=-0.0066666666666665, base_top_prompt_domain=general, domain_switch=False
  feature 5650: score=46.169, delta_token_freq=-0.0800617957106506, delta_mean_prompt_activation=-13.6066475, delta_prompt_freq=0.0, base_top_prompt_domain=code, domain_switch=False
  feature 14599: score=28.514, delta_token_freq=-0.0389858233369683, delta_mean_prompt_activation=-12.207115, delta_prompt_freq=0.0, base_top_prompt_domain=math, domain_switch=False
- Newly Activated:
  feature 8569: score=300.000, base_prompt_count=0, adapted_prompt_count=300, delta_token_freq=0.0272628135223555, adapted_top_prompt_domain=creative
  feature 4585: score=299.000, base_prompt_count=1, adapted_prompt_count=300, delta_token_freq=0.0272628135223555, adapted_top_prompt_domain=creative
  feature 2321: score=144.000, base_prompt_count=2, adapted_prompt_count=146, delta_token_freq=0.0130861504907306, adapted_top_prompt_domain=safety
- Killed:
  feature 461: score=300.000, base_prompt_count=300, adapted_prompt_count=0, delta_token_freq=-0.0272628135223555, base_top_prompt_domain=code
  feature 1744: score=300.000, base_prompt_count=300, adapted_prompt_count=0, delta_token_freq=-0.0272628135223555, base_top_prompt_domain=code
  feature 2075: score=300.000, base_prompt_count=300, adapted_prompt_count=0, delta_token_freq=-0.0272628135223555, base_top_prompt_domain=math
- Context Shifted:
  feature 3387: score=3.296, context_shift=0.6152989864349365, prompt_flip_rate=0.03, base_top_prompt_domain=math, adapted_top_prompt_domain=math, domain_switch=False
  feature 11563: score=2.977, context_shift=0.5557775795459747, prompt_flip_rate=0.03, base_top_prompt_domain=code, adapted_top_prompt_domain=code, domain_switch=False
  feature 9117: score=2.851, context_shift=0.532303512096405, prompt_flip_rate=0.07, base_top_prompt_domain=math, adapted_top_prompt_domain=general, domain_switch=True

### safety
- Amplified:
  feature 1041: score=118.475, delta_token_freq=0.0816066884769174, delta_mean_prompt_activation=97.95996, delta_prompt_freq=0.0, adapted_top_prompt_domain=math, domain_switch=False
  feature 13295: score=34.580, delta_token_freq=0.0408033442384587, delta_mean_prompt_activation=20.63488, delta_prompt_freq=0.0, adapted_top_prompt_domain=code, domain_switch=False
  feature 8215: score=29.436, delta_token_freq=0.0480734278444202, delta_mean_prompt_activation=11.315918, delta_prompt_freq=0.0833333333333333, adapted_top_prompt_domain=math, domain_switch=False
- Suppressed:
  feature 6810: score=93.895, delta_token_freq=-0.0637041075972373, delta_mean_prompt_activation=-78.09122, delta_prompt_freq=-0.0066666666666665, base_top_prompt_domain=general, domain_switch=False
  feature 2291: score=44.669, delta_token_freq=-0.0492548164303889, delta_mean_prompt_activation=-28.27353, delta_prompt_freq=0.0, base_top_prompt_domain=math, domain_switch=False
  feature 5472: score=30.581, delta_token_freq=-0.0518902217375499, delta_mean_prompt_activation=-10.844168, delta_prompt_freq=-0.1066666666666666, base_top_prompt_domain=creative, domain_switch=False
- Newly Activated:
  feature 1019: score=296.000, base_prompt_count=4, adapted_prompt_count=300, delta_token_freq=0.0272628135223555, adapted_top_prompt_domain=general
  feature 6957: score=296.000, base_prompt_count=4, adapted_prompt_count=300, delta_token_freq=0.02735368956743, adapted_top_prompt_domain=code
  feature 7263: score=296.000, base_prompt_count=4, adapted_prompt_count=300, delta_token_freq=0.0274445656125045, adapted_top_prompt_domain=general
- Killed:
  feature 193: score=300.000, base_prompt_count=300, adapted_prompt_count=0, delta_token_freq=-0.0272628135223555, base_top_prompt_domain=math
  feature 1744: score=300.000, base_prompt_count=300, adapted_prompt_count=0, delta_token_freq=-0.0272628135223555, base_top_prompt_domain=code
  feature 2456: score=300.000, base_prompt_count=300, adapted_prompt_count=0, delta_token_freq=-0.0272628135223555, base_top_prompt_domain=creative
- Context Shifted:
  feature 10191: score=1.940, context_shift=0.422659695148468, prompt_flip_rate=0.04, base_top_prompt_domain=general, adapted_top_prompt_domain=creative, domain_switch=True
  feature 10880: score=1.912, context_shift=0.4164331555366516, prompt_flip_rate=0.04, base_top_prompt_domain=safety, adapted_top_prompt_domain=general, domain_switch=True
  feature 12002: score=1.806, context_shift=0.3934199213981628, prompt_flip_rate=0.04, base_top_prompt_domain=general, adapted_top_prompt_domain=general, domain_switch=False

### creative
- Amplified:
  feature 6810: score=52.702, delta_token_freq=0.0680661577608142, delta_mean_prompt_activation=88.48633, delta_prompt_freq=0.0033333333333334, adapted_top_prompt_domain=general, domain_switch=False
  feature 14599: score=38.457, delta_token_freq=0.0996910214467466, delta_mean_prompt_activation=32.46495, delta_prompt_freq=0.0, adapted_top_prompt_domain=math, domain_switch=False
  feature 15848: score=25.193, delta_token_freq=0.0790621592148309, delta_mean_prompt_activation=12.440189, delta_prompt_freq=0.2033333333333333, adapted_top_prompt_domain=math, domain_switch=False
- Suppressed:
  feature 1041: score=50.814, delta_token_freq=-0.0635223555070883, delta_mean_prompt_activation=-86.66748, delta_prompt_freq=0.0, base_top_prompt_domain=math, domain_switch=False
  feature 2620: score=36.698, delta_token_freq=-0.0089058524173027, delta_mean_prompt_activation=-86.319336, delta_prompt_freq=0.0, base_top_prompt_domain=math, domain_switch=False
  feature 11306: score=31.820, delta_token_freq=-0.0806070519810978, delta_mean_prompt_activation=-28.068527, delta_prompt_freq=-0.0066666666666665, base_top_prompt_domain=creative, domain_switch=True
- Newly Activated:
  feature 4691: score=300.000, base_prompt_count=0, adapted_prompt_count=300, delta_token_freq=0.0272628135223555, adapted_top_prompt_domain=code
  feature 11848: score=300.000, base_prompt_count=0, adapted_prompt_count=300, delta_token_freq=0.0272628135223555, adapted_top_prompt_domain=safety
  feature 12722: score=300.000, base_prompt_count=0, adapted_prompt_count=300, delta_token_freq=0.0272628135223555, adapted_top_prompt_domain=medical
- Killed:
  feature 2075: score=300.000, base_prompt_count=300, adapted_prompt_count=0, delta_token_freq=-0.0272628135223555, base_top_prompt_domain=math
  feature 2456: score=300.000, base_prompt_count=300, adapted_prompt_count=0, delta_token_freq=-0.0272628135223555, base_top_prompt_domain=creative
  feature 2594: score=300.000, base_prompt_count=300, adapted_prompt_count=0, delta_token_freq=-0.0272628135223555, base_top_prompt_domain=safety
- Context Shifted:
  feature 3387: score=2.265, context_shift=0.806988313794136, prompt_flip_rate=0.0466666666666666, base_top_prompt_domain=math, adapted_top_prompt_domain=general, domain_switch=True
  feature 16001: score=1.962, context_shift=0.6990385353565216, prompt_flip_rate=0.12, base_top_prompt_domain=general, adapted_top_prompt_domain=code, domain_switch=True
  feature 3064: score=1.873, context_shift=0.6675285696983337, prompt_flip_rate=0.0366666666666666, base_top_prompt_domain=creative, adapted_top_prompt_domain=general, domain_switch=True
