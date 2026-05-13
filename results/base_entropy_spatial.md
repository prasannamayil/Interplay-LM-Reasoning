# BASE-model per-token entropy: spatial breakdown within Define lines

_BASE checkpoint: `/fast/pmayilvahanan/Interplay-LM-Reasoning/saves/gsm_infinity/pt_op2-10_10B_alltemps_skewed_v4`_

_rollouts: `/fast/pmayilvahanan/Interplay-LM-Reasoning/results/gsm_infinity_rl_v4/BASE_v4/global_step_0/eval_phase1c/rollouts/rollouts.2146019.jsonl`_

_ops: [14, 17, 18, 20], n_prompts/op: 25, n_samples/prompt: 16, max_len: 1024_


## op14

n_rollouts (after max-len filter): 384

### Per-region entropy (BASE model)

| region | all H | first-tok H | correct H | wrong H | gap (c - w) | n |
|---|---:|---:|---:|---:|---:|---:|
| define_kw | 0.0053 | 0.0053 | 0.0043 | 0.0076 | -0.0032 | 2387 |
| var_name | 0.0255 | 0.1348 | 0.0264 | 0.0231 | +0.0033 | 2387 |
| as_link | 0.8182 | 0.0000 | 0.8512 | 0.7418 | +0.1094 | 2387 |
| lhs_body | 0.2944 | 3.0777 | 0.2995 | 0.2794 | +0.0201 | 2211 |
| eq | 0.0010 | 0.0010 | 0.0007 | 0.0019 | -0.0012 | 2211 |
| rhs | 0.0121 | 0.0137 | 0.0059 | 0.0288 | -0.0229 | 2211 |
| format_tail | 0.0032 | 0.0005 | 0.0005 | 0.0105 | -0.0101 | 2211 |
| body_no_eq | 0.0001 | 0.0002 | 0.0001 | 0.0001 | +0.0000 | 176 |

### Within-rollout median Spearman ρ(signal, step_correct)

| signal | median ρ | n_rollouts |
|---|---:|---:|
| full_mean_H | +0.2887 | 340 |
| rhs_mean_H | -0.4140 | 330 |
| lhs_body_mean_H | +0.1309 | 330 |
| varname_mean_H | +0.1111 | 340 |

### Mean BASE entropy at token offset relative to '='

| offset | mean H | n |
|---:|---:|---:|
| -3 | 0.0205 | 2211 |
| -2 | 0.0011 | 2211 |
| -1 | 0.0096 | 2211 |
| +0 | 0.0010 | 2211 |
| +1 | 0.0137 | 2211 |
| +2 | 0.0004 | 2211 |
| +3 | 0.0044 | 2211 |
| +4 | 0.0816 | 2211 |
| +5 | 0.0372 | 2211 |
| +6 | 0.0222 | 2211 |


## op17

n_rollouts (after max-len filter): 368

### Per-region entropy (BASE model)

| region | all H | first-tok H | correct H | wrong H | gap (c - w) | n |
|---|---:|---:|---:|---:|---:|---:|
| define_kw | 0.0043 | 0.0043 | 0.0030 | 0.0064 | -0.0034 | 2266 |
| var_name | 0.0269 | 0.1199 | 0.0285 | 0.0224 | +0.0060 | 2266 |
| as_link | 0.8141 | 0.0000 | 0.8651 | 0.7331 | +0.1319 | 2266 |
| lhs_body | 0.2926 | 3.0887 | 0.2994 | 0.2818 | +0.0176 | 2090 |
| eq | 0.0026 | 0.0026 | 0.0001 | 0.0072 | -0.0071 | 2090 |
| rhs | 0.0513 | 0.0514 | 0.0161 | 0.1174 | -0.1013 | 2090 |
| format_tail | 0.0036 | 0.0001 | 0.0005 | 0.0092 | -0.0087 | 2090 |
| body_no_eq | 0.0001 | 0.0001 | 0.0001 | 0.0001 | +0.0000 | 176 |

### Within-rollout median Spearman ρ(signal, step_correct)

| signal | median ρ | n_rollouts |
|---|---:|---:|
| full_mean_H | +0.2887 | 365 |
| rhs_mean_H | -0.6124 | 313 |
| lhs_body_mean_H | +0.1309 | 313 |
| varname_mean_H | +0.0563 | 365 |

### Mean BASE entropy at token offset relative to '='

| offset | mean H | n |
|---:|---:|---:|
| -3 | 0.0331 | 2090 |
| -2 | 0.0033 | 2090 |
| -1 | 0.0348 | 2090 |
| +0 | 0.0026 | 2090 |
| +1 | 0.0514 | 2090 |
| +2 | 0.0002 | 2090 |
| +3 | 0.0048 | 2090 |
| +4 | 0.0699 | 2090 |
| +5 | 0.0496 | 2090 |
| +6 | 0.0505 | 2090 |


## op18

n_rollouts (after max-len filter): 352

### Per-region entropy (BASE model)

| region | all H | first-tok H | correct H | wrong H | gap (c - w) | n |
|---|---:|---:|---:|---:|---:|---:|
| define_kw | 0.0013 | 0.0013 | 0.0002 | 0.0028 | -0.0026 | 2209 |
| var_name | 0.0214 | 0.1426 | 0.0171 | 0.0276 | -0.0105 | 2209 |
| as_link | 0.8085 | 0.0000 | 0.8394 | 0.7639 | +0.0754 | 2209 |
| lhs_body | 0.2900 | 3.0734 | 0.2977 | 0.2769 | +0.0208 | 2026 |
| eq | 0.0022 | 0.0022 | 0.0001 | 0.0056 | -0.0055 | 2026 |
| rhs | 0.0181 | 0.0197 | 0.0256 | 0.0060 | +0.0196 | 2026 |
| format_tail | 0.0008 | 0.0006 | 0.0001 | 0.0019 | -0.0018 | 2026 |
| body_no_eq | 0.0001 | 0.0002 | 0.0001 | 0.0001 | -0.0000 | 183 |

### Within-rollout median Spearman ρ(signal, step_correct)

| signal | median ρ | n_rollouts |
|---|---:|---:|
| full_mean_H | +0.0000 | 342 |
| rhs_mean_H | -0.2887 | 323 |
| lhs_body_mean_H | +0.0976 | 323 |
| varname_mean_H | -0.1309 | 342 |

### Mean BASE entropy at token offset relative to '='

| offset | mean H | n |
|---:|---:|---:|
| -3 | 0.0240 | 2026 |
| -2 | 0.0033 | 2026 |
| -1 | 0.0191 | 2026 |
| +0 | 0.0022 | 2026 |
| +1 | 0.0197 | 2026 |
| +2 | 0.0006 | 2026 |
| +3 | 0.0015 | 2026 |
| +4 | 0.0977 | 2026 |
| +5 | 0.0222 | 2026 |
| +6 | 0.0232 | 2026 |


## op20

n_rollouts (after max-len filter): 383

### Per-region entropy (BASE model)

| region | all H | first-tok H | correct H | wrong H | gap (c - w) | n |
|---|---:|---:|---:|---:|---:|---:|
| define_kw | 0.0068 | 0.0068 | 0.0019 | 0.0147 | -0.0128 | 2445 |
| var_name | 0.0344 | 0.1301 | 0.0280 | 0.0390 | -0.0110 | 2445 |
| as_link | 0.8315 | 0.0000 | 0.8468 | 0.8069 | +0.0399 | 2445 |
| lhs_body | 0.3039 | 3.0676 | 0.3228 | 0.2740 | +0.0488 | 2302 |
| eq | 0.0004 | 0.0004 | 0.0001 | 0.0010 | -0.0009 | 2302 |
| rhs | 0.0261 | 0.0259 | 0.0051 | 0.0596 | -0.0546 | 2302 |
| format_tail | 0.0041 | 0.0009 | 0.0011 | 0.0094 | -0.0083 | 2302 |
| body_no_eq | 0.0004 | 0.0057 | 0.0005 | 0.0004 | +0.0001 | 143 |

### Within-rollout median Spearman ρ(signal, step_correct)

| signal | median ρ | n_rollouts |
|---|---:|---:|
| full_mean_H | +0.1443 | 368 |
| rhs_mean_H | -0.4880 | 351 |
| lhs_body_mean_H | +0.2887 | 351 |
| varname_mean_H | +0.0000 | 368 |

### Mean BASE entropy at token offset relative to '='

| offset | mean H | n |
|---:|---:|---:|
| -3 | 0.0478 | 2302 |
| -2 | 0.0020 | 2302 |
| -1 | 0.0191 | 2302 |
| +0 | 0.0004 | 2302 |
| +1 | 0.0259 | 2302 |
| +2 | 0.0015 | 2302 |
| +3 | 0.0081 | 2302 |
| +4 | 0.0860 | 2302 |
| +5 | 0.0467 | 2302 |
| +6 | 0.0232 | 2302 |
