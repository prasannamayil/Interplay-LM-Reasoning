#!/usr/bin/env python3
"""Sanity check on the re-verification cache. Should reproduce key numbers
from the existing reports."""
import pickle
import math
from pathlib import Path

PROJECT_ROOT = Path("/fast/pmayilvahanan/Interplay-LM-Reasoning")
CACHE = PROJECT_ROOT / "results" / "_reverify_cache.pkl"

with open(CACHE, "rb") as f:
    R = pickle.load(f)

def fmt(x):
    if x is None: return "  n/a"
    if isinstance(x, float):
        if math.isnan(x): return "  nan"
        return f"{x:+.3f}"
    return str(x)

print("\nDISCOVERED (run, step) PAIRS:")
for k in sorted(R.keys()):
    print(f"  {k}")

print("\n# Reproduce Phase 1c headline cell: grpo_edge_v4 @ 388, op17")
cell = R[("grpo_edge_v4", 388)]
op17 = cell["rollouts"][17]
print(f"  n_rollouts: {op17['n_rollouts']}, n_prompts: {op17['n_prompts']}")
print(f"  outcome: {op17['outcome_mean']:.3f}, process: {op17['process_mean']:.3f}")
print(f"  T5 pooled: {fmt(op17['pooled']['mean_kl_policy_ref'])}")
print(f"  T5 within-prompt: {fmt(op17['within_prompt']['mean_kl_policy_ref'])}")
print(f"  T1 pooled: {fmt(op17['pooled']['mean_logprob_policy'])}")
print(f"  T1 within-prompt: {fmt(op17['within_prompt']['mean_logprob_policy'])}")
print(f"  T4 pooled: {fmt(op17['pooled']['mean_entropy_policy'])}")
print(f"  T4 within-prompt: {fmt(op17['within_prompt']['mean_entropy_policy'])}")
print(f"  outcome_reward pooled: {fmt(op17['pooled']['outcome_reward'])}")
print(f"  outcome_reward within-prompt: {fmt(op17['within_prompt']['outcome_reward'])}")
print(f"  fractions: aw={op17['frac_all_wrong']:.2f} mx={op17['frac_mixed']:.2f} ac={op17['frac_all_correct']:.2f}")

print("\n# Reproduce per-Define-step rho: grpo_edge_v4 @ 388, op17")
ds17 = cell["define_steps"][17]
print(f"  n_all: {ds17['all_n']}, gg_n: {ds17['gg_n']}, gg frac: {ds17['gg_n']/max(1,ds17['all_n']):.3f}")
print(f"  mean step_correct (all): {ds17['all_mean_correct']:.3f}")
print(f"  mean step_correct (gg): {ds17['gg_mean_correct']:.3f}")
print(f"  logp pooled (all): {fmt(ds17['all_mean_logprob_policy_step_pooled'])}")
print(f"  logp pooled (gg):  {fmt(ds17['gg_mean_logprob_policy_step_pooled'])}")
print(f"  logp wp (gg):      {fmt(ds17['gg_mean_logprob_policy_step_wp'])}")
print(f"  logp wr (gg):      {fmt(ds17['gg_mean_logprob_policy_step_wr'])}")
print(f"  KL pooled (gg):    {fmt(ds17['gg_mean_kl_step_pooled'])}")
print(f"  KL wp (gg):        {fmt(ds17['gg_mean_kl_step_wp'])}")
print(f"  KL wr (gg):        {fmt(ds17['gg_mean_kl_step_wr'])}")
print(f"  H pooled (gg):     {fmt(ds17['gg_mean_entropy_step_pooled'])}")
print(f"  H wp (gg):         {fmt(ds17['gg_mean_entropy_step_wp'])}")
print(f"  H wr (gg):         {fmt(ds17['gg_mean_entropy_step_wr'])}")

print("\n# Reproduce BASE_v4 @ 0, op17 (Finding C: per-step entropy)")
ds17_b = R[("BASE_v4", 0)]["define_steps"][17]
print(f"  H wp (gg):  {fmt(ds17_b['gg_mean_entropy_step_wp'])}")
print(f"  H wr (gg):  {fmt(ds17_b['gg_mean_entropy_step_wr'])}")

print("\n# Reproduce grpo_uniform_v4 @ 388, op17 - hardness checking")
op17u = R[("grpo_uniform_v4", 388)]["rollouts"][17]
print(f"  outcome: {op17u['outcome_mean']:.3f}, process: {op17u['process_mean']:.3f}")
print(f"  T5 pooled: {fmt(op17u['pooled']['mean_kl_policy_ref'])}")
print(f"  T5 within-prompt: {fmt(op17u['within_prompt']['mean_kl_policy_ref'])}")
print(f"  fractions: aw={op17u['frac_all_wrong']:.2f} mx={op17u['frac_mixed']:.2f} ac={op17u['frac_all_correct']:.2f}")

print("\n# Phase 1d sanity: grpo_edge_v4 @ 388, op20 prefix_gold_2 mixed")
p1d_e388 = R[("grpo_edge_v4", 388)]["phase1d"]
op20p = p1d_e388[20]
print(f"  prefix_gold_2 wp mixed (early): {fmt(op20p['prefix_gold_2_early_wp_mixed'])} (n={op20p['prefix_gold_2_early_wp_mixed_n']})")
print(f"  prefix_gold_2 wp all-wrong:     {fmt(op20p['prefix_gold_2_early_wp_all_wrong'])} (n={op20p['prefix_gold_2_early_wp_all_wrong_n']})")
print(f"  prefix_gold_2 mean R:           {fmt(op20p['prefix_gold_2_early_mean_R'])}")

print("\n# Structural zero-variance check: grpo_edge_v4 @ 388 hard ops")
for op in [17, 18, 19, 20]:
    rs = R[("grpo_edge_v4", 388)]["rollouts"][op]
    print(f"  op{op}: aw={rs['frac_all_wrong']:.2f} mx={rs['frac_mixed']:.2f} ac={rs['frac_all_correct']:.2f}")
