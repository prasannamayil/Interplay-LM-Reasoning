#!/usr/bin/env python3
"""Fix the results.ipynb notebook to work with available v3 runs."""

import json

NOTEBOOK_PATH = "results.ipynb"

# The v3 runs that actually exist with complete evals
AVAILABLE_V3_RUNS = [
    "base_model_eval_skewed_pass128",
    "grpo_edge_v3",
    "grpo_clip_cov_edge_v3",
    "grpo_kl_cov_edge_v3",
    "grpo_ent_cov_edge_v3",
    "grpo_mgpo_edge_v3",
    "grpo_rup_edge_v3",
    "grpo_mgpo_hard_v3",
    "grpo_rup_hard_v3",
]


def fix_notebook():
    with open(NOTEBOOK_PATH, "r") as f:
        nb = json.load(f)

    cells = nb["cells"]
    
    # Find and update v3-related cells
    for i, cell in enumerate(cells):
        if cell["cell_type"] != "code":
            continue
            
        source = "".join(cell["source"])
        
        # Cell 20: The loading cell - this should work fine with the updated process_results.py
        # Just make sure it reloads properly
        if "V2_RUNS = ALL_RUNS_V3" in source and "get_all_runs_data" in source:
            new_source = """# ── Reload with v3 runs ───────────────────────────────────────────────────────
import importlib
import process_results
importlib.reload(process_results)
from process_results import (
    get_all_runs_data, data_to_dataframe, summary_table,
    PASS_K_VALUES, DIFFICULTY_GROUPS, DISPLAY_NAMES, ALL_RUNS_V3
)

V3_RUNS = ALL_RUNS_V3
v3_data = get_all_runs_data(V3_RUNS)
print(f"Loaded {len(v3_data)} v3 runs: {list(v3_data.keys())}")

summary_table(V3_RUNS)
"""
            cell["source"] = [line + "\n" for line in new_source.strip().split("\n")]
            print(f"Updated cell {i}: v3 loading cell")
        
        # Cell 24: The bar chart cell - update variable names
        if "run_order = ALL_RUNS_V3" in source and "v2_data" in source:
            new_source = """# ── OOD ID / mid / hard: pass@128 for v3 runs ─────────────────────────────────
# Plots the three difficulty groups (ID, OOD-mid, OOD-hard) for every run in v3_data.

run_order = ALL_RUNS_V3
runs_done = [r for r in run_order if r in v3_data]
if not runs_done:
    runs_done = list(v3_data.keys())

groups = ["ID (op=2-10)", "OOD-mid (op=11-14)", "OOD-hard (op=15-20)"]
labels = [DISPLAY_NAMES.get(r, r) for r in runs_done]

fig, ax = plt.subplots(figsize=(max(10, len(runs_done) * 1.1), 5))
x = np.arange(len(runs_done))
width = 0.25

for i, group in enumerate(groups):
    vals = [v3_data[r][group][128] * 100 for r in runs_done]
    ax.bar(x + (i - 1) * width, vals, width, label=group)

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha="right")
ax.set_ylabel("pass@128 Accuracy (%)")
ax.set_title("V3 Results: ID vs OOD-mid vs OOD-hard")
ax.legend(loc="upper right", framealpha=0.9)
ax.grid(True, alpha=0.2, axis="y", linestyle="--")
fig.tight_layout()
fig.savefig("figures/v3_ood_id_mid_hard.png", dpi=200, bbox_inches="tight")
plt.show()
"""
            cell["source"] = [line + "\n" for line in new_source.strip().split("\n")]
            print(f"Updated cell {i}: bar chart cell")

    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(nb, f, indent=2)
    print(f"\nSuccessfully updated {NOTEBOOK_PATH}")


if __name__ == "__main__":
    fix_notebook()
