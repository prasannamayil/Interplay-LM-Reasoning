
import json
import re

NOTEBOOK_PATH = "analyze/results.ipynb"

def update_notebook():
    with open(NOTEBOOK_PATH, "r") as f:
        nb = json.load(f)

    cells = nb["cells"]
    
    # 1. Update V2_RUNS definition to use ALL_RUNS_V3
    for cell in cells:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            if "V2_RUNS = [" in source and "grpo_id_v2" in source:
                new_source = source.replace(
                    'from process_results import (\n    get_all_runs_data, data_to_dataframe, summary_table,\n    PASS_K_VALUES, DIFFICULTY_GROUPS, DISPLAY_NAMES,\n)',
                    'from process_results import (\n    get_all_runs_data, data_to_dataframe, summary_table,\n    PASS_K_VALUES, DIFFICULTY_GROUPS, DISPLAY_NAMES, ALL_RUNS_V3\n)'
                )
                
                # Regex to replace the V2_RUNS list definition
                pattern = r'V2_RUNS = \[[^\]]*\]'
                replacement = 'V2_RUNS = ALL_RUNS_V3'
                new_source = re.sub(pattern, replacement, new_source, flags=re.DOTALL)
                
                cell["source"] = [s + "\n" if not s.endswith("\n") else s for s in new_source.splitlines()]
                print("Updated V2_RUNS definition.")

    # 2. Update plotting dictionaries
    # We'll look for specific keys to identify the plotting cells
    
    for cell in cells:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            
            # Update grpo_v2_runs (simple plot)
            if 'grpo_v2_runs = {' in source:
                new_source = re.sub(
                    r'grpo_v2_runs = \{[^}]*\}',
                    '''grpo_v2_runs = {
    "Base (skewed)": ["base_model_eval_skewed_pass128"],
    "GRPO-v3 (ID)": ["grpo_id_v3"],
    "GRPO-v3 (Edge)": ["grpo_edge_v3"],
    "GRPO-v3 (Hard)": ["grpo_hard_v3"],
    "GRPO-v3 (Mixed)": ["grpo_mixed_v3"],
}''',
                    source, flags=re.DOTALL
                )
                
                new_source = re.sub(
                    r'grpo_v2_colors = \{[^}]*\}',
                    '''grpo_v2_colors = {
    "Base (skewed)": "#888888",
    "GRPO-v3 (ID)": "#1f77b4",
    "GRPO-v3 (Edge)": "#ff7f0e",
    "GRPO-v3 (Hard)": "#d62728",
    "GRPO-v3 (Mixed)": "#2ca02c",
}''',
                    new_source, flags=re.DOTALL
                )
                cell["source"] = [s + "\n" if not s.endswith("\n") else s for s in new_source.splitlines()]
                print("Updated grpo_v2_runs plot.")

            # Update regimes dict (4x2 grid, now showing methods for edge/hard)
            if 'regimes = {' in source and '"ID (op=7-10)"' in source:
                new_regimes = '''regimes = {
    "ID (op=7-10)": {
        "Base": "base_model_eval_skewed_pass128",
        "GRPO-v3": "grpo_id_v3",
    },
    "Edge (op=11-14)": {
        "Base": "base_model_eval_skewed_pass128",
        "GRPO-v3": "grpo_edge_v3",
        "ClipCov": "grpo_clip_cov_edge_v3",
        "KLCov": "grpo_kl_cov_edge_v3",
        "EntCov": "grpo_ent_cov_edge_v3",
        "MGPO": "grpo_mgpo_edge_v3",
        "RUP": "grpo_rup_edge_v3",
    },
    "Hard (op=17-20)": {
        "Base": "base_model_eval_skewed_pass128",
        "GRPO-v3": "grpo_hard_v3",
        "ClipCov": "grpo_clip_cov_hard_v3",
        "KLCov": "grpo_kl_cov_hard_v3",
        "EntCov": "grpo_ent_cov_hard_v3",
        "MGPO": "grpo_mgpo_hard_v3",
        "RUP": "grpo_rup_hard_v3",
    },
    "Mixed (op=9-12)": {
        "Base": "base_model_eval_skewed_pass128",
        "GRPO-v3": "grpo_mixed_v3",
    },
}'''
                new_source = re.sub(r'regimes = \{[^}]+\}[^}]+\}[^}]+\}[^}]+\}', new_regimes, source, flags=re.DOTALL)
                cell["source"] = [s + "\n" if not s.endswith("\n") else s for s in new_source.splitlines()]
                print("Updated regimes plot configuration.")

            # Update bar chart lists
            if 'regime_grpo = [' in source:
                new_source = source.replace(
                    'regime_grpo = ["grpo_id_v2", "grpo_edge_v2", "grpo_hard_v2", "grpo_mixed_v2"]',
                    'regime_grpo = ["grpo_id_v3", "grpo_edge_v3", "grpo_hard_v3", "grpo_mixed_v3"]'
                )
                # Comment out old RUP lists as we don't have perfect alignment across all regimes for bar chart yet
                # Or repurpose them. Let's just update the main one for now.
                cell["source"] = [s + "\n" if not s.endswith("\n") else s for s in new_source.splitlines()]
                print("Updated bar chart lists.")

            # Update run_order for final plot
            if 'run_order = [' in source:
                new_source = re.sub(
                    r'run_order = \[[^\]]*\]',
                    'run_order = ALL_RUNS_V3',
                    source, flags=re.DOTALL
                )
                cell["source"] = [s + "\n" if not s.endswith("\n") else s for s in new_source.splitlines()]
                print("Updated run_order.")

    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(nb, f, indent=2)
    print(f"Successfully updated {NOTEBOOK_PATH}")

if __name__ == "__main__":
    update_notebook()
