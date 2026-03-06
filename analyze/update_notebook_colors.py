
import json
import re

NOTEBOOK_PATH = "analyze/results.ipynb"

def update_colors():
    with open(NOTEBOOK_PATH, "r") as f:
        nb = json.load(f)

    cells = nb["cells"]
    
    for cell in cells:
        if cell["cell_type"] == "code":
            source = "".join(cell["source"])
            
            # Find the cell with method_colors definition
            if 'method_colors = {' in source and '"Base": "#888888"' in source:
                
                # Replace the entire style definition block
                new_style_block = '''method_colors = {
    "Base": "#888888",
    "GRPO-v3": "#1f77b4",
    "ClipCov": "#ff7f0e",
    "KLCov": "#2ca02c",
    "EntCov": "#d62728",
    "MGPO": "#9467bd",
    "RUP": "#8c564b",
}
method_markers = {
    "Base": "s", "GRPO-v3": "o", "ClipCov": "v", "KLCov": "^", 
    "EntCov": "<", "MGPO": ">", "RUP": "D"
}
method_ls = {
    "Base": "--", "GRPO-v3": "-", "ClipCov": "-", "KLCov": "-", 
    "EntCov": "-", "MGPO": "-", "RUP": "-"
}'''
                
                # Replace the old definitions using regex
                # We replace from method_colors down to method_ls definition
                pattern = r'method_colors = \{[^}]+\}\nmethod_markers = \{[^}]+\}\nmethod_ls = \{[^}]+\}'
                
                # If regex is too complex/brittle, we can try to just replace the specific text block if it matches known state
                # But regex allows for whitespace variance. Let's try to be precise.
                
                # First, check if we can just find the start and replace until the end of the problematic section
                # But actually, the code you pasted has a trailing comma inside the regimes dict which might be a syntax error in your paste or valid python tuple
                # "Mixed (op=9-12)": {\n        "Base": "base_model_eval_skewed_pass128",\n        "GRPO-v3": "grpo_mixed_v3",\n    },\n},\n}
                
                # Let's replace the whole cell content if possible, or just the colors part.
                # Since I know the structure of what I want to inject:
                
                if 'method_colors = {' in source:
                    # Construct regex to match the old style dicts
                    # Old:
                    # method_colors = {
                    #     "Base": "#888888",
                    #     "GRPO": "#1f77b4",
                    #     "GRPO+RUP": "#ff7f0e",
                    #     "GRPO+RUP-5x": "#d62728",
                    # }
                    # method_markers = {"Base": "s", "GRPO": "o", "GRPO+RUP": "^", "GRPO+RUP-5x": "D"}
                    # method_ls = {"Base": "--", "GRPO": "-", "GRPO+RUP": "-", "GRPO+RUP-5x": "-"}
                    
                    # I'll just replace the method_colors block and assume the others follow
                    # Actually, better to replace the specific keys if they exist, or append new ones.
                    # But overwriting is cleaner.
                    
                    # Regex to find method_colors = { ... }
                    colors_pattern = r'method_colors = \{.*?\n\}'
                    markers_pattern = r'method_markers = \{.*?\n\}' # Single line definition in your snippet
                    ls_pattern = r'method_ls = \{.*?\n\}' # Single line definition in your snippet
                    
                    # Wait, in the source snippet provided in the prompt:
                    # method_markers = {"Base": "s", "GRPO": "o", "GRPO+RUP": "^", "GRPO+RUP-5x": "D"}
                    
                    # Let's try to replace the whole block of 3 dicts
                    
                    # Identify the range of lines to replace
                    lines = source.splitlines()
                    start_idx = -1
                    end_idx = -1
                    
                    for i, line in enumerate(lines):
                        if 'method_colors = {' in line:
                            start_idx = i
                        if 'method_ls = {' in line:
                            end_idx = i
                    
                    if start_idx != -1 and end_idx != -1:
                        # Construct new lines
                        new_lines = lines[:start_idx] + new_style_block.splitlines() + lines[end_idx+1:]
                        
                        # Also need to fix the loop:
                        # ax.plot(..., marker=method_markers[method_label], ..., color=method_colors[method_label], ...)
                        # This part is fine as long as keys match
                        
                        # Also need to fix the title
                        # fig.suptitle("GRPO vs GRPO+RUP vs GRPO+RUP-5x by Data Regime", ...)
                        # -> "GRPO vs Methods (v3) by Data Regime"
                        
                        final_source = "\n".join(new_lines)
                        final_source = final_source.replace(
                            'fig.suptitle("GRPO vs GRPO+RUP vs GRPO+RUP-5x by Data Regime"',
                            'fig.suptitle("GRPO vs Methods (v3) by Data Regime"'
                        )
                        
                        cell["source"] = [s + "\n" if not s.endswith("\n") else s for s in final_source.splitlines()]
                        print("Updated style dictionaries and title.")

    with open(NOTEBOOK_PATH, "w") as f:
        json.dump(nb, f, indent=2)
    print(f"Successfully updated {NOTEBOOK_PATH}")

if __name__ == "__main__":
    update_colors()
