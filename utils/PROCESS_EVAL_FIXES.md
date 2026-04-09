# Process Evaluation Fixes for Diffusion LLM Variable-Name Collapse

## Background

The process scorer in `utils/solution_dependency_graph.py` parses GSM-Infinity solutions into dependency graphs and compares them against gold. It was designed for AR models that produce clean variable names. Diffusion LLMs (MDLM, BD3LM) generate solutions with two systematic defects:

1. **Variable-name collapse**: All steps reuse the same letter (usually `t`) instead of distinct variables.
2. **Phantom variables**: Random single letters (`s`, `w`, `a`, `e`) appear in expressions but were never defined.

These defects cause the original scorer to reject solutions where the arithmetic and entity references are entirely correct.

---

## Impact Summary

All numbers are pass@k computed over 128 samples per example, 200 examples per op, on the 160M MDLM 1-epoch checkpoint.

| op | pass@k | old process+outcome | outcome-only | fixed process+outcome |
| -- | ------ | ------------------- | ------------ | --------------------- |
| 2  | @1     | 0.538               | 0.995        | 0.995                 |
| 2  | @128   | 0.990               | 0.995        | 0.995                 |
| 5  | @1     | 0.032               | 0.965        | 0.665                 |
| 5  | @128   | 0.070               | 0.970        | 0.845                 |
| 10 | @1     | 0.000               | 0.455        | 0.004                 |
| 10 | @128   | 0.020               | 0.825        | 0.045                 |

Gold self-compare passes 200/200 on all ops. No regressions.

---

## Fix 1: Variable Reuse Tolerance

### Problem

The parser maintained `var_to_param: Dict[str, str]` -- a single mapping from variable letter to parameter name. When the model reuses `t` for every step, each new `Define ... as t;` overwrites the previous mapping.

**Gold** (unique variables -- works fine):
```
Define public highschool in Riverton City as s; so s = 4.
Define regional medical school in Riverton City as T; B = s = 4; so T = 3 + B = 7.
Define total number of schools in Riverton City as v; so v = s + T = 4 + 7 = 11.
```
Parser state after each step:
- Step 1: `var_to_param = {'s': 'public highschool in Riverton City'}`
- Step 2: `var_to_param = {'s': 'public highschool...', 'T': 'regional medical school...'}`
- Step 3: `v = s + T` resolves both deps correctly.

**Model** (variable collapse -- fails):
```
Define public highschool in Riverton City as t; so t = 4.
Define regional medical school in Riverton City as t; t = a = 4; so t = 3 + t = 7.
Define total number of schools in Riverton City as t; so t = t + t = 7 + 4 = 11.
```
Parser state (old):
- Step 1: `var_to_param = {'t': 'public highschool...'}`
- Step 2: `var_to_param = {'t': 'regional medical school...'}` **overwrites!**
- Step 3: `t = t + t` -- both `t` resolve to `'regional medical school...'`. Dependency on `'public highschool...'` is **lost**.

Result: `dependency_mismatches` on the total step. Process check fails.

### Fix

Changed `var_to_param: Dict[str, str]` to `var_to_params: Dict[str, List[str]]`. Each `Define ... as t;` **appends** to the list. When resolving dependencies, all parameter names mapped to a letter are included.

```python
# OLD (line 259, 366):
self.var_to_param: Dict[str, str] = {}
self.var_to_param[current_var] = step.parameter_name

# NEW:
self.var_to_params: Dict[str, List[str]] = {}
self.var_to_params.setdefault(current_var, []).append(step.parameter_name)
```

```python
# OLD _collect_dependencies:
if token in self.var_to_param:
    deps.add(self.var_to_param[token])

# NEW:
if token in self.var_to_params:
    deps.update(self.var_to_params[token])
```

After fix, step 3's `t = t + t` resolves to both `'public highschool...'` and `'regional medical school...'`. Matches gold.

### Impact

op=2: 53.8% -> 74.0%. op=5: 7.0% -> 23.5%.

---

## Fix 2: Phantom Variable Value Extraction

### Problem

The model generates expressions like `t = s = 3; so t = 3 * w = 9` where `s` and `w` are phantom letters that don't appear in `var_to_params`. The parser's `ASSIGNMENT_RE` regex captures `target='t', expr='s = 3'` -- the target is `t`, so `s` is never stored as a key in any values dict.

**Gold**: `R; n = x = 3; so R = 3 * n = 9`
- `n` is intermediate, `x` maps to step 1 -> dependency resolved.

**Model**: `t; t = s = 3; so t = 3 * w = 9`
- `s` and `w` are phantom. Never defined. No dependency resolved.

### Fix

Before processing each assignment, extract sub-assignments from within the expression:

```python
for sub_var, sub_val in re.findall(
    r"([A-Za-z])\s*=\s*([-+]?\d+(?:\.\d+)?)", expr
):
    if sub_var != target_var:
        intermediate_values[sub_var] = float(sub_val)
        self.variable_values[sub_var] = float(sub_val)
```

This stores `s=3` and `w=3` in `variable_values`. Then the second-pass phantom resolver (also new) matches them to prior steps by value:

```python
if unresolved_letters:
    for letter in unresolved_letters:
        val = intermediate_values.get(letter) or self.variable_values.get(letter)
        if val is None:
            continue
        for var, params in self.var_to_params.items():
            if var in self.variable_values and abs(self.variable_values[var] - val) < 1e-9:
                step.dependencies.update(params)
```

### Impact

op=5: 23.5% -> 40.0%.

---

## Fix 3: Intermediate Dependency Propagation

### Problem

In equation-style solutions, intermediate phantom variables accumulate dependencies on prior steps via `x` references, but these dependencies are stored in `intermediate_deps` (keyed by the phantom letter) and never propagated to `step.dependencies` (which only gets updated when `target_var == current_var`).

**Gold**: `Define total as A; so A = x + p = x + 2*x = 3*x. We know A = 9...`
- `A` is current_var. `A = x + p` assigns deps directly to the step.

**Model**: `Define total as V; so Y = x + B = 2*x* x = 3*x. We know Q = 9...`
- `V` is current_var but never appears on the LHS of any assignment.
- `Y = x + B` assigns deps to `intermediate_deps['Y']`, which is never merged into `step.dependencies`.

### Fix

After processing all assignments in a step, propagate all intermediate dependencies:

```python
for _int_var, _int_deps in intermediate_deps.items():
    step.dependencies.update(_int_deps)
```

This ensures that even when the model uses phantom letters as assignment targets, the dependency chain flows through to the step.

### Impact

op=2: 74.0% -> 99.5%. op=5: 40.0% -> 68.0%.

---

## Fix 4: "We know" Value Extraction + Intermediate Fallback

### Problem

For equation-style steps where `current_var` never appears on the LHS, the step's value stays `None`. The gold parser gets the value from `A = 9` (where `A` is current_var), but the model writes `Q = 9` (where `Q` is a phantom).

**Gold**: `Define total as A; ... We know A = 9, so we have 3*x = 9 ...`
- `A = 9` matches current_var -> `step.value = 9`.

**Model**: `Define total as V; ... We know Q = 9, we have 3*x = 9 ...`
- `V` never assigned -> `step.value = None` -> value_mismatch with gold's 9.

### Fix

Two fallbacks for when `current_var` has no direct assignment:

```python
# 1. Extract from "We know X = N" pattern
if last_value is None:
    we_know = re.search(
        r"[Ww]e\s+know\s+[A-Za-z]\s*=\s*([-+]?\d+(?:\.\d+)?)",
        step.raw_body,
    )
    if we_know:
        last_value = float(we_know.group(1))

# 2. Inherit from last intermediate value
if last_value is None and intermediate_values:
    last_value = list(intermediate_values.values())[-1]
```

### Impact

Included in the jump to 99.5% / 68.0% above.

---

## Remaining Failures

After all fixes, remaining failures at op=5 (64/200):
- **35 dependency-only**: Phantom variable chains too garbled for value-matching to resolve.
- **19 value+dependency**: Equation-style solutions where symbolic evaluation fails through garbled intermediaries.
- **3 missing nodes**: Model omits entire computation steps.

At op=10 (199/200 fail):
- **64 missing nodes**: Model generates fewer steps than gold (e.g. 4 of 5 required). This is a genuine model capacity issue at 160M, not a scorer artifact.
- Of 94 outcome-correct examples, only 52 have the correct step count.

---

## File Changes

- `utils/solution_dependency_graph.py`: All four fixes in `SolutionParser._process_step()` and `_collect_dependencies()`.
- `dllm/examples/gsm_infinity/eval_pass128.py`: Removed `prompt_text[:200]` truncation in detail JSON (line 398).
