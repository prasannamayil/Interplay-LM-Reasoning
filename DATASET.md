# Dataset overview — `composition` and `rl_finetune`

This document describes the synthetic math-reasoning dataset used in this
repo for both pretraining (next-token prediction) and RL finetuning. All
data lives under `data/` and is generated with the **GSM-Infinite**
framework (`gsm_infinite/`), which produces fully synthetic, programmatically
verifiable arithmetic word problems whose difficulty can be dialed
in continuously.

The same generator produces every split. Splits differ only in
**how many "operations" (`op`)** the underlying reasoning chain has,
**which solving mode** is asked of the model, and **how problems are
mixed**. There are no LLMs in the loop and no human-written text.

---

## 1. Where the data lives

```
data/
├── PRESET.json            # named sampling mixtures used by the trainer
├── dataset_info.json      # LLaMA-Factory style data registry
├── composition/           # pretraining / SFT-style data
│   ├── train/{2..10}/op{N}_<tokens>M.jsonl     # main pretraining shards
│   ├── val/op{N}-200.jsonl                     # 200 examples / op for dev
│   └── heldout/op{N}-50000.jsonl               # 50K examples / op (incl. op11, op12)
└── rl_finetune/
    └── train/
        ├── id/id_200k.jsonl         # op = 2..10            (in-distribution)
        ├── edge/edge_200k.jsonl     # op = 11..14           (just past training)
        ├── hard/hard_200k.jsonl     # op = 17..20           (much past training)
        ├── uniform/uniform_200k.jsonl  # op = 2..20 uniform
        └── mixed/mixed_200k.jsonl   # ~67k each from id+edge+hard
```

The pretraining corpus (`composition/train/`) is large — roughly
**4.8 B tokens of solution traces** spread across nine op buckets
(the `…M.jsonl` suffix is the token count in millions, e.g.
`op2_1111M.jsonl` ≈ 1.11 B tokens). Concretely:


| op  | file              | examples  | tokens  |
| --- | ----------------- | --------- | ------- |
| 2   | `op2_1111M.jsonl` | 2,222,222 | ~1.11 B |
| 3   | `op3_1067M.jsonl` | 2,135,223 | ~1.07 B |
| 4   | `op4_981M.jsonl`  | 1,963,300 | ~0.98 B |
| 5   | `op5_533M.jsonl`  | 1,066,634 | ~0.53 B |
| 6   | `op6_339M.jsonl`  | ~680 K    | ~0.34 B |
| 7   | `op7_230M.jsonl`  | ~460 K    | ~0.23 B |
| 8   | `op8_220M.jsonl`  | ~440 K    | ~0.22 B |
| 9   | `op9_166M.jsonl`  | ~330 K    | ~0.17 B |
| 10  | `op10_120M.jsonl` | ~242 K    | ~0.12 B |


(The token budget per op shrinks with op because each example gets
longer with more reasoning steps, so the same disk budget produces
fewer rows.)

The RL finetune sets are each exactly 200 K examples by construction
(see `scripts/download_rl_data.py` and `scripts/prepare_uniform_rl_data.py`).

---

## 2. Schema of a single example

Every line in every `.jsonl` file is the same JSON object:

```json
{
  "problem":  "<premises, one fact per sentence>",
  "question": "<single natural-language question>",
  "solution": "<symbolic step-by-step trace ending in 'Answer: N.'>",
  "op":       <int>,        // # of computation steps in the gold trace
  "id":       "<string>",   // generator seed/index, e.g. "17" or "11_p44"
  "template": "crazy_zootopia" | "teachers_in_school" | "movie_festival_awards",
  "mode":     "normalforward" | "forwardreverse",
  "length":   "zero_context",
  "d":        2 | 3         // depth of the underlying structure graph
}
```

### Field meanings

- `**problem**` — a paragraph of declarative facts. Each fact either sets
a variable to a number, or sets a variable to a (possibly multi-term)
expression over other variables. Order is randomized; the model has to
resolve the dependency DAG itself.
- `**question**` — one of the variables, asked as a natural-language
question.
- `**solution**` — the gold program: `Define <var> as <letter>; so <letter> = … = <number>.` repeated, ending with `Answer: <N>.` In
`forwardreverse` mode the trace contains an algebraic step that solves
for an unknown `x`.
- `**op**` — number of reasoning operations (≈ steps of the form
`Define … so X = …`) needed to reach the answer. This is the principal
difficulty knob. In this repo `op ∈ {2, …, 20}`.
- `**d**` — depth of the structure graph (number of layers of
abstraction). `d = 2` is "medium", `d = 3` is "hard" in the
GSM-Infinite paper. Pretraining shards mix both; the RL sets use
`d = 2`.
- `**length**` — how much distractor context is padded in. All training
data here is `"zero_context"` (no padding). The benchmark suite supports
thousands of irrelevant tokens of padding; we don't use that during
training.
- `**id**` — generator-internal id. A `_p<k>` suffix appears on
forward-reverse problems and indexes the variable that was made the
unknown.

### How the trainer feeds it to the model

`verl/dataset.py::compose_prompt` packs the row into

```
<question> {problem} {question} </question> <solution> {solution} </solution> <answer> {answer} </answer>
```

so the same JSONL drives both autoregressive pretraining
(prompt + solution + answer concatenated) and RL rollouts (prompt is
everything up to `<solution>`, the reward checks the final `Answer: …`
matches and, optionally, that the symbolic intermediate steps match too).

---

## 3. How the data is generated

The generator lives in `gsm_infinite/gsm-infinite/data/realistic/` and
implements the construction from the [Physics of LM Part 2 / iGSM
paper](https://arxiv.org/abs/2407.20311) plus the GSM-Infinite extensions
(`forward_generator.py`, `reverse_generator.py`,
`StructureGraphThree.py`, `DependencyGraph.py`,
`datagenerationworker.py`).

The pipeline per example is:

1. **Sample a structure graph.** A bipartite/tripartite layered graph is
  built with `d` layers (`d ∈ {2, 3}`); each layer is a level of
   abstraction (e.g. *category* → *subcategory* → *attribute*).
2. **Sample an instance graph.** The structure is grounded into concrete
  entities drawn from one of three template families (see §4) — e.g.
   "adult bear in Bundle Ranch" or "futuristic sci-fi movie in Verdi
   Movie Festival".
3. **Sample a dependency DAG over those instance variables.** Each node
  is either (a) a literal integer, or (b) an arithmetic combination of
   parent nodes via `+`, `-`, `*`, `sum of …`, `times the difference  between …`. The number of *non-literal* nodes is the `op` count.
4. **Render facts.** Each edge of the DAG becomes one English sentence
  ("The number of adult bear in Bundle Ranch equals the number of adult
   deer in Bundle Ranch."). Sentences are shuffled.
5. **Pick the question.**
  - `mode = "normalforward"` — pick a node, ask its value. The solution
   is a forward topological evaluation.
  - `mode = "forwardreverse"` — pick an *internal* node `x`, *withhold*
  its value, instead supply the value of a downstream node, and ask
  for `x`. The solution carries `x` symbolically through the chain
  and ends with a one-variable linear equation that the trace solves
  step by step (see the op = 13 example below).
6. **Solve symbolically.** A reference solver (`solver.py` /
  `DependencyGraph.py`) emits the gold reasoning trace token-for-token,
   so the answer is always exactly verifiable.
7. Arithmetic is done modulo a prime (`--mod`, default 23) — every
  intermediate value stays a small non-negative integer, which removes
   tokenization-of-large-numbers as a confound.

Generation is sharded across processes by `datagenerationworker.py`
(rejection sampling: the generator keeps drawing until the resulting
graph hits the requested op count exactly, per template).

---

## 4. The three "templates" (surface vocabulary)

A template is just the noun set used to instantiate the structure graph.
The arithmetic structure is identical across templates, but the surface
text changes:


| template                | layer-1 entity                                                   | layer-2 entity                                                           | layer-3 attribute                          |
| ----------------------- | ---------------------------------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------ |
| `crazy_zootopia`        | location (e.g. "Cedar Valley", "Bundle Ranch")                   | animal (e.g. "adult bear", "adult crow")                                 | "average number of newborn children per …" |
| `teachers_in_school`    | city (e.g. "Riverton City", "Ruby Bay")                          | school type (e.g. "public highschool", "regional medical school")        | "average number of teachers per …"         |
| `movie_festival_awards` | festival (e.g. "Taylor Movie Festival", "Festival de Clairmont") | movie genre (e.g. "upbeat metropolis comedy", "futuristic sci-fi movie") | "average number of nominations per …"      |


The `"context"` family of mixtures in `PRESET.json` controls the
template proportions (e.g. `"contextzoo_0.9zoo_0.1teacher"` = 90 %
zootopia, 10 % schools).

---

## 5. The two solving modes

Both modes coexist in the data. The split is roughly 50 / 50 in the
RL sets; the `composition/train` shards we ship for op ≤ 10 are
~100 % `normalforward`; `composition/heldout` for op = 11, 12 is
~50 / 50.

- `**normalforward`** — every variable in the problem has an explicit
numeric definition. The solution is just a forward topological
evaluation: define each variable, plug in numbers, propagate forward.
- `**forwardreverse**` — exactly one variable `x` is left unknown
(marked in the problem as *"... exists, and its number is greater
than 0."*) and a downstream node is given a numeric value. The trace
carries `x` symbolically forward, builds a linear equation
`a·x + b = c`, then solves it via a fixed sub-template
(*"Move all terms to one side: … Isolate the term with x: …
Divide both sides: … Solution: x = …"*). This subsumes the forward
mode as a strict superset of skills.

---

## 6. Splits and what they're used for


| split                       | files         | op range | purpose                                |
| --------------------------- | ------------- | -------- | -------------------------------------- |
| `composition/train`         | per-op shards | 2..10    | next-token pretraining / SFT           |
| `composition/val`           | 200 / op      | 2..10    | dev metric                             |
| `composition/heldout`       | 50 K / op     | 2..12    | held-out eval, including OOD ops 11–12 |
| `rl_finetune/train/id`      | 200 K         | 2..10    | RL with **in-distribution** difficulty |
| `rl_finetune/train/edge`    | 200 K         | 11..14   | RL just past the training horizon      |
| `rl_finetune/train/hard`    | 200 K         | 17..20   | RL well past the training horizon      |
| `rl_finetune/train/uniform` | 200 K         | 2..20    | RL with uniform op weighting           |
| `rl_finetune/train/mixed`   | 200 K         | 2..20    | RL with ~67K each from id, edge, hard  |


For the pretraining mixtures, `data/PRESET.json` defines named recipes
that the dataloader samples from on the fly. Examples:

- `composition-10B / op_level / id2-10_uniform` — uniform over op = 2..10
for a 10 B-token pretrain.
- `composition-10B / op_level / id2-10_0.5easy_0.3medium_0.2hard` —
skewed toward easier ops.
- `composition-10B / op_level_cpt / cpt0.2-uniform_0.8-9-12` — CPT
recipe: 20 % uniform op = 2..10 plus 80 % concentrated on op 9–12.
- `composition-10B / op_level_rl / op11-14_uniform`,
`op17-20_uniform` — the recipes feeding the `edge` / `hard` RL sets.
- `mixed-10B / mixed_level / 0.9zoo_op2-20+0.1teacher_op2` — joint
template + op mixture: 90 % zootopia uniform over op 2..20 plus 10 %
schools restricted to op = 2.

So the dataset is best viewed as **one synthetic distribution** (a
graph-grounded arithmetic word problem with two solving modes and three
surface vocabularies) sliced and re-weighted by `op`, `template`, and
`mode`.

---

## 7. Concrete examples across `op`

These are real samples taken from the shipped files (`crazy_zootopia`
or `movie_festival_awards`). Long premise paragraphs are shown verbatim;
the solution traces are also verbatim.

### `op = 2` — `composition/train/2/`, `normalforward`, `d = 3`

> **Problem.** The average number of newborn children per adult blue jay in
> Cedar Valley equals 5. The average number of newborn children per adult
> owl in Maple Creek equals the sum of the total number of adult animals
> in Cedar Valley, the number of adult blue jay in Cedar Valley, and the
> number of adult owl in Maple Creek. … *(13 facts, most distractors)* …
> The average number of newborn children per adult owl in Beverly Forest
> equals the average number of newborn children per adult crow in Beverly
> Forest.
>
> **Question.** How many adult blue jay does Cedar Valley have?
>
> **Solution.**
> Define average number of newborn children per adult crow in Beverly
> Forest as `t`; so `t = 2`.
> Define adult blue jay in Cedar Valley as `d`; so `d = t = 2`.
> **Answer: 2.**

Two reasoning steps, hence `op = 2`. Most of the paragraph is
distractor — the model has to find the two relevant edges in the DAG.

### `op = 5` — `composition/train/5/`, `normalforward`, `d = 3`

> **Problem.** *(22 facts about animals in Bundle Ranch, South Zoo,
> Hamilton Farm, Mayer Aquarium, Jefferson Circus)*
>
> **Question.** What is the total number of newborn animal children in
> Jefferson Circus?
>
> **Solution.**
> Define average number of newborn children per adult deer in Bundle
> Ranch as `e`; so `e = 6`.
> Define average number of newborn children per adult bear in Jefferson
> Circus as `P`; so `P = e = 6`.
> Define adult bear in Jefferson Circus as `V`; so `V = e = 6`.
> Define total number of newborn animal children in Jefferson Circus as
> `l`; so `l = V * P = 6 * 6 = 36`.
> **Answer: 36.**

Five steps, including a multiplication. Note how nodes are introduced
top-down even though the problem text was shuffled.

### `op = 8` — `composition/train/8/`, `normalforward`, `d = 3`

> **Question.** What is the average number of nominations per solemn
> period drama in Cinéma de Montreval?
>
> **Solution.**
> Define avg nominations per calm road movie in Festival de Clairmont
> as `U`; so `U = 21`.
> Define solemn period drama in Rêves de Belleville as `S`; so `S = U = 21`.
> Define avg nominations per calm road movie in Cinéma de Montreval as
> `H`; so `H = S = 21`.
> Define avg nominations per futuristic sci-fi movie in Festival de
> Clairmont as `K`; `e = H + U = 21 + 21 = 42`; so `K = 2 * e = 2 * 42 = 84`.
> Define avg nominations per solemn period drama in Cinéma de Montreval
> as `n`; `A = K = 84`; so `n = 19 + A = 19 + 84 = 103`.
> **Answer: 103.**

Eight ops, with named intermediate "scratch" letters (`e`, `A`) for
sums.

### `op = 13` — `rl_finetune/train/edge/`, `forwardreverse`, `d = 2`

> **Problem.** The total number of movies in Festival de Saint-Rivage
> equals 36. The number of calm road movie in Festival de Saint-Rivage
> equals 3 times the sum of the number of intense detective thriller in
> Festival de Saint-Rivage and the number of solemn period drama in
> Festival de Saint-Rivage. The number of solemn period drama in
> Festival de Saint-Rivage equals the number of intense detective
> thriller in Festival de Saint-Rivage. **The number of intense detective
> thriller in Festival de Saint-Rivage exists, and its number is greater
> than 0.** The number of futuristic sci-fi movie in Festival de
> Saint-Rivage equals 3 plus the number of intense detective thriller in
> Festival de Saint-Rivage. The number of upbeat metropolis comedy in
> Festival de Saint-Rivage equals the number of futuristic sci-fi movie
> in Festival de Saint-Rivage.
>
> **Question.** How many intense detective thriller does Festival de
> Saint-Rivage have?
>
> **Solution.**
> The question is difficult, so we use equations to solve it. Define
> intense detective thriller in Festival de Saint-Rivage as `x`; we don't
> know its value yet but will find it out later.
> Define solemn period drama as `w`; so `w = x`.
> Define futuristic sci-fi movie as `C`; `d = x`; so `C = 3 + d = x + 3`.
> Define calm road movie as `s`; `E = x + w = 2x`; so `s = 3 * E = 6x`.
> Define upbeat metropolis comedy as `i`; so `i = C = x + 3`.
> Define total number of movies as `G`; `q = x + s = 7x`; `r = q + w = 8x`; `v = r + C = 9x + 3`; so `G = v + i = 10x + 6`.
> We know `G = 36`, so we have `10x + 6 = 36`. Simplifying: `10x + 6 = 36`. Move all terms to one side: `10x − 30 = 0`. Isolate the term:
> `10x = 30`. Divide both sides by 10: `x = 30 / 10`. Solution: `x = 3`.
> **Answer: 3.**

This is the canonical `forwardreverse` style. The `exists, and its number is greater than 0` sentence is the generator's way of marking
the unknown. The trace propagates `x` symbolically and ends with the
equation-solving sub-template.

### `op = 20` — `rl_finetune/train/hard/`, `forwardreverse`, `d = 2`

> **Question.** How many solemn period drama does Festival de Clairmont
> have?
>
> **Solution (excerpt).**
> Define solemn period drama in Festival de Clairmont as `x`; we don't
> know its value yet but will find it out later. Define upbeat
> metropolis comedy as `r`; so `r = x`. Define futuristic sci-fi movie
> as `C`; so `C = x`. Define total movies in Festival de Clairmont as
> `m`; `W = x + r = 2x`; so `m = W + C = 3x`. Define upbeat metropolis
> comedy in Rêves de Belleville as `p`; so `p = m − x = 2x`. … *(eight
> more `Define` steps carrying `x` across two festivals)* … Define total
> movies in Festival Lumière de Valmont as `b`; `O = L + n = 5x`; so
> `b = O + H = 10x + 3`. We know `b = 43`, so we have `10x + 3 = 43`.
> Solving: `x = 4`. **Answer: 4.**

Twenty operations, four festivals, ten symbolic carries before the
linear equation closes. This is the kind of trace the `hard` RL set
trains the model to produce.

---

## 8. How complex is the graph at high `op`? (empirical)

A natural worry is: maybe `op = 20` is just a 20-link chain `x → a → b → c → … → answer`. It is **not**. The dependency graph at high `op` is a
wide-ish layered DAG with branching, multi-parent merges, and hub nodes
that feed many descendants.

The numbers below were measured by parsing the gold solution traces of
200 random examples per `op` (named variables = the `Define …` calls;
fan-in / fan-out counted on the named variables only; longest chain =
deepest topo order):


| op  | mode mix     | named vars | longest chain | distinct locations | fan-in (mean / max) | fan-out (mean / max) | arithmetic ops in trace |
| --- | ------------ | ---------- | ------------- | ------------------ | ------------------- | -------------------- | ----------------------- |
| 2   | nf only      | 2.0        | 2             | 1.4                | 0.5 / 1             | 1.0 / 1              | 0.0                     |
| 5   | nf only      | 3.6        | 3.4           | 2.0                | 0.9 / 2             | 1.2 / 2              | 2.8                     |
| 8   | nf only      | 5.1        | 4.4           | 2.1                | 1.1 / 4             | 1.4 / 4              | 5.8                     |
| 10  | nf only      | 5.9        | 5.0           | 2.3                | 1.3 / 4             | 1.5 / 4              | 8.3                     |
| 11  | ~50/50 nf+fr | 6.1        | 4.9           | 2.1                | 1.5 / 5             | 1.8 / 7              | 15.2                    |
| 12  | ~50/50 nf+fr | 6.1        | 5.3           | 2.2                | 1.6 / 5             | 1.9 / 7              | 19.0                    |
| 13  | ~50/50 nf+fr | 6.5        | 5.8           | 2.4                | 1.7 / 5             | 1.9 / 9              | 21.8                    |
| 14  | ~50/50 nf+fr | 7.6        | 6.2           | 2.8                | 1.6 / 5             | 1.8 / 10             | 21.0                    |
| 17  | ~50/50 nf+fr | 9.8        | 7.4           | 3.3                | 1.4 / 5             | 1.6 / 11             | 22.2                    |
| 20  | ~50/50 nf+fr | 11.5       | 8.0           | 3.4                | 1.4 / 5             | 1.6 / 12             | 24.4                    |


(`nf` = `normalforward`, `fr` = `forwardreverse`.)

What this says:

- **It is not a chain.** Even at `op = 8` you already see fan-in 4 and
fan-out 4 — nodes that combine four parents, and nodes that feed four
downstream consumers. The graph is shaped like a 2-D grid (one row
per *attribute*, one column per *location/festival*) plus
cross-aggregation edges.
- **Width grows with `op`, depth grows slower.** Going from op = 10 to
op = 20, the longest dependency chain only roughly doubles (5.0 → 8.0)
while the number of distinct named variables roughly doubles (5.9 →
11.5) **and** distinct entities (locations / festivals) grows from
~2 to ~3.4. So harder problems mostly *widen* the graph by adding more
entities and more attributes that interact, rather than just lengthening
one chain.
- **There are real "hub" nodes.** The maximum fan-out climbs steadily
from 4 (at `op = 10`) to **12** (at `op = 20`). In `forwardreverse`
the unknown `x` is one of those hubs — every later node reads `x` —
and the per-entity "total" (e.g. *total number of movies in Festival
X*) is another, because it is summed into the next-layer-up
aggregation as well as referenced by other festivals.
- **Multi-parent merges are common.** Mean fan-in stabilizes around
1.4–1.7 for op ≥ 8, but the maximum is consistently 4–5: aggregation
nodes (`sum of …, …, … and …`) routinely combine 3–5 parents in one
arithmetic step.
- `**forwardreverse` roughly doubles the arithmetic-token cost per
named variable.** Compare op = 10 (≈ 8 arithmetic operators in the
trace, ~6 named vars → 1.4 ops/var) with op = 14 (≈ 21 ops, ~7.6 vars →
2.8 ops/var). The doubling is exactly the symbolic carry: every
`Define` step has to write *both* the symbolic expression in `x` *and*
the numeric form, then the trace ends with a 5-step linear-equation
solve.
- **Op count counts every micro-step.** The repo's `op` is the number
of *atomic* operations in the trace (each binary `+`, `−`, `*` plus
each step of the equation-solving sub-template), not the number of
named variables. That is why op = 20 has only ~11 named variables but
~24 arithmetic operators plus the 5 equation-solving steps.

Concretely, the op = 20 example shown in §7 has this dependency graph
(arrows point parent → child):

```
                       x  (unknown, solemn drama @ Clairmont)
                      /|\
                     / | \  (fan-out 4: r, C, m, p)
                    r  C  m──p          ──────►  L
                    |  |  |\ |\         festival 2
                    V  |  | \| \        ──────►  T
                       |  |  n  V       ──────►  h ─► H
                       └──┘             festival 3
                                        ──────►  b = L+n+H = 43  (anchor)
```

That is 12 named values across **3 festivals** with **4 movie-genre
attributes per festival**, an aggregation layer (`total movies in …`),
and one cross-festival anchor that closes a single linear equation in
`x`. Both the fan-out from `x` and the fan-in into the anchor `b` exceed
2, so a forward greedy "next variable I can compute" strategy will not
work — the model has to maintain a multi-symbolic state (every later
variable is a polynomial in `x`) until the anchor closes the system.

So: high-`op` examples are **structurally non-trivial** (wide layered
DAGs with hubs and merges), not just longer chains.

---

## 9. Things worth knowing for downstream use

- **All examples are exactly verifiable.** A reward function can either
(a) extract the integer after `Answer:` and compare, or (b) replay
the symbolic trace step by step and check each `<lhs> = <rhs>` line
against the gold (`verl/reward_fn.py::compute_score_with_step_process`
does the latter for process-verified `pass@K`).
- **Numbers stay small.** Arithmetic is computed mod 23 by default, so
every intermediate value is a single- or double-digit non-negative
integer. This isolates *reasoning-graph traversal* from
*long-arithmetic*.
- **The op axis is the difficulty axis.** `op = 2` problems live in a
single sentence's worth of relevant facts; `op = 20` problems chain
twenty defines and a linear solve. The "edge" (11–14) and "hard"
(17–20) RL sets are deliberately *outside* the pretraining op range
(2–10), which is the central experimental knob in this repo.
- **Templates are interchangeable for arithmetic, not for surface
features.** Crossing template boundaries (e.g. train on
`crazy_zootopia`, evaluate on `teachers_in_school`) tests whether
the model has learned *graph-traversal arithmetic* vs. *templated
surface patterns*. The `context_level_rl` mixtures in `PRESET.json`
exist precisely to study this.
- `**length: zero_context` everywhere.** This codebase trains and
evaluates *only* on the no-distractor variant. The GSM-Infinite
generator can pad arbitrary amounts of irrelevant context for
long-context evaluation, but those shards are not part of `data/`.

