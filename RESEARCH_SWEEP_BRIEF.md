# Research Sweep Brief — for the next (research) agent

> **Purpose.** A fresh agent should run a thorough literature + landscape sweep to
> inform `RESEARCH_PROPOSAL.md` before we commit experiments. Read
> `RESEARCH_PROPOSAL.md` and `DLLM_PROJECT_GUIDE.md` first for context.
>
> **PI steer (important):** prioritize **realistic distributions**, not toys. The
> dream result is a **domain-level dissociation** — diffusion generalizing better
> on one real domain (e.g. math / structured reasoning), AR better on another
> (e.g. comprehension / free-form language) — with the autoregressive
> architectures themselves staying on a single universality line. Treat "diffusion
> better everywhere" as a non-goal. Methodology is deliberately *open*; the sweep
> should *recommend*, not assume.

## The question in one line
Is there empirical or theoretical reason to expect **diffusion (and/or
multi-token-prediction) language models to constitute a distinct function class**
that generalizes *differently from autoregressive models on realistic data* — and
where, specifically, would that show up?

## What to find (sweep dimensions)

1. **Beyond-toy evidence of objective-driven generalization differences.**
   - Where do masked/diffusion LMs or MTP beat AR (or vice versa) on *non-synthetic*
     tasks? Math, code, reasoning, comprehension, infilling, planning, retrieval.
   - Distinguish *toy* (path-star, star-graph, synthetic reversal) from *realistic*
     (actual benchmarks / corpora). We have the toy precedents; we need the real ones
     or the absence thereof.

2. **Domain-specific inductive biases.**
   - Any evidence that diffusion/non-AR objectives have domain-specific strengths
     (e.g. global-constraint / bidirectional reasoning → math; left-to-right
     fluency → language)? What predicts which domain favors which objective?

3. **Reversal & factorization curse.**
   - State of the art on the reversal curse and its mitigation by non-AR
     objectives (MLM, permutation LM, diffusion). Key refs (verify, find newer):
     Berglund et al. (reversal curse); Kitouni et al. (factorization curse); any
     diffusion-specific reversal results. Does the mitigation require training on
     the *joint* (full-sequence denoising) vs directional SFT?

4. **Planning / look-before-you-leap.**
   - Bachmann & Nagarajan (pitfalls of next-token prediction / path-star) and
     follow-ups. Does teacherless / MTP / diffusion provably or empirically help?
     Any realistic planning-flavored benchmark analogues?

5. **Multi-token prediction (MTP).**
   - Gloeckle et al., DeepSeek-V3 MTP, and any generalization (not just speed)
     results. Is MTP a meaningful intermediate point on an AR→diffusion dial, or
     too close to AR to leave the line?

6. **Cross-objective measurement methodology (critical).**
   - How have people fairly compared generalization/likelihood across AR and
     diffusion? Any-order / exact likelihood (our DUEL), ELBO tightness,
     bits-per-byte comparisons, matched-compute vs matched-loss vs matched-ID
     anchoring. What is the defensible axis for an "on the line / off the line"
     claim across objectives? (This directly resolves the `[OPEN]` section of the
     proposal.)

7. **Universality-of-generalization prior art + the contrast set.**
   - "LLMs on the line" and related (scaling-law universality, architecture-vs-data
     for generalization). What would constitute a credible *break* of universality
     in the eyes of that literature?
   - We want to show the line is *broad*: not just attention variants but
     **Tokenformer** (token-parameter attention) and **Mixture-of-Experts** should
     also lie on it — as the contrast to MTP/diffusion leaving it. Find any prior
     evidence on Tokenformer / MoE generalization-vs-dense (do they stay on the
     line?), and practical notes on training small Tokenformer/MoE for matched
     comparisons.

8. **Training & finetuning diffusion *efficiently* (first-class, not just
   generalization).**
   - This sweep is **not only about whether diffusion generalizes differently — it
     is equally about how to train/finetune diffusion properly and with *less
     compute*** to get the same results. Survey: best recipes for masked/block
     diffusion finetuning and A2D conversion; objective weighting / noise schedules;
     packing & length-grouping; EOS/padding-supervision pitfalls; convergence speed
     vs AR; any "diffusion training is N× AR, here's how to close it" results.
   - Goal: a recommended compute-efficient diffusion finetuning recipe, and the
     fairest **compute-matched** comparison protocol. Internal context:
     `scripts/TRAINING_SPEEDUP_STATUS.md`, GSM log "Bug 6" / no-pack throughput.

9. **Decoding-time artifacts that confound math evals.**
   - We have a concrete one: diffusion's GSM weakness is largely a decoding-order
     symbolic-coordination artifact (`results/DIFFUSION_GSM_VARNAME_PROBLEM.md`).
     Survey decoding strategies for diffusion (remasking order, step count,
     constrained decoding, any-order planning) and how others control for them when
     comparing diffusion vs AR on reasoning. We must not measure decoding when we
     mean to measure the function class.

10. **Candidate datasets.**
   - Concrete realistic datasets per target domain (math: which beyond
     OpenMathInstruct/GSM; comprehension: RACE/NLI/QA; code; multi-hop/retrieval)
     with the structural property that would favor one objective. Note licensing /
     availability / size.

## Deliverables
- An **annotated bibliography** (claim, evidence quality, toy-vs-real, relevance).
- A **ranked shortlist of realistic distributions** likely to show a diffusion-vs-AR
  dissociation, with the hypothesized mechanism per distribution.
- A **measurement recommendation** to close the `[OPEN]` axis question.
- A list of **strongest counter-arguments / null results** we must engage with
  (e.g. diffusion just being a worse LM; any-order likelihoods being
  incomparable).
- Concrete **edits proposed** to `RESEARCH_PROPOSAL.md` (which experiments to
  promote/drop, which datasets to commit to).

## Constraints / notes for the agent
- Verify citations; the codebase has two relevant PDFs at root:
  - `2512.07783v1.pdf` = the Interplay paper (the *other*, already-published project).
  - `2603.01367v2.pdf` = **DUEL: Exact Likelihood for Masked Diffusion via
    Deterministic Unmasking** (Turok, De Sa, Kuleshov, Cornell, arXiv:2603.01367,
    Mar 2026). **Directly relevant to dimension 6 (measurement).** It is the
    exact-likelihood method already wired into our diffusion eval. Key claims to
    build on: ELBO is loose *and* computed under the train (not test-time)
    distribution; DUEL gives proper test-time perplexity; under proper likelihood
    the AR–MDM gap shrinks up to 32% in-domain / 82% zero-shot; and **oracle search
    over unmasking orderings lets MDMs surpass AR** (AG News 36.47 vs 52.11 ppl) —
    i.e. the likelihood is strongly order-dependent. This confirms our internal
    finding that `prob_margin` DUEL is order-optimistic and `left_to_right` is the
    fair cross-objective currency (`results/LINE_A_NLL_FINDINGS.md`). Read it
    closely for the measurement recommendation.
- Keep the PI's realistic-distribution priority central; flag when a promising
  result is toy-only.
- Output as markdown suitable to drop into the repo (e.g.
  `results/research_sweep_findings.md`).
</content>
