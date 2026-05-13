import math
import re
from typing import Any, Dict, Optional, Union

from utils.solution_dependency_graph import SolutionParser


def extract_answer(text: str) -> str:
    """Extract answer from generated text.
    Priority:
      1) Between <answer>...</answer>
      2) If <answer> exists but no closing tag, take text after it up to a tag start '<' or newline.
    """
    # 1) Proper tags
    m = re.search(r"<answer>(.*?)</answer>", text, flags=re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()
    # 2) Open tag only
    pos = text.lower().rfind("<answer>")
    if pos != -1:
        tail = text[pos + len("<answer>"):]
        # find the numeric text up to the next tag or at the end, if not found return ""
        m2 = re.search(r"(.*?)(<|\n|$)", tail, flags=re.DOTALL)
        if m2:
            return m2.group(1).strip()
    return ""




def compute_score(solution_str: str, ground_truth: Dict[str,str], data_source=None, extra_info=None) -> bool:
    gold_answer = ground_truth['answer']
    answer = extract_answer(solution_str)
    ret_score = answer.strip().rstrip(".") == gold_answer.strip().rstrip(".")  

    return ret_score

def parse_graph(solution_str):
    global solution_parser
    if solution_parser is None:
        solution_parser = SolutionParser()
    parsed = solution_parser.parse(solution_str)
    graph = solution_parser.build_graph(parsed)
    setattr(graph, "steps", len(parsed.steps))
    setattr(graph, "parsed_solution", parsed)
    return graph

solution_parser = None

def _normalise_answer(text: Optional[str]) -> str:
    if not isinstance(text, str):
        return ""
    return text.strip().rstrip(".")


def _compute_step_process_reward(
    solution_str: str,
    ground_truth: Union[Dict, str],
    extra_info: Optional[Dict[str, Any]],
    *,
    answer_weight: float,
    step_weight: float,
    value_tolerance: float,
    zero_on_process_mismatch: bool = False,
) -> Dict[str, Any]:
    extra_info_dict = extra_info if isinstance(extra_info, dict) else {}

    gold_answer: Optional[str]
    gold_solution: Optional[str]
    if isinstance(ground_truth, dict):
        gold_answer = ground_truth.get("answer")
        gold_solution = ground_truth.get("solution") or ground_truth.get("gold_solution")
    else:
        gold_answer = ground_truth
        gold_solution = None

    if not gold_answer:     
        gold_answer = extra_info_dict.get("gold_answer")
    if not gold_solution:
        gold_solution = extra_info_dict.get("gold_solution")

    predicted_answer = extract_answer(solution_str)
    outcome_reward = 0.0
    if gold_answer:
        outcome_reward = 1.0 if _normalise_answer(predicted_answer) == _normalise_answer(gold_answer) else 0.0

    process_reward = 0.0
    gold_graph = None
    process_mismatch = False
    if isinstance(gold_solution, str) and gold_solution.strip():
        try:
            gold_graph = parse_graph(gold_solution)
        except Exception:
            gold_graph = None

    if gold_graph is not None:
        try:
            pred_graph = parse_graph(solution_str)
        except Exception:
            pred_graph = None

        if pred_graph is not None:
            report = gold_graph.compare(pred_graph, value_tolerance=value_tolerance)
            total_nodes = max(len(gold_graph.nodes), 1)
            structural_penalty = (
                len(report["value_mismatches"])
                + len(report["dependency_mismatches"])
                + len(report["missing_in_pred"])
            )
            # Extra predicted nodes are allowed and do not affect ProcessAcc according to paper A.4
            # structural_penalty += min(len(report["extra_in_pred"]), total_nodes)
            if report["answer_mismatch"] is not None:
                structural_penalty += 1
            process_reward = max(0.0, 1.0 - structural_penalty / total_nodes)
            process_mismatch = structural_penalty > 0
        else:
            process_reward = 0.0
            process_mismatch = True
    else:
        # Fall back to outcome reward if process supervision is unavailable.
        process_reward = outcome_reward
        process_mismatch = False

    if extra_info_dict is not None:
        extra_info_dict["outcome_reward"] = outcome_reward
        extra_info_dict["process_reward"] = process_reward
        extra_info_dict["process_reward_has_gold"] = gold_graph is not None
        if zero_on_process_mismatch:
            extra_info_dict["process_zeroed_due_to_mismatch"] = False

    total_weight = answer_weight + step_weight
    if total_weight <= 0:
        answer_weight = step_weight = 0.5
        total_weight = 1.0

    answer_component = (answer_weight / total_weight) * outcome_reward
    step_component = (step_weight / total_weight) * process_reward
    reward = max(0.0, min(1.0, answer_component + step_component))
    if outcome_reward < 1.0:
        reward = 0.0
    zeroed_due_to_mismatch = False
    if (
        zero_on_process_mismatch
        and gold_graph is not None
        and outcome_reward >= 1.0
        and process_mismatch
    ):
        reward = 0.0
        zeroed_due_to_mismatch = True
    if extra_info_dict is not None and zero_on_process_mismatch:
        extra_info_dict["process_zeroed_due_to_mismatch"] = zeroed_due_to_mismatch

    return {
        "score": reward,
        "outcome_reward": outcome_reward,
        "process_reward": process_reward,
        "process_reward_has_gold": gold_graph is not None,
        "process_zeroed_due_to_mismatch": zeroed_due_to_mismatch,
    }


def compute_score_with_step_process(
    solution_str: str,
    ground_truth: Union[Dict, str],
    data_source: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    *,
    answer_weight: float = 0.2,
    step_weight: float = 0.8,
    value_tolerance: float = 1e-6,
    zero_on_process_mismatch: bool = False,
) -> Dict[str, Any]:
    """
    Blend answer-level accuracy with a process reward derived from dependency graphs.

    Args:
        solution_str: Model generated solution text.
        ground_truth: Ground-truth payload; may be either a string answer or a dict containing
            keys like ``answer`` and ``solution``.
        data_source: Identifier for the example (unused, but kept for compatibility).
        extra_info: Mutable metadata dict that can be enriched with reward diagnostics.
        answer_weight: Relative weight assigned to the outcome reward.
        step_weight: Relative weight assigned to the process reward.
        value_tolerance: Tolerance for numeric comparisons during graph matching.
        zero_on_process_mismatch: If True, zero the blended reward when the answer is correct but the process mismatches.

    Returns:
        A dictionary containing the blended ``score`` alongside outcome/process diagnostics.
    """

    return _compute_step_process_reward(
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        answer_weight=answer_weight,
        step_weight=step_weight,
        value_tolerance=value_tolerance,
        zero_on_process_mismatch=zero_on_process_mismatch,
    )


def compute_score_with_step_process_batched(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    *,
    answer_weight: float = 0.2,
    step_weight: float = 0.8,
    value_tolerance: float = 1e-6,
    zero_on_process_mismatch: bool = False,
):
    """
    Batched variant of ``compute_score_with_step_process`` compatible with ``BatchRewardManager``.
    """

    _ = data_sources  # kept for API symmetry with other reward fns
    results: list[Dict[str, Any]] = []
    for solution_str, ground_truth, extra_info in zip(
        solution_strs, ground_truths, extra_infos, strict=True
    ):
        results.append(
            _compute_step_process_reward(
                solution_str=solution_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                answer_weight=answer_weight,
                step_weight=step_weight,
                value_tolerance=value_tolerance,
                zero_on_process_mismatch=zero_on_process_mismatch,
            )
        )
    return results


def compute_score_with_step_process_strict(
    solution_str: str,
    ground_truth: Union[Dict, str],
    data_source: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    *,
    answer_weight: float = 0.2,
    step_weight: float = 0.8,
    value_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """
    Variant of ``compute_score_with_step_process`` that zeros the reward if the process mismatches despite a correct answer.
    """

    _ = data_source  # kept for API compatibility
    return _compute_step_process_reward(
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        answer_weight=answer_weight,
        step_weight=step_weight,
        value_tolerance=value_tolerance,
        zero_on_process_mismatch=True,
    )


def compute_score_with_step_process_strict_batched(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    *,
    answer_weight: float = 0.2,
    step_weight: float = 0.8,
    value_tolerance: float = 1e-6,
):
    """
    Batched variant of ``compute_score_with_step_process_strict``.
    """

    return compute_score_with_step_process_batched(
        data_sources=data_sources,
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
        answer_weight=answer_weight,
        step_weight=step_weight,
        value_tolerance=value_tolerance,
        zero_on_process_mismatch=True,
    )


# ---------------------------------------------------------------------------
# Proposal A / B reward function: process-acc as the headline score.
#
# Returns the *raw* graph-comparison process_reward in [0, 1] as the score,
# WITHOUT the outcome gate that ``_compute_step_process_reward`` applies.
# This means rollouts whose final integer is wrong can still receive >0
# credit if they recover the correct dependency graph nodes/values.
#
# pass@K of this score on val data is the "process-acc pass@K" used in
# Proposal A to detect graph capture independent of answer correctness.
#
# If the env var ``PROPOSAL_B_DUMP_DIR`` is set, each rollout's
# structural breakdown is appended to a per-process JSONL sidecar file
# in that directory.  Cheap (~100 bytes/rollout) and gives Proposal B
# (per-rollout structural histograms) for free in the same eval pass.
# ---------------------------------------------------------------------------

def _structural_score(
    solution_str: str,
    ground_truth: Union[Dict, str],
    extra_info: Optional[Dict[str, Any]],
    value_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    extra = extra_info if isinstance(extra_info, dict) else {}

    if isinstance(ground_truth, dict):
        gold_answer = ground_truth.get("answer")
        gold_solution = (
            ground_truth.get("solution")
            or ground_truth.get("gold_solution")
        )
    else:
        gold_answer = ground_truth
        gold_solution = None
    if not gold_answer:
        gold_answer = extra.get("gold_answer")
    if not gold_solution:
        gold_solution = extra.get("gold_solution")

    pred_answer = extract_answer(solution_str)
    outcome_reward = 0.0
    if gold_answer:
        outcome_reward = (
            1.0
            if _normalise_answer(pred_answer) == _normalise_answer(gold_answer)
            else 0.0
        )

    process_reward = 0.0
    n_gold = 0
    n_pred = 0
    n_value_mismatch = 0
    n_dep_mismatch = 0
    n_missing = 0
    n_extra_in_pred = 0
    answer_mismatch = False
    parsed_ok = False
    has_gold_graph = False

    if isinstance(gold_solution, str) and gold_solution.strip():
        try:
            gold_graph = parse_graph(gold_solution)
        except Exception:
            gold_graph = None
        if gold_graph is not None:
            has_gold_graph = True
            n_gold = max(len(gold_graph.nodes), 1)
            try:
                pred_graph = parse_graph(solution_str)
            except Exception:
                pred_graph = None
            if pred_graph is not None:
                parsed_ok = True
                n_pred = len(pred_graph.nodes)
                report = gold_graph.compare(pred_graph, value_tolerance=value_tolerance)
                n_value_mismatch = len(report["value_mismatches"])
                n_dep_mismatch = len(report["dependency_mismatches"])
                n_missing = len(report["missing_in_pred"])
                n_extra_in_pred = len(report["extra_in_pred"])
                answer_mismatch = report["answer_mismatch"] is not None
                penalty = (
                    n_value_mismatch
                    + n_dep_mismatch
                    + n_missing
                    + (1 if answer_mismatch else 0)
                )
                process_reward = max(0.0, 1.0 - penalty / n_gold)

    breakdown = {
        "outcome_reward": outcome_reward,
        "process_reward": process_reward,
        "has_gold_graph": has_gold_graph,
        "parsed_ok": parsed_ok,
        "n_gold_nodes": n_gold,
        "n_pred_nodes": n_pred,
        "n_value_mismatch": n_value_mismatch,
        "n_dep_mismatch": n_dep_mismatch,
        "n_missing_in_pred": n_missing,
        "n_extra_in_pred": n_extra_in_pred,
        "answer_mismatch_flag": answer_mismatch,
    }
    if isinstance(extra_info, dict):
        extra_info.update(breakdown)
    return breakdown


def _maybe_dump_proposal_b(
    breakdown: Dict[str, Any],
    solution_str: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
) -> None:
    """Append per-rollout structural info + Phase-1 fields to a sidecar JSONL.

    Enabled by env var PROPOSAL_B_DUMP_DIR (path to write to).
    Phase-1 fields (op, example_id, predicted_answer, length_chars,
    solution_str_truncated) are added to enable post-hoc self-consistency
    and log-prob analyses without re-running rollouts.
    """
    import os, json
    out_dir = os.environ.get("PROPOSAL_B_DUMP_DIR")
    if not out_dir:
        return
    try:
        os.makedirs(out_dir, exist_ok=True)
        record = dict(breakdown)
        if isinstance(extra_info, dict):
            for key in ("op", "example_id", "template", "mode", "index"):
                if key in extra_info and key not in record:
                    record[key] = extra_info[key]
        if isinstance(solution_str, str):
            record["length_chars"] = len(solution_str)
            record["predicted_answer"] = extract_answer(solution_str)
            # Truncate to keep disk usage bounded; covers ~all gold rollouts.
            cap = int(os.environ.get("PROPOSAL_B_TEXT_CHAR_CAP", "2000"))
            record["solution_str_truncated"] = solution_str[:cap]
        path = os.path.join(out_dir, f"rollouts.{os.getpid()}.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def compute_score_process_only(
    solution_str: str,
    ground_truth: Union[Dict, str],
    data_source: Optional[str] = None,
    extra_info: Optional[Dict[str, Any]] = None,
    *,
    value_tolerance: float = 1e-6,
) -> Dict[str, Any]:
    """Headline score = process_reward (no outcome gate).

    pass@K of this score is the "process-acc" used in Proposal A.
    """
    _ = data_source
    breakdown = _structural_score(
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        value_tolerance=value_tolerance,
    )
    _maybe_dump_proposal_b(breakdown, solution_str=solution_str, extra_info=extra_info)
    return {"score": breakdown["process_reward"], **breakdown}


def compute_score_process_only_batched(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    *,
    value_tolerance: float = 1e-6,
):
    _ = data_sources
    return [
        compute_score_process_only(
            solution_str=s,
            ground_truth=g,
            data_source=None,
            extra_info=ei,
            value_tolerance=value_tolerance,
        )
        for s, g, ei in zip(solution_strs, ground_truths, extra_infos, strict=True)
    ]


# ---------------------------------------------------------------------------
# Phase 1e: step-level sibling-consensus reward.
#
# Replaces the gold dependency graph used by ``compute_score_process_only``
# with a model-internal proxy: for each rollout, parse the intermediate
# ``Define <param> as <var>; ... <var> = <value>`` lines, and for each
# such (var_name, value) compute the fraction of K-1 sibling rollouts of
# the SAME prompt that emitted the SAME (var_name, value) pair (i.e.
# ``cons_nc`` from `phase1e_consensus_findings.md`). The per-rollout score
# is the mean of these per-step consensus values, so it lies in [0, 1] and
# fills the same slot as ``process_reward``.
#
# Critical assumption: the batched reward function receives ALL rollouts
# of a GRPO group together (verl's BatchRewardManager passes the full
# batch). We rely on ``extra_info["example_id"]`` (set by
# ``verl/dataset.py::CustomRLHFDataset``) to group sibling rollouts. If
# ``example_id`` is missing we fall back to the prompt position fingerprint
# (the dataset always sets it for GSM-Infinity).
#
# Variants returned:
#   compute_score_consensus_only        -- score = mean cons_nc over
#                                          gold-grounded Define steps in
#                                          the rollout. Direct deployable
#                                          analogue of process_reward.
#   compute_score_consensus_outcome     -- score = outcome_reward *
#                                          (1 + gamma * mean_cons_nc),
#                                          renormalized to [0, 1]. The
#                                          per-token loss-mass-redistribution
#                                          framing of `proposed_phase1e_
#                                          training.md` §2.1.
#   compute_score_consensus_blend       -- score = (1-alpha) * outcome +
#                                          alpha * mean_cons_nc, no gate.
#                                          Closest to the dense-process
#                                          recipe (which is alpha=1, but
#                                          with gold instead of cons).
# ---------------------------------------------------------------------------

_VARNAME_LINE_RE = re.compile(
    r"Define\s+(?P<param>.+?)\s+(?:as|a)\s+(?:[A-Za-z]+\s+)*(?P<var>[A-Za-z]);"
    r"(?P<body>[^.]*?)\.",
    re.IGNORECASE | re.DOTALL,
)


def _extract_define_pairs(solution_str: str):
    """Return a list of ``(var_name, value)`` pairs for each Define step.

    Uses the same parser as ``_structural_score`` so the cons_nc proxy and
    the gold-graph reference are computed against the same parse. ``value``
    is a normalized float (or ``None`` if the step did not yield a numeric
    value); ``var_name`` is the parameter name (the text between
    ``Define`` and ``as``).
    """
    if not isinstance(solution_str, str) or not solution_str:
        return []
    try:
        parser = SolutionParser()
        parsed = parser.parse(solution_str)
    except Exception:
        return []
    pairs = []
    for step in parsed.steps:
        # ``parameter_name`` is the cleaned Define noun phrase
        # (matches `define_steps.jsonl::var_name` in compute_phase1c.py).
        var_name = (step.parameter_name or "").strip()
        if not var_name:
            continue
        value = step.value
        if value is None:
            pairs.append((var_name, None))
            continue
        try:
            value = float(value)
        except (TypeError, ValueError):
            value = None
        pairs.append((var_name, value))
    return pairs


def _value_token(v) -> Optional[str]:
    """Canonical hashable token for a numeric value (so 4.0 == 4)."""
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    if abs(f - round(f)) < 1e-9:
        return f"{int(round(f))}"
    return f"{f:.6f}"


def _group_key(extra_info, fallback_index: int):
    """Best-effort sibling-group key.

    Prefers ``example_id`` (set by CustomRLHFDataset from the dataset's
    ``id`` field); falls back to the index-mod-K-by-K-rollouts behaviour
    by returning ``None`` (caller treats every rollout as its own group,
    so cons_nc collapses to NaN -> 0 score).
    """
    if isinstance(extra_info, dict):
        eid = extra_info.get("example_id")
        if eid is not None:
            op = extra_info.get("op")
            return (str(op), str(eid))
    return None


def _compute_cons_nc_per_rollout(
    pairs_per_rollout,
    group_key_per_rollout,
):
    """For each rollout, produce a list of cons_nc scores aligned with its
    Define steps.

    cons_nc[var_name, value] = fraction of K-1 sibling rollouts (same
    group_key) that defined ``var_name`` with the same numeric value,
    over the count of siblings that defined ``var_name`` at all. Steps
    where no other sibling defined ``var_name`` -> NaN (excluded from
    the per-rollout mean).
    """
    n = len(pairs_per_rollout)
    by_group: Dict[Any, list] = {}
    for i, gk in enumerate(group_key_per_rollout):
        if gk is None:
            continue
        by_group.setdefault(gk, []).append(i)

    sibling_namemap: Dict[Any, list] = {}
    for gk, idxs in by_group.items():
        nm_per_rollout = []
        for i in idxs:
            namemap: Dict[str, Optional[str]] = {}
            for var_name, value in pairs_per_rollout[i]:
                if not var_name:
                    continue
                token = _value_token(value)
                namemap.setdefault(var_name, token)
            nm_per_rollout.append(namemap)
        sibling_namemap[gk] = nm_per_rollout

    out: list = []
    for i in range(n):
        gk = group_key_per_rollout[i]
        if gk is None or len(by_group.get(gk, [])) < 2:
            out.append([])
            continue
        idxs = by_group[gk]
        my_pos = idxs.index(i)
        nms = sibling_namemap[gk]
        per_step_scores = []
        for var_name, value in pairs_per_rollout[i]:
            if not var_name:
                continue
            tok = _value_token(value)
            n_def = 0
            n_match = 0
            for j_pos, j in enumerate(idxs):
                if j_pos == my_pos:
                    continue
                other_tok = nms[j_pos].get(var_name)
                if other_tok is None:
                    continue
                n_def += 1
                if tok is not None and other_tok == tok:
                    n_match += 1
            if n_def == 0 or tok is None:
                per_step_scores.append(float("nan"))
            else:
                per_step_scores.append(n_match / n_def)
        out.append(per_step_scores)
    return out


def _consensus_breakdown(
    solution_strs,
    ground_truths,
    extra_infos,
    *,
    value_tolerance: float,
):
    """Run ``_structural_score`` per rollout and ``cons_nc`` aggregation
    over the batch. Returns one dict per rollout containing both the gold
    diagnostics (``outcome_reward``, ``process_reward``, etc.) and the
    new ``consensus_reward`` field (mean cons_nc over Define steps with
    a defined sibling pool).
    """
    breakdowns = []
    pairs_per_rollout = []
    group_keys = []
    for i, (s, g, ei) in enumerate(
        zip(solution_strs, ground_truths, extra_infos, strict=True)
    ):
        bd = _structural_score(
            solution_str=s,
            ground_truth=g,
            extra_info=ei,
            value_tolerance=value_tolerance,
        )
        breakdowns.append(bd)
        pairs_per_rollout.append(_extract_define_pairs(s))
        group_keys.append(_group_key(ei, i))

    cons_per_rollout = _compute_cons_nc_per_rollout(
        pairs_per_rollout, group_keys
    )

    for i, bd in enumerate(breakdowns):
        scores = cons_per_rollout[i]
        valid = [v for v in scores if v == v]  # filter NaN
        n_def = len(pairs_per_rollout[i])
        if not valid:
            mean = 0.0
            cons_has_signal = False
        else:
            mean = sum(valid) / len(valid)
            cons_has_signal = True
        bd["consensus_reward"] = float(max(0.0, min(1.0, mean)))
        bd["consensus_n_steps_total"] = n_def
        bd["consensus_n_steps_valid"] = len(valid)
        bd["consensus_has_signal"] = cons_has_signal
        bd["consensus_group_size"] = (
            len([k for k in group_keys if k == group_keys[i]])
            if group_keys[i] is not None else 1
        )
    return breakdowns


def compute_score_consensus_only_batched(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    *,
    value_tolerance: float = 1e-6,
):
    """Headline score = mean ``cons_nc`` over Define steps with a defined
    sibling pool. Drop-in replacement for ``compute_score_process_only`` —
    same shape, same dataset hooks, but the dense ``[0, 1]`` signal comes
    from sibling consensus instead of the gold dependency graph.
    """
    _ = data_sources
    breakdowns = _consensus_breakdown(
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
        value_tolerance=value_tolerance,
    )
    results = []
    for s, ei, bd in zip(solution_strs, extra_infos, breakdowns, strict=True):
        out = {"score": bd["consensus_reward"], **bd}
        _maybe_dump_proposal_b(bd, solution_str=s, extra_info=ei)
        results.append(out)
    return results


def compute_score_consensus_outcome_batched(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    *,
    value_tolerance: float = 1e-6,
    gamma: float = 0.5,
):
    """Score = ``outcome_reward * (1 + gamma * mean_cons_nc)`` clipped to
    [0, 1]. Matches `proposed_phase1e_training.md` §2.1's "shaped reward"
    framing — keeps the binary outcome gate so the shaper only
    redistributes credit among correct rollouts.
    """
    _ = data_sources
    breakdowns = _consensus_breakdown(
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
        value_tolerance=value_tolerance,
    )
    results = []
    for s, ei, bd in zip(solution_strs, extra_infos, breakdowns, strict=True):
        out_rwd = bd.get("outcome_reward", 0.0)
        cons = bd["consensus_reward"]
        score = out_rwd * (1.0 + gamma * cons)
        score = max(0.0, min(1.0, score))
        bd["score_gamma"] = gamma
        out = {"score": score, **bd}
        _maybe_dump_proposal_b(bd, solution_str=s, extra_info=ei)
        results.append(out)
    return results


def compute_score_consensus_blend_batched(
    data_sources,
    solution_strs,
    ground_truths,
    extra_infos,
    *,
    value_tolerance: float = 1e-6,
    alpha: float = 0.5,
):
    """Score = ``(1 - alpha) * outcome_reward + alpha * mean_cons_nc``,
    no outcome gate. Closest to the dense-process recipe but using
    sibling consensus instead of gold.
    """
    _ = data_sources
    breakdowns = _consensus_breakdown(
        solution_strs=solution_strs,
        ground_truths=ground_truths,
        extra_infos=extra_infos,
        value_tolerance=value_tolerance,
    )
    a = max(0.0, min(1.0, alpha))
    results = []
    for s, ei, bd in zip(solution_strs, extra_infos, breakdowns, strict=True):
        out_rwd = bd.get("outcome_reward", 0.0)
        cons = bd["consensus_reward"]
        score = (1.0 - a) * out_rwd + a * cons
        score = max(0.0, min(1.0, score))
        bd["score_alpha"] = a
        out = {"score": score, **bd}
        _maybe_dump_proposal_b(bd, solution_str=s, extra_info=ei)
        results.append(out)
    return results
