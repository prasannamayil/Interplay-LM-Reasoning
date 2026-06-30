"""Widen-the-line on FineWeb-Edu (Exp B): do AR architectures fall on ONE line?

Unlike GSM-Infinity (which saturates -> no spread), FineWeb downstream accuracy grows
smoothly with training tokens, so multi-checkpoint eval gives a real SPREAD per arch.
The universality claim: across all AR archs (dense/gqa/moe/looped/tokenformer/...),
downstream capability is one shared function of upstream fit -- every (arch, ckpt)
point lands on a single curve. A different *objective* (diffusion, scored on the same
axis elsewhere) is the thing expected to leave the line.

Reads the in-training eval trajectories written by fineweb_run.sh:
  results/widen_line/fineweb_b/widen_b_<arch>/metrics.eval.jsonl        (downstream tasks)
  results/widen_line/fineweb_b/widen_b_<arch>/metrics.validation.jsonl  (upstream NLL on FineWeb val)
Each line = one checkpoint (keyed by global_step).

Outputs: a per-(arch, ckpt) table, a single global fit (R^2) of mean-downstream-accuracy
vs upstream avg log-prob/token across ALL archs+ckpts, and a scatter PNG (points colored
by arch, global fit overlaid). High R^2 with archs interleaved on the curve => the line.

Usage:
    python analyze/widen_line_fineweb.py [--exp_root ...] [--no_plot]
"""
import argparse
import glob
import json
import os
import re

ARCHS = ["dense", "gqa", "moe", "looped", "tokenformer"]
# downstream tasks evaluated in-training (see configs_fineweb/widen_400M_fineweb_b.yaml)
TASKS = ["hellaswag", "piqa", "winogrande", "arc_easy", "arc_challenge",
         "openbookqa", "lambada_openai"]


def _read_jsonl(path):
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def _task_acc(task_metrics):
    """Prefer acc_norm (lm-eval keys are like 'acc_norm,none' / 'acc,none')."""
    if not isinstance(task_metrics, dict):
        return None
    for pref in ("acc_norm,none", "acc,none", "acc_norm", "acc"):
        if pref in task_metrics:
            try:
                return float(task_metrics[pref])
            except (TypeError, ValueError):
                return None
    # last resort: first numeric value that isn't a stderr
    for k, v in task_metrics.items():
        if "stderr" not in k:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return None


def load_downstream(eval_path):
    """step -> {task: acc, '_mean': mean_over_present_tasks}."""
    out = {}
    for row in _read_jsonl(eval_path):
        step = row.get("global_step")
        if step is None:
            continue
        accs = {}
        for t in TASKS:
            a = _task_acc(row.get(t))
            if a is not None:
                accs[t] = a
        if accs:
            accs["_mean"] = sum(accs.values()) / len(accs)
            out[int(step)] = accs
    return out


def load_upstream(val_path):
    """step -> avg log-prob/token on the FineWeb val split (higher = better fit).

    eval_on_val stores summed loglikelihood (negative); 'nll_per_token' is the
    per-token avg log-prob. We pick the FineWeb source if present, else the first.
    """
    out = {}
    for row in _read_jsonl(val_path):
        step = row.get("global_step")
        if step is None:
            continue
        src = None
        for k in row:
            if "fineweb" in k.lower():
                src = k
                break
        if src is None:
            src = next((k for k, v in row.items() if isinstance(v, dict)), None)
        if src and isinstance(row[src], dict) and "nll_per_token" in row[src]:
            try:
                out[int(step)] = float(row[src]["nll_per_token"])
            except (TypeError, ValueError):
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", default="results/widen_line/fineweb_b")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    # points[arch] = list of (step, upstream_logprob, mean_acc, per_task_dict)
    points = {}
    for arch in ARCHS:
        adir = os.path.join(args.exp_root, f"widen_b_{arch}")
        down = load_downstream(os.path.join(adir, "metrics.eval.jsonl"))
        up = load_upstream(os.path.join(adir, "metrics.validation.jsonl"))
        rows = []
        for step in sorted(down):
            accs = down[step]
            rows.append((step, up.get(step), accs["_mean"], accs))
        if rows:
            points[arch] = rows

    if not points:
        print(f"No FineWeb eval metrics found yet under {args.exp_root}.")
        print("(Jobs may still be training to the first eval at step 2000.)")
        return

    print(f"\n=== (arch, ckpt) -> upstream logprob/tok, mean downstream acc ===")
    hdr = "arch".ljust(12) + "ckpt".rjust(7) + "up_lp/tok".rjust(11) + "meanAcc".rjust(9)
    print(hdr + "   " + "  ".join(t[:5] for t in TASKS))
    xs, ys = [], []
    for arch in points:
        for step, up, mean_acc, accs in points[arch]:
            ups = f"{up:11.4f}" if up is not None else "        n/a"
            per = "  ".join(f"{accs.get(t, float('nan')):.3f}" for t in TASKS)
            print(arch.ljust(12) + f"{step:7d}" + ups + f"{mean_acc:9.3f}" + "   " + per)
            if up is not None:
                xs.append(up)
                ys.append(mean_acc)

    # single global fit across ALL (arch, ckpt): mean downstream acc vs upstream logprob
    r2 = slope = intercept = float("nan")
    if len(xs) >= 2:
        n = len(xs)
        mx, my = sum(xs) / n, sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        if sxx > 0:
            slope = sxy / sxx
            intercept = my - slope * mx
            ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
            ss_tot = sum((y - my) ** 2 for y in ys)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"\nGlobal fit (meanAcc vs upstream logprob/tok) across all archs/ckpts: "
          f"acc = {slope:.4f}*lp + {intercept:.4f}   R^2 = {r2:.3f}   (n={len(xs)})")
    print("(High R^2 with archs interleaved => AR universality line holds on FineWeb.)")

    if args.no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa
        print(f"[plot skipped] matplotlib unavailable: {e}")
        return

    fig, ax = plt.subplots(figsize=(7.5, 6))
    for arch in points:
        ar = [(p[1], p[2]) for p in points[arch] if p[1] is not None]
        if ar:
            ax.plot([p[0] for p in ar], [p[1] for p in ar],
                    marker="o", ms=8, label=arch, alpha=0.85)
    if slope == slope and xs:
        lo, hi = min(xs), max(xs)
        ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept],
                "k--", alpha=0.6, label=f"global fit (R²={r2:.3f})")
    ax.set_xlabel("upstream avg log-prob / token on FineWeb val  (higher = better fit)")
    ax.set_ylabel("mean downstream accuracy")
    ax.set_title("Widen the line: downstream vs upstream across AR archs (FineWeb-Edu)")
    ax.legend()
    ax.grid(alpha=0.3)
    out = os.path.join(args.exp_root, "widen_line_fineweb.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nSaved plot -> {out}")


if __name__ == "__main__":
    main()
