"""Compare apps.widen architectures on the GSM-Infinity ID->OOD axis (Exp 3).

Reads each arch's pass@1 eval metrics (results/widen_line/exp_gsm/eval/<arch>/metrics.jsonl,
produced by exp_run.sh) and reports the universality-line view:
  - per-op pass@1 table,
  - ID (ops 2-10) vs OOD (ops 12-20) mean pass@1 per arch,
  - a scatter (OOD vs ID) + per-op curves PNG.

"On the line" = archs cluster along a single ID->OOD relationship. Usage:
    python analyze/widen_line_compare.py [--exp_root results/widen_line/exp_gsm] [--no_plot]
"""
import argparse
import json
import os
import re
from glob import glob

ARCHS = ["dense", "gqa", "moe", "looped", "tokenformer"]
ID_OPS = [2, 4, 6, 8, 10]
OOD_OPS = [12, 14, 16, 18, 20]


def load_passk(metrics_path, k=1):
    """Return {op: pass@k} from an eval_pass128 metrics.jsonl."""
    with open(metrics_path) as f:
        last = [json.loads(line) for line in f if line.strip()][-1]
    m = last["metrics"] if "metrics" in last else last
    out = {}
    pat = re.compile(rf"difficulty-5B/(\d+)/reward/pass@{k}$")
    for key, val in m.items():
        g = pat.search(key)
        if g:
            out[int(g.group(1))] = float(val)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", default="results/widen_line/exp_gsm")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    per_arch = {}
    for arch in ARCHS:
        mp = os.path.join(args.exp_root, "eval", arch, "metrics.jsonl")
        if os.path.exists(mp):
            per_arch[arch] = load_passk(mp, args.k)
        else:
            print(f"[skip] no metrics for {arch} ({mp})")

    if not per_arch:
        print("No metrics found yet.")
        return

    all_ops = sorted({op for d in per_arch.values() for op in d})
    print(f"\n=== pass@{args.k} by op (GSM-Infinity) ===")
    header = "arch".ljust(12) + "".join(f"op{op:>2}".rjust(8) for op in all_ops)
    print(header)
    for arch, d in per_arch.items():
        row = arch.ljust(12) + "".join(f"{d.get(op, float('nan')):8.3f}" for op in all_ops)
        print(row)

    print(f"\n=== ID (ops {ID_OPS}) vs OOD (ops {OOD_OPS}) mean pass@{args.k} ===")
    print("arch".ljust(12) + "ID".rjust(8) + "OOD".rjust(8) + "  OOD-ID".rjust(10))
    summary = {}
    for arch, d in per_arch.items():
        idv = [d[o] for o in ID_OPS if o in d]
        oov = [d[o] for o in OOD_OPS if o in d]
        id_m = sum(idv) / len(idv) if idv else float("nan")
        ood_m = sum(oov) / len(oov) if oov else float("nan")
        summary[arch] = (id_m, ood_m)
        print(arch.ljust(12) + f"{id_m:8.3f}{ood_m:8.3f}{ood_m - id_m:10.3f}")

    if args.no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa
        print(f"[plot skipped] matplotlib unavailable: {e}")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for arch, d in per_arch.items():
        ops = sorted(d)
        ax1.plot(ops, [d[o] for o in ops], marker="o", label=arch)
    ax1.axvline(11, ls="--", c="gray", alpha=0.6)
    ax1.set_xlabel("op level (>=11 = OOD)")
    ax1.set_ylabel(f"pass@{args.k}")
    ax1.set_title("GSM-Infinity accuracy vs depth")
    ax1.legend()
    ax1.grid(alpha=0.3)

    for arch, (idm, oodm) in summary.items():
        ax2.scatter(idm, oodm, s=80)
        ax2.annotate(arch, (idm, oodm), textcoords="offset points", xytext=(6, 4))
    ax2.set_xlabel("ID mean pass@1 (ops 2-10)")
    ax2.set_ylabel("OOD mean pass@1 (ops 12-20)")
    ax2.set_title("Universality view: OOD vs ID (on the line?)")
    ax2.grid(alpha=0.3)

    out = os.path.join(args.exp_root, f"widen_line_gsm_pass{args.k}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nSaved plot -> {out}")


if __name__ == "__main__":
    main()
