"""Widen-the-line: do AR architectures fall on ONE OOD-vs-ID line on GSM-Infinity?

Reads the multi-checkpoint pass@1 evals produced by exp_run.sh
(results/widen_line/exp_gsm/eval/<arch>/ckpt-<step>/metrics.jsonl). Each checkpoint
is one point at a different ID-accuracy level, so per arch we get a TRAJECTORY of
(ID, OOD) points -> the spread needed to actually see a line. If all archs' points
fall on one OOD-vs-ID curve, that's the universality-line phenomenon.

Outputs: per-(arch,ckpt) table, a single global line fit (R^2), and a scatter PNG
(points colored by arch, global fit overlaid). Usage:
    python analyze/widen_line_compare.py [--exp_root ...] [--no_plot]
"""
import argparse
import glob
import json
import os
import re

ARCHS = ["dense", "gqa", "moe", "looped", "tokenformer"]
ID_OPS = [2, 4, 6, 8, 10]
OOD_OPS = [12, 14, 16, 18, 20]


def load_passk(metrics_path, k=1):
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


def id_ood(passk):
    idv = [passk[o] for o in ID_OPS if o in passk]
    oov = [passk[o] for o in OOD_OPS if o in passk]
    return (sum(idv) / len(idv) if idv else float("nan"),
            sum(oov) / len(oov) if oov else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_root", default="results/widen_line/exp_gsm")
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    # points[arch] = list of (step, id_acc, ood_acc)
    points = {}
    for arch in ARCHS:
        adir = os.path.join(args.exp_root, "eval", arch)
        # multi-ckpt layout
        cks = sorted(glob.glob(os.path.join(adir, "ckpt-*", "metrics.jsonl")))
        # fall back to the old flat single-eval layout
        flat = os.path.join(adir, "metrics.jsonl")
        rows = []
        for mp in cks:
            step = int(re.search(r"ckpt-(\d+)", mp).group(1))
            idm, oodm = id_ood(load_passk(mp, args.k))
            rows.append((step, idm, oodm))
        if not rows and os.path.exists(flat):
            idm, oodm = id_ood(load_passk(flat, args.k))
            rows.append((0, idm, oodm))
        if rows:
            points[arch] = sorted(rows)

    if not points:
        print("No eval metrics found yet.")
        return

    print(f"\n=== (arch, ckpt) -> ID/OOD mean pass@{args.k} ===")
    print("arch".ljust(12) + "ckpt".rjust(8) + "ID".rjust(8) + "OOD".rjust(8))
    xs, ys = [], []
    for arch in points:
        for step, idm, oodm in points[arch]:
            print(arch.ljust(12) + f"{step:8d}{idm:8.3f}{oodm:8.3f}")
            if idm == idm and oodm == oodm:  # not nan
                xs.append(idm)
                ys.append(oodm)

    # single global line fit across ALL (arch, ckpt) points
    r2 = slope = intercept = float("nan")
    if len(xs) >= 2:
        n = len(xs)
        mx = sum(xs) / n
        my = sum(ys) / n
        sxx = sum((x - mx) ** 2 for x in xs)
        sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        if sxx > 0:
            slope = sxy / sxx
            intercept = my - slope * mx
            ss_res = sum((y - (slope * x + intercept)) ** 2 for x, y in zip(xs, ys))
            ss_tot = sum((y - my) ** 2 for y in ys)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    print(f"\nGlobal OOD-vs-ID line fit across all archs/ckpts: "
          f"OOD = {slope:.3f}*ID + {intercept:.3f}   R^2 = {r2:.3f}   (n={len(xs)})")
    print("(High R^2 with archs interleaved on the line => universality-line phenomenon.)")

    if args.no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa
        print(f"[plot skipped] matplotlib unavailable: {e}")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for arch in points:
        px = [p[1] for p in points[arch]]
        py = [p[2] for p in points[arch]]
        ax.plot(px, py, marker="o", ms=8, label=arch, alpha=0.85)
    if slope == slope:
        lo = min(xs)
        hi = max(xs)
        ax.plot([lo, hi], [slope * lo + intercept, slope * hi + intercept],
                "k--", alpha=0.6, label=f"global fit (R²={r2:.3f})")
    ax.set_xlabel(f"ID mean pass@{args.k} (ops 2-10)")
    ax.set_ylabel(f"OOD mean pass@{args.k} (ops 12-20)")
    ax.set_title("Widen the line: OOD vs ID across AR architectures (GSM-Infinity)")
    ax.legend()
    ax.grid(alpha=0.3)
    out = os.path.join(args.exp_root, f"widen_line_gsm_pass{args.k}.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"\nSaved plot -> {out}")


if __name__ == "__main__":
    main()
