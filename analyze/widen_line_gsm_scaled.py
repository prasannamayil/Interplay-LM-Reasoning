"""Widen-the-line on GSM-Infinity via a MODEL-SIZE SWEEP -> a real OOD-vs-ID spread.

Single-checkpoint GSM pass@1 saturates (templated data) so one model is a point, not a
line. Training several SIZES per arch (xs..xl) and taking each model's CONVERGED accuracy
gives a spread of (ID, OOD) points along the ID axis. The universality claim: every
(arch, size) point lands on ONE OOD-vs-ID curve regardless of architecture.

Reads:
  results/widen_line/exp_gsm_scaled/eval/<arch>_<size>/ckpt-<step>/metrics.jsonl   (sweep)
  results/widen_line/exp_gsm/eval/<arch>/ckpt-<step>/metrics.jsonl                 (the m=79M size)
Per (arch,size) we take the LAST evaluated checkpoint (converged). Outputs a table, a
single global line fit (R^2) across all (arch,size), and a scatter PNG (one marker shape
per arch, color by size; global fit overlaid). Usage:
    python analyze/widen_line_gsm_scaled.py [--no_plot]
"""
import argparse
import glob
import json
import os
import re

ARCHS = ["dense", "gqa", "moe", "looped", "tokenformer"]
SIZES = ["xs", "s", "m", "l", "xl"]
ID_OPS = [2, 4, 6, 8, 10]
OOD_OPS = [12, 14, 16, 18, 20]
SCALED_ROOT = "results/widen_line/exp_gsm_scaled"
MSIZE_ROOT = "results/widen_line/exp_gsm"   # the existing ~79M run == size 'm'
RAMP_ROOT = "results/widen_line/exp_gsm_ramp"  # early-checkpoint ramp (low-end spread)


def load_passk(metrics_path, k=1):
    with open(metrics_path) as f:
        rows = [json.loads(line) for line in f if line.strip()]
    if not rows:
        return {}
    m = rows[-1].get("metrics", rows[-1])
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


def converged_point(eval_dir):
    """Take the highest-step checkpoint eval in eval_dir as the converged point."""
    cks = glob.glob(os.path.join(eval_dir, "ckpt-*", "metrics.jsonl"))
    if not cks:
        return None
    cks.sort(key=lambda p: int(re.search(r"ckpt-(\d+)", p).group(1)))
    return id_ood(load_passk(cks[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    # points[arch] = list of (label, id, ood); spread comes from BOTH levers:
    #   size sweep (converged point per size) + early-checkpoint ramp (per ckpt).
    points = {}
    for arch in ARCHS:
        rows = []
        # 1) size sweep: one converged point per size
        for size in SIZES:
            if size == "m":
                d = os.path.join(MSIZE_ROOT, "eval", arch)
            else:
                d = os.path.join(SCALED_ROOT, "eval", f"{arch}_{size}")
            pt = converged_point(d)
            if pt and pt[0] == pt[0]:
                rows.append((f"sz:{size}", pt[0], pt[1]))
        # 2) early-checkpoint ramp: every evaluated ckpt is a point (pre-saturation = low ID)
        for cp in sorted(glob.glob(os.path.join(RAMP_ROOT, "eval", arch, "ckpt-*", "metrics.jsonl")),
                         key=lambda p: int(re.search(r"ckpt-(\d+)", p).group(1))):
            step = int(re.search(r"ckpt-(\d+)", cp).group(1))
            idm, oodm = id_ood(load_passk(cp))
            if idm == idm:
                rows.append((f"r:{step}", idm, oodm))
        if rows:
            points[arch] = rows

    if not points:
        print("No scaled GSM eval metrics yet (jobs still training).")
        return

    print("\n=== (arch, point) -> ID / OOD pass@1  [sz:=size sweep, r:=ramp ckpt] ===")
    print("arch".ljust(12) + "point".ljust(9) + "ID".rjust(8) + "OOD".rjust(8))
    xs, ys = [], []
    for arch in points:
        for label, idm, oodm in points[arch]:
            print(arch.ljust(12) + label.ljust(9) + f"{idm:8.3f}{oodm:8.3f}")
            xs.append(idm); ys.append(oodm)

    r2 = slope = intercept = float("nan")
    if len(xs) >= 2:
        n = len(xs); mx = sum(xs)/n; my = sum(ys)/n
        sxx = sum((x-mx)**2 for x in xs); sxy = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        if sxx > 0:
            slope = sxy/sxx; intercept = my - slope*mx
            ss_res = sum((y-(slope*x+intercept))**2 for x, y in zip(xs, ys))
            ss_tot = sum((y-my)**2 for y in ys)
            r2 = 1 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    idspread = (max(xs)-min(xs)) if xs else 0.0
    print(f"\nGlobal OOD-vs-ID fit across all (arch,size): OOD = {slope:.3f}*ID + {intercept:.3f}"
          f"   R^2 = {r2:.3f}   (n={len(xs)}, ID spread={idspread:.3f})")
    print("(Wide ID spread + high R^2 with archs interleaved => universality LINE on GSM.)")

    if args.no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:  # noqa
        print(f"[plot skipped] matplotlib unavailable: {e}")
        return
    markers = {"dense": "o", "gqa": "s", "moe": "^", "looped": "D", "tokenformer": "v"}
    fig, ax = plt.subplots(figsize=(7.5, 6))
    for arch in points:
        pts = sorted(points[arch], key=lambda r: r[1])
        ax.plot([p[1] for p in pts], [p[2] for p in pts],
                marker=markers.get(arch, "o"), ms=9, label=arch, alpha=0.85)
    if slope == slope and xs:
        lo, hi = min(xs), max(xs)
        ax.plot([lo, hi], [slope*lo+intercept, slope*hi+intercept], "k--",
                alpha=0.6, label=f"global fit (R²={r2:.3f})")
    ax.set_xlabel("ID mean pass@1 (ops 2-10)")
    ax.set_ylabel("OOD mean pass@1 (ops 12-20)")
    ax.set_title("Widen the line: OOD vs ID across AR archs x model size (GSM-Infinity)")
    ax.legend(); ax.grid(alpha=0.3)
    out = os.path.join(SCALED_ROOT, "widen_line_gsm_scaled.png")
    os.makedirs(SCALED_ROOT, exist_ok=True)
    fig.tight_layout(); fig.savefig(out, dpi=130)
    print(f"\nSaved plot -> {out}")


if __name__ == "__main__":
    main()
