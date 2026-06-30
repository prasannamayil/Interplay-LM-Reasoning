"""Widen-the-line on the HARD-skewed GSM distribution -> a real OOD-vs-ID spread.

The easy distribution saturated at ID~0.25 (op8-10 floored) at every size. On the
hard-skewed source (op8-10 = 50%), the hard ops become learnable, so accuracy both rises
and SPREADS along two axes: model size (xs..l) and training tokens (ckpt 4k/8k/12k).
EVERY (arch, size, ckpt) is a point; the universality claim is that all of them fall on
ONE OOD-vs-ID curve regardless of architecture.

Reads results/widen_line/exp_gsm_hard/eval/<arch>_<size>/ckpt-<step>/metrics.jsonl.
Outputs a table, a global line fit (R^2) across all points, and a scatter PNG
(marker per arch, every checkpoint a point). Usage: python analyze/widen_line_gsm_hard.py [--no_plot]
"""
import argparse, glob, json, os, re

ROOT = "results/widen_line/exp_gsm_hard/eval"
ARCHS = ["dense", "gqa", "moe", "looped", "tokenformer"]
SIZES = ["xs", "s", "m", "l", "xl"]
# OOD = NEAR-extrapolation (op11-14). Far ops 16-20 are a dead arch-invariant wall and
# wash the signal out; op11-14 has real capability-tracking spread (op12 0.13->0.36). Only
# the even ops (12,14) are in the main sweep eval; op11/13 come from eval_nearood (slow
# re-eval). id_ood() uses whichever OOD ops are present, so this is mean(op12,op14) now and
# becomes mean(op11..14) once nearood is merged. ID stays in-distribution mean(op2-10).
ID_OPS = [2, 4, 6, 8, 10]; OOD_OPS = [11, 12, 13, 14]


def _ops(path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    if not rows: return {}
    m = rows[-1].get("metrics", rows[-1])
    out = {}
    for op in range(2, 21):
        v = m.get(f"val-aux/difficulty-5B/{op}/reward/pass@1")
        if v is not None: out[op] = v
    return out


def id_ood(path):
    m = _ops(path)
    # merge op11/13 (and 12/14) from the eval_nearood re-eval if present
    near = path.replace("/eval/", "/eval_nearood/")
    if near != path and os.path.exists(near):
        m.update(_ops(near))
    idv = [m[o] for o in ID_OPS if o in m]
    oov = [m[o] for o in OOD_OPS if o in m]
    if not idv or not oov: return None
    return sum(idv)/len(idv), sum(oov)/len(oov)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()
    points = {}  # arch -> [(size, step, id, ood)]
    for arch in ARCHS:
        rows = []
        for size in SIZES:
            # {arch}_{size} plus optional _s<seed> variants (extra seeds = more line points)
            cands = glob.glob(f"{ROOT}/{arch}_{size}/ckpt-*/metrics.jsonl") + \
                    glob.glob(f"{ROOT}/{arch}_{size}_s*/ckpt-*/metrics.jsonl")
            for cp in sorted(cands, key=lambda p: int(re.search(r"ckpt-(\d+)", p).group(1))):
                step = int(re.search(r"ckpt-(\d+)", cp).group(1))
                r = id_ood(cp)
                if r: rows.append((size, step, r[0], r[1]))
        if rows: points[arch] = rows
    if not points:
        print(f"No hard-GSM eval points yet under {ROOT}."); return

    print("\n=== (arch, size, ckpt) -> ID / OOD pass@1  [HARD distribution] ===")
    print("arch".ljust(12)+"size".rjust(4)+"ckpt".rjust(7)+"ID".rjust(8)+"OOD".rjust(8))
    xs, ys = [], []
    for arch in points:
        for size, step, idm, oodm in points[arch]:
            print(arch.ljust(12)+size.rjust(4)+f"{step:7d}{idm:8.3f}{oodm:8.3f}")
            xs.append(idm); ys.append(oodm)
    r2 = slope = intercept = float("nan")
    if len(xs) >= 2:
        n=len(xs); mx=sum(xs)/n; my=sum(ys)/n
        sxx=sum((x-mx)**2 for x in xs); sxy=sum((x-mx)*(y-my) for x,y in zip(xs,ys))
        if sxx>0:
            slope=sxy/sxx; intercept=my-slope*mx
            ss_res=sum((y-(slope*x+intercept))**2 for x,y in zip(xs,ys)); ss_tot=sum((y-my)**2 for y in ys)
            r2=1-ss_res/ss_tot if ss_tot>0 else float("nan")
    print(f"\nGlobal OOD-vs-ID fit (all archs/sizes/ckpts): OOD={slope:.3f}*ID+{intercept:.3f}"
          f"  R^2={r2:.3f}  n={len(xs)}  ID spread={ (max(xs)-min(xs)) if xs else 0:.3f}")
    print("(Wide ID spread + high R^2 with archs interleaved => universality LINE on hard GSM.)")
    if args.no_plot: return
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[plot skipped] {e}"); return
    mk={"dense":"o","gqa":"s","moe":"^","looped":"D","tokenformer":"v"}
    fig,ax=plt.subplots(figsize=(7.5,6))
    for arch in points:
        ax.scatter([p[2] for p in points[arch]],[p[3] for p in points[arch]],
                   marker=mk.get(arch,"o"),s=60,alpha=0.8,label=arch)
    if slope==slope and xs:
        lo,hi=min(xs),max(xs)
        ax.plot([lo,hi],[slope*lo+intercept,slope*hi+intercept],"k--",alpha=.6,label=f"fit R²={r2:.3f}")
    ax.set_xlabel("ID mean pass@1 (ops 2-10)"); ax.set_ylabel("OOD mean pass@1 (ops 12-20)")
    ax.set_title("Widen the line: OOD vs ID across archs x size x ckpt (HARD GSM)")
    ax.legend(); ax.grid(alpha=.3)
    out=os.path.join(os.path.dirname(ROOT),"widen_line_gsm_hard.png")
    fig.tight_layout(); fig.savefig(out,dpi=130); print(f"\nSaved -> {out}")


if __name__ == "__main__":
    main()
