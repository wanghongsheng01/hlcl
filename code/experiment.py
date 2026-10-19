"""Main experiment: validity and power of label-level acceptance rules.

Methods (all see only the calibration half of each random split):
  HLCL        per-depth thresholds, ancestor-closed back-off, LTT (ours)
  HLCL-tied   single shared threshold, back-off, LTT            (ablation)
  Flat-LTT    per-depth thresholds, no ancestor closure, LTT    (ablation)
  ItemCS      item-level conformal selection + BH (Jin & Candes 2023;
              the selection rule used by Conformal Labeling, 2025):
              an item is accepted with all its labels or routed entirely
  DepthCS     conformal selection + BH run separately at each depth
              (label-level, no hierarchy coupling)
  Conf>=0.9   fixed threshold on the score, no guarantee (practitioner rule)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from hlcl import Annotated, accepted_mask, evaluate, ltt_calibrate, make_grid

ROOT = Path(__file__).resolve().parents[1]


def bh(p: np.ndarray, alpha: float) -> np.ndarray:
    m = len(p)
    if m == 0:
        return np.zeros(0, bool)
    o = np.argsort(p)
    ok = p[o] <= alpha * np.arange(1, m + 1) / m
    k = np.max(np.nonzero(ok)[0]) + 1 if ok.any() else 0
    sel = np.zeros(m, bool)
    sel[o[:k]] = True
    return sel


def conformal_p(cal_s, cal_bad, test_s, rng):
    """Randomised conformal p-values for H0_j: test unit j is wrong
    (Jin & Candes 2023, clipped score V = s for nulls)."""
    null_s = np.sort(cal_s[cal_bad])
    n = len(cal_s)
    ge = len(null_s) - np.searchsorted(null_s, test_s, side="left")
    eq = np.searchsorted(null_s, test_s, side="right") - np.searchsorted(null_s, test_s, side="left")
    u = rng.random(len(test_s))
    return (ge - eq + u * (eq + 1)) / (n + 1)


def item_cs(cal: Annotated, test: Annotated, alpha, rng):
    D = cal.depth.max() + 1

    def per_item(a):
        leaf = np.full(a.n_items, -1)
        for j in np.argsort(a.depth):
            leaf[a.item[j]] = j
        s = np.where(leaf >= 0, a.score[np.maximum(leaf, 0)], 0.0)
        # item is "good" only if the full predicted path exists and is correct
        full = np.bincount(a.item, minlength=a.n_items) == D
        good = np.bincount(a.item, weights=a.correct, minlength=a.n_items) == D
        s = np.where(full, s, 0.0)
        # path score = min over nodes (vote is non-increasing along the path)
        mn = np.full(a.n_items, np.inf)
        np.minimum.at(mn, a.item, a.score)
        s = np.where(full, mn, 0.0)
        return s, good

    cs, cg = per_item(cal)
    ts, _ = per_item(test)
    p = conformal_p(cs, ~cg, ts, rng)
    sel = bh(p, alpha)
    return sel[test.item]


def item_ltt(cal: Annotated, test: Annotated, q, alpha, delta, D):
    """Item-level acceptance with the same high-probability guarantee as HLCL:
    an item's whole path is accepted iff its weakest node clears one threshold."""
    def collapse(a):
        mn = np.full(a.n_items, np.inf)
        np.minimum.at(mn, a.item, a.score)
        full = np.bincount(a.item, minlength=a.n_items) == D
        s = np.where(full, mn, -1.0)
        return Annotated(a.item, a.depth, s[a.item], a.correct, a.parent, a.n_items)

    c, t = collapse(cal), collapse(test)
    lam = ltt_calibrate(c, make_grid(D, q, tie=True), alpha, delta, hier=False,
                        max_labels_per_item=D)
    if lam is None:
        return np.zeros(len(test.score), bool)
    return t.score >= lam[0]


def depth_cs(cal: Annotated, test: Annotated, alpha, rng):
    acc = np.zeros(len(test.score), bool)
    for d in range(cal.depth.max() + 1):
        cm, tm = cal.depth == d, test.depth == d
        p = conformal_p(cal.score[cm], cal.correct[cm] == 0, test.score[tm], rng)
        acc[np.nonzero(tm)[0]] = bh(p, alpha)
    return acc


def run(a: Annotated, alphas, R, delta, out_name, meta):
    D = a.depth.max() + 1
    # candidate thresholds: observed score levels only (a threshold above every
    # attainable score accepts nothing and can never be certified, which would
    # stop fixed-sequence testing at its first step)
    q = np.unique(a.score)[::-1]
    q = q[np.linspace(0, len(q) - 1, min(len(q), 120)).astype(int)]
    grid_d = make_grid(D, q)
    grid_t = make_grid(D, q, tie=True)
    res = []
    for r in range(R):
        rng = np.random.default_rng(r)
        perm = rng.permutation(a.n_items)
        ci, ti = np.sort(perm[: a.n_items // 2]), np.sort(perm[a.n_items // 2:])
        cal, test = a.subset(ci), a.subset(ti)
        for al in alphas:
            row = {"split": r, "alpha": al}
            for name, grid, hier in (("HLCL", grid_d, True), ("HLCL-tied", grid_t, True),
                                     ("Flat-LTT", grid_d, False)):
                lam = ltt_calibrate(cal, grid, al, delta, hier=hier, max_labels_per_item=D)
                acc = (accepted_mask(test, lam, hier) if lam is not None
                       else np.zeros(len(test.score), bool))
                row[name] = evaluate(test, acc)
            row["ItemCS"] = evaluate(test, item_cs(cal, test, al, rng))
            row["ItemLTT"] = evaluate(test, item_ltt(cal, test, q, al, delta, D))
            row["DepthCS"] = evaluate(test, depth_cs(cal, test, al, rng))
            row["Conf>=0.9"] = evaluate(test, test.score >= 0.9)
            res.append(row)
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / f"{out_name}.json").write_text(json.dumps({"meta": meta, "runs": res}))
    summarize(res, alphas)


def summarize(res, alphas):
    methods = [k for k in res[0] if k not in ("split", "alpha")]
    for al in alphas:
        rows = [r for r in res if r["alpha"] == al]
        print(f"\nalpha={al}  (R={len(rows)})")
        print(f"{'method':12s} {'meanFDP':>8s} {'P(FDP>a)':>9s} {'power':>7s} {'%lab acc':>9s} {'%items auto':>11s}")
        for m in methods:
            f = np.array([r[m]["fdp"] for r in rows])
            pw = np.mean([r[m]["power"] for r in rows])
            fl = np.mean([r[m]["frac_labels"] for r in rows])
            fa = np.mean([r[m]["items_fully_auto"] for r in rows])
            print(f"{m:12s} {f.mean():8.4f} {np.mean(f > al):9.3f} {pw:7.3f} {fl:9.3f} {fa:11.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--R", type=int, default=100)
    ap.add_argument("--delta", type=float, default=0.1)
    ap.add_argument("--alphas", default="0.02,0.05,0.1")
    ap.add_argument("--out", default=None)
    x = ap.parse_args()
    from scores import build
    a, vote, verb, meta = build(x.dataset, x.tag, x.n, x.k)
    print(meta)
    run(a, [float(v) for v in x.alphas.split(",")], x.R, x.delta,
        x.out or f"{x.dataset}_{x.tag}", meta)
