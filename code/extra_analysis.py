"""Supplementary analyses for the Machine Learning version (cached responses only).

For every setting it repeats the 100 random splits of experiment2.py and keeps,
for HLCL, the item-level certified rule (three test choices), the tied variant,
HSC and Per-field:
  q_d          error rate among the labels accepted at depth d (test split)
  pow_d        share of correct depth-d labels accepted
  stored       error rate of all stored labels when a reviewer keeps the
               accepted prefix and completes the path (a wrong accepted node
               makes every label below it wrong)
  eff_lin      effort with fine-heavy weights w_d proportional to d
  eff_c2       effort with review overhead c0 = 2
It also runs a population check: the 3,000 items are treated as the
population, fit and calibration samples are drawn from them with replacement,
and the exact mFDR of each deployed rule is computed on all 3,000 items.

Run:  python extra_analysis.py --out v10extra   (writes results/v10extra_<key>.json)
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import hlcl2
from experiment2 import dense, hsc, item_certified, item_score, levels, perfield_ltt
from features import build, fit_score
from hlcl2 import accept, calibrate, effort, full_grid

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = {
    "dbpedia": ("dbpedia", "dbpedia_deepseek_default_k5_t1.0", [0.01, 0.02, 0.05, 0.1], 8),
    "wos": ("wos", "wos_deepseek_default_k5_t1.0", [0.1, 0.15, 0.2, 0.3], 8),
    "bgc": ("bgc", "bgc_deepseek_default_k5_t1.0", [0.02, 0.05, 0.1, 0.15], 8),
    "bgc_haiku": ("bgc", "bgc_alt_claude-haiku-4-5_k5_t1.0", [0.02, 0.05, 0.1, 0.15], 8),
    "bgc_gpt": ("bgc", "bgc_alt_gpt-5.6-luna_k5_t1.0", [0.02, 0.05, 0.1, 0.15], 8),
    "amazon": ("amazon", "amazon_deepseek_default_k5_t1.0", [0.02, 0.05, 0.1, 0.15], 6),
}
DELTA = 0.1


def describe(A, C, M):
    D = A.shape[1]
    wrong = A & (C == 0)
    out = {"fdp": float(wrong.sum() / max(1, A.sum())),
           "power": float((A & (C == 1)).sum() / max(1, (C == 1).sum())),
           "effort": effort(A, M),
           "eff_lin": effort(A, M, w=np.arange(1, D + 1)),
           "eff_c2": effort(A, M, c0=2.0)}
    for d in range(D):
        n = A[:, d].sum()
        out[f"q{d+1}"] = float(wrong[:, d].sum() / n) if n else None
        out[f"n{d+1}"] = int(n)
        out[f"pow{d+1}"] = float((A[:, d] & (C[:, d] == 1)).sum() / max(1, (C[:, d] == 1).sum()))
    first = np.where(wrong.any(1), wrong.argmax(1), D)
    out["stored"] = float((D - first).sum() / (len(A) * D))
    return out


def rules(a, X, fi, ci, ti, D, al, m_levels):
    """Deployed acceptance matrices on the items `ti` for every rule."""
    fit_nodes = np.isin(a.item, fi)
    s = fit_score(X[fit_nodes], a.correct[fit_nodes], X)
    S, C, V, M = dense(a, X, s)
    ufi = np.unique(fi)
    g_t = [np.full(D, v) for v in np.unique(np.concatenate(
        [np.quantile(S[ufi][M[ufi]], np.linspace(0, 1, 4 * m_levels)), [np.inf]]))]
    g_l = full_grid(levels(S, ufi, m_levels)) + g_t
    out = {}
    for name, grid in (("HLCL", g_l), ("HLCL-tied", g_t)):
        lam, _, how = calibrate(S[ci], C[ci], M[ci], grid, al, DELTA, order=(S[fi], C[fi], M[fi]))
        out[name] = (accept(S[ti], lam) if lam is not None else np.zeros((len(ti), D), bool)) & M[ti]
    Sti = np.where(M[ti], item_score(S[ti], M[ti])[:, None], np.nan)
    for name, tt in (("ItemLTT", ("wsr", "iut")), ("ItemLTT-wsr", ("wsr",)), ("ItemLTT-iut", ("iut",))):
        lam, _, _ = item_certified(S, C, M, fi, ci, al, DELTA, tests=tt)
        out[name] = (accept(Sti, lam, hier=False) if lam is not None
                     else np.zeros((len(ti), D), bool)) & M[ti]
    out["HSC"] = hsc(S, C, M, fi, ci, ti, al, DELTA)
    out["PerField-LTT"] = perfield_ltt(S, C, M, fi, ci, ti, al, DELTA) & M[ti]
    return out, C[ti], M[ti]


def job(args):
    key, al, R = args
    dataset, tag, _, m_levels = SETTINGS[key]
    a, X, meta = build(dataset, tag, 3000, 5)
    D = meta["depth"]
    N = a.n_items
    split, pop = [], []
    for r in range(R):
        rng = np.random.default_rng(1000 + r)            # same splits as experiment2.py
        perm = rng.permutation(N)
        fi, ci, ti = perm[:600], np.sort(perm[600:1800]), np.sort(perm[1800:])
        A, C, M = rules(a, X, fi, ci, ti, D, al, m_levels)
        split.append({m: describe(A[m], C, M) for m in A})
        rb = np.random.default_rng(5000 + r)             # population check
        fb, cb = rb.integers(0, N, 600), np.sort(rb.integers(0, N, 1200))
        A, C, M = rules(a, X, fb, cb, np.arange(N), D, al, m_levels)
        pop.append({m: float((A[m] & (C == 0)).sum() / max(1, A[m].sum())) for m in A})
    return key, al, split, pop


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="v10extra")
    ap.add_argument("--R", type=int, default=100)
    ap.add_argument("--keys", default=",".join(SETTINGS))
    ap.add_argument("--workers", type=int, default=8)
    x = ap.parse_args()
    hlcl2.C0 = 0.5
    jobs = [(k, al, x.R) for k in x.keys.split(",") for al in SETTINGS[k][2]]
    res: dict = {}
    with Pool(x.workers) as p:
        for key, al, split, pop in p.imap_unordered(job, jobs):
            res.setdefault(key, {})[str(al)] = {"split": split, "pop": pop}
            print("done", key, al, flush=True)
    for key, v in res.items():
        (ROOT / "results" / f"{x.out}_{key}.json").write_text(json.dumps(v))
