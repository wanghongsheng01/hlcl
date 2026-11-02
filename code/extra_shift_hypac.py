"""HyPAC baseline and distribution-shift checks (cached responses only).

HyPAC (Zeng et al. 2026, arXiv:2602.02550) with one model and a person: an item
is labeled by the model if its score is at least u and by a person otherwise.
Its risk is the population annotation error over all items, with reviewed
items counted as correct; here the loss of an accepted item is its share of
wrong labels. The threshold is certified by fixed-sequence testing over the
same 64 item-score candidates as Item-cert., ordered from conservative to
permissive (HyPAC's monotone sequence), with the betting p-value (HyPAC's
finite-sample variant), and the most permissive certified threshold, which
has the lowest cost, is deployed.

Shift checks for HLCL, calibrated as in experiment2.py:
  hard     test items restricted to the level-1 classes whose consensus accuracy
           on the fit split is below the median (label shift towards hard items)
  annot    Blurb only: rule fitted and calibrated on one annotator, applied to
           another annotator's outputs for the same test items (model change)

Run: python extra_shift_hypac.py   (writes results/v11shift_<key>.json)
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

import hlcl2
from experiment2 import dense, item_score, levels
from extra_analysis import SETTINGS, describe
from datasets_local import load_dataset_items
from features import build, fit_score
from hlcl2 import accept, betting_p, calibrate, full_grid

ROOT = Path(__file__).resolve().parents[1]
DELTA = 0.1
BLURB = {"bgc": "bgc_deepseek_default_k5_t1.0", "bgc_haiku": "bgc_alt_claude-haiku-4-5_k5_t1.0",
         "bgc_gpt": "bgc_alt_gpt-5.6-luna_k5_t1.0"}


def hypac(S, C, M, fi, ci, ti, eps):
    D = S.shape[1]
    Si = item_score(S, M)
    qs = np.unique(np.concatenate([np.quantile(Si[fi], np.linspace(0, 1, 64)), [np.inf]]))[::-1]
    best = np.inf
    for u in qs:                                   # conservative first
        acc = Si[ci] >= u
        loss = acc * (C[ci] == 0).sum(1) / D       # share of wrong labels of accepted items
        if betting_p(loss - eps, -eps, 1 - eps) > DELTA:
            break
        best = u
    A = np.repeat((Si[ti] >= best)[:, None], D, 1) & M[ti]
    return A


def hlcl_rule(S, C, M, fi, ci, D, al, m_levels):
    g_t = [np.full(D, v) for v in np.unique(np.concatenate(
        [np.quantile(S[fi][M[fi]], np.linspace(0, 1, 4 * m_levels)), [np.inf]]))]
    g_l = full_grid(levels(S, fi, m_levels)) + g_t
    lam, _, _ = calibrate(S[ci], C[ci], M[ci], g_l, al, DELTA, order=(S[fi], C[fi], M[fi]))
    return lam


def job(args):
    key, al, R = args
    dataset, tag, _, m_levels = SETTINGS[key]
    a, X, meta = build(dataset, tag, 3000, 5)
    D = meta["depth"]
    N = a.n_items
    others = {}
    if key in BLURB:
        for k2, t2 in BLURB.items():
            if k2 != key:
                a2, X2, _ = build("bgc", t2, 3000, 5)
                others[k2] = (a2, X2)
    y1 = np.array([it["gold"][0] for it in load_dataset_items(dataset, 3000)[0]])
    out = []
    for r in range(R):
        rng = np.random.default_rng(1000 + r)
        perm = rng.permutation(N)
        fi, ci, ti = perm[:600], np.sort(perm[600:1800]), np.sort(perm[1800:])
        fit_nodes = np.isin(a.item, fi)
        s = fit_score(X[fit_nodes], a.correct[fit_nodes], X)
        S, C, V, M = dense(a, X, s)
        row = {"HyPAC": describe(hypac(S, C, M, fi, ci, ti, al), C[ti], M[ti])}
        lam = hlcl_rule(S, C, M, fi, ci, D, al, m_levels)
        A = (accept(S[ti], lam) if lam is not None else np.zeros((len(ti), D), bool)) & M[ti]
        row["HLCL"] = describe(A, C[ti], M[ti])
        # label shift: level-1 classes with below-median consensus accuracy on the fit split
        cls = np.unique(y1[fi])
        acc1 = np.array([C[fi][y1[fi] == c, 0].mean() for c in cls])
        hard = set(cls[acc1 < np.median(acc1)]) or set(cls[acc1 <= np.median(acc1)])
        th = np.array([y1[i] in hard for i in ti])
        if th.sum() > 0:
            row["HLCL-hard"] = describe(A[th], C[ti][th], M[ti][th]) | {"n": int(th.sum())}
        # annotator change: the rule of this annotator applied to another annotator's outputs
        for k2, (a2, X2) in others.items():
            s2 = fit_score(X[fit_nodes], a.correct[fit_nodes], X2)   # same fitted score, other inputs
            S2, C2, _, M2 = dense(a2, X2, s2)
            A2 = (accept(S2[ti], lam) if lam is not None else np.zeros((len(ti), D), bool)) & M2[ti]
            row[f"HLCL-to-{k2}"] = describe(A2, C2[ti], M2[ti])
        out.append(row)
    return key, al, out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="v11shift")
    ap.add_argument("--R", type=int, default=100)
    ap.add_argument("--keys", default=",".join(SETTINGS))
    ap.add_argument("--workers", type=int, default=8)
    x = ap.parse_args()
    hlcl2.C0 = 0.5
    jobs = [(k, al, x.R) for k in x.keys.split(",") for al in SETTINGS[k][2]]
    res: dict = {}
    with Pool(x.workers) as p:
        for key, al, rows in p.imap_unordered(job, jobs):
            res.setdefault(key, {})[str(al)] = rows
            print("done", key, al, flush=True)
    for key, v in res.items():
        (ROOT / "results" / f"{x.out}_{key}.json").write_text(json.dumps(v))
