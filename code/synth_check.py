"""Synthetic sanity check of the LTT guarantee (exact ground truth, many splits)."""
import numpy as np

from experiment import run
from hlcl import Annotated


def synth(n=4000, D=3, seed=0):
    rng = np.random.default_rng(seed)
    item, dep, score, corr, par = [], [], [], [], []
    for i in range(n):
        ok, prev = True, -1
        diff = rng.beta(2, 5)
        for d in range(D):
            p_ok = (0.97 - 0.08 * d) * (1 - 0.6 * diff)
            ok = ok and rng.random() < p_ok
            s = np.clip(rng.normal(0.75 if ok else 0.45, 0.18), 0, 1)
            s = min(s, score[prev]) if prev >= 0 else s
            item.append(i); dep.append(d); score.append(round(s * 5) / 5)
            corr.append(int(ok)); par.append(prev); prev = len(item) - 1
    return Annotated(np.array(item), np.array(dep), np.array(score),
                     np.array(corr), np.array(par), n)


if __name__ == "__main__":
    a = synth()
    print("acc by depth", [a.correct[a.depth == d].mean() for d in range(3)])
    run(a, [0.05, 0.1], R=40, delta=0.1, out_name="synthetic_sanity", meta={})
