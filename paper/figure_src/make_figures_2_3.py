"""Figures 2 and 3 of the ESWA manuscript (same data and code as the released make_figures.py,
redrawn at the 5.3 in text width of the elsarticle preprint layout).

Original docstring: Figures for the paper, drawn from the released per-split results.

fig_power_alpha.pdf : power (left) and human effort (right) against alpha for
                      every dataset/annotator, HLCL vs the certified baselines.
fig_validity.pdf    : share of splits whose realised FDP exceeded alpha, all
                      methods; the contract is <= delta.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# the code repository is either the parent of this paper folder (paper/ inside the
# repository) or its sibling hlcl-code/ (separate working copy)
_P = Path(__file__).resolve().parents[1]
CODE = _P.parent if (_P.parent / "results").is_dir() else _P.parent / "hlcl-code"
ROOT = CODE   # released per-split results
OUT = Path(__file__).resolve().parents[1] / "figures"
VER = sys.argv[1] if len(sys.argv) > 1 else "v9"

SETTINGS = [("DBPedia (A)", "dbpedia"), ("WOS-46985 (A)", "wos"),
            ("Blurb (A)", "bgc"), ("Blurb (B)", "bgc_haiku"), ("Blurb (C)", "bgc_gpt"), ("Amazon (A)", "amazon")]
METHODS = [("HLCL", "HLCL (ours)", "#1b6ca8", "o", "-"),
           ("ItemBest", "Item-level certified (best test)", "#d1495b", "s", "--"),
           ("HSC", "Hierarchical selective cls.", "#00798c", "P", "--"),
           ("PerField-LTT", "Per-field risk control", "#66a182", "D", ":")]
ALL = METHODS + [("ConfLabel", "Item-level conformal", "#edae49", "^", "--"),
                 ("DepthBH", "Depth-wise BH", "#8d6a9f", "v", "-."),
                 ("Conf>=0.9", "Fixed threshold ($s\\geq0.9$)", "#555555", "x", "-.")]

# drawn at the final width (elsarticle preprint text width 384 pt = 5.33 in); no text below 7 pt
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "mathtext.fontset": "dejavuserif",
                     "axes.linewidth": 0.6, "legend.fontsize": 8, "xtick.labelsize": 8,
                     "ytick.labelsize": 8, "axes.titlesize": 8, "pdf.fonttype": 42, "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})


def series(runs, m, key):
    al = sorted({r["alpha"] for r in runs})
    return al, [np.mean([r[m][key] for r in runs if r["alpha"] == a]) for a in al]


def viol(runs, m):
    al = sorted({r["alpha"] for r in runs})
    return al, [np.mean([r[m]["fdp"] > a + 1e-12 for r in runs if r["alpha"] == a]) for a in al]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    data = {k: json.loads((ROOT / "results" / f"{VER}_{k}.json").read_text())["runs"]
            for _, k in SETTINGS}
    # Item-cert. with the best of its three test choices per alpha (as in the tables)
    for runs in data.values():
        for a in sorted({r["alpha"] for r in runs}):
            R = [r for r in runs if r["alpha"] == a]
            have = [m for m in ("ItemLTT", "ItemLTT-wsr", "ItemLTT-iut") if m in R[0]]
            b = min(have, key=lambda m: np.mean([r[m]["effort"] for r in R]))
            for r in R:
                r["ItemBest"] = r[b]

    fig, ax = plt.subplots(2, len(SETTINGS), figsize=(5.15, 3.7), sharex="col", sharey=True)
    for j, (title, k) in enumerate(SETTINGS):
        for m, lab, col, mk, ls in METHODS:
            a, p = series(data[k], m, "power")
            _, e = series(data[k], m, "effort")
            _, v = viol(data[k], m)
            bad = [x > 0.1 + 1e-9 for x in v]          # realised FDP > alpha in more than delta of splits
            for row, y in ((0, p), (1, e)):
                ax[row, j].plot(a, y, color=col, ls=ls, lw=1, label=lab if row == 0 else None)
                ax[row, j].plot([x for x, b in zip(a, y) if True], y, ls="none", marker=mk, ms=3,
                                color=col, markevery=[i for i, b in enumerate(bad) if not b])
                ax[row, j].plot(a, y, ls="none", marker=mk, ms=3.4, mfc="white", mec=col, mew=0.8,
                                markevery=[i for i, b in enumerate(bad) if b])
        ax[0, j].set_title(title)
        ax[1, j].set_xlabel(r"$\alpha$")
        ax[0, j].set_ylim(-0.02, 1.02)
        ax[1, j].set_ylim(-0.02, 1.02)
        al = sorted({r["alpha"] for r in data[k]})
        ax[1, j].set_xticks(al[1::2], [f"{x:g}" for x in al[1::2]])
    ax[0, 0].set_ylabel("power")
    ax[1, 0].set_ylabel("human effort")
    h = [plt.Line2D([], [], color=col, ls=ls, lw=1, marker=mk, ms=3) for _, _, col, mk, ls in METHODS]
    l = [lab for _, lab, _, _, _ in METHODS]
    h.append(plt.Line2D([], [], ls="none", marker="o", ms=3.4, mfc="white", mec="#555555", mew=0.8))
    l.append(r"hollow: target missed in $>10\%$ of splits")
    fig.tight_layout(rect=(0, 0.13, 1, 1), w_pad=0.4)
    fig.legend(h, l, loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(OUT / "Figure_2.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(Path(__file__).with_name("Figure_2_preview.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)

    fig, ax = plt.subplots(1, len(SETTINGS), figsize=(5.15, 2.6), sharey=True)
    for j, (title, k) in enumerate(SETTINGS):
        for m, lab, col, mk, ls in ALL:
            a, v = viol(data[k], m)
            ax[j].plot(a, v, color=col, marker=mk, ls=ls, ms=3, lw=1, label=lab)
        ax[j].axhline(0.1, color="k", lw=0.6, ls=(0, (2, 2)))
        ax[j].set_title(title)
        ax[j].set_xlabel(r"$\alpha$")
        ax[j].set_ylim(-0.03, 1.03)
        al = sorted({r["alpha"] for r in data[k]})
        ax[j].set_xticks(al[1::2], [f"{x:g}" for x in al[1::2]])
    ax[0].set_ylabel(r"share of splits, FDP $>\alpha$")
    h, l = ax[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0.25, 1, 1), w_pad=0.4)
    fig.legend(h, l, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.0))
    fig.savefig(OUT / "Figure_3.pdf", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(Path(__file__).with_name("Figure_3_preview.png"), dpi=200, bbox_inches="tight", pad_inches=0.02)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
