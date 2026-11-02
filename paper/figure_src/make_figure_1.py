"""Figure 1: overview of HLCL (pipeline and an illustrative acceptance decision).

Drawn at the 5.3 in text width of the elsarticle preprint layout so that no text is scaled.
Run:  python3 make_figure_1.py   -> ../figures/Figure_1.pdf (vector, TrueType fonts embedded)
The scores in panel (b) are illustrative, not measured values.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "figures"
plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "mathtext.fontset": "dejavuserif",
                     "pdf.fonttype": 42, "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold"})

BLUE, GREEN, ORANGE = "#dbe8f5", "#dcefdc", "#fde3c8"
EDGE = "#4a4a4a"
W, H = 5.15, 5.71


def box(ax, x, y, w, h, title, body, fc, size=7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.006,rounding_size=0.01",
                                fc=fc, ec=EDGE, lw=0.7))
    ax.text(x + w / 2, y + h - 0.016, title, ha="center", va="top", weight="bold", fontsize=8)
    ax.text(x + w / 2, y + h - 0.052, body, ha="center", va="top", fontsize=size, linespacing=1.25)


def arrow(ax, p, q, **kw):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=7, lw=0.8, color=EDGE, **kw))


fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(-0.012, 1.012)
ax.set_ylim(0.105, 1.004)
ax.axis("off")

# (a) pipeline -----------------------------------------------------------------
ax.text(0.005, 0.995, "(a)", weight="bold", fontsize=8, va="top")
sy, sh, sw = 0.80, 0.115, 0.17
sx = [0.004, 0.207, 0.41, 0.613, 0.816]
for x, t, b, fc in [(sx[0], "Items", "texts to label\n(products, books,\npapers, ...)", "#f2f2f2"),
                    (sx[1], "LLM annotator", "$k$ sampled paths\nwith stated\nconfidences", BLUE),
                    (sx[2], "HLCL", "accept or route\neach predicted\nnode", GREEN),
                    (sx[3], "Review queue", "reviewer decides\nthe open levels;\nmay overturn", ORANGE),
                    (sx[4], "Label database", "accepted prefixes,\nreviewed levels,\npolicy log", "#f2f2f2")]:
    box(ax, x, sy, sw, sh, t, b, fc, size=7)
for i in range(4):
    arrow(ax, (sx[i] + sw + 0.002, sy + sh / 2), (sx[i + 1] - 0.002, sy + sh / 2))
ax.add_patch(FancyArrowPatch((sx[2] + sw / 2, sy + sh + 0.004), (sx[4] + sw / 2, sy + sh + 0.004),
             connectionstyle="arc3,rad=-0.16", arrowstyle="-|>", mutation_scale=7, lw=0.8, color=EDGE))
ax.text((sx[2] + sx[4] + sw) / 2, sy + sh + 0.048, "accepted prefix written directly", ha="center", fontsize=8)
ax.add_patch(FancyArrowPatch((sx[4] + sw / 2, sy - 0.004), (sx[2] + sw / 2, sy - 0.004),
             connectionstyle="arc3,rad=-0.18", arrowstyle="-|>", mutation_scale=7, lw=0.8, color=EDGE, ls="--"))
ax.text((sx[2] + sx[4] + sw) / 2, sy - 0.062, "random audit sample: monitor and recalibrate", ha="center", fontsize=8,
        style="italic")
ax.text(0.005, 0.745, "(b)", weight="bold", fontsize=8, va="top")
w, h, g = 0.23, 0.163, 0.026
xs = [0.004 + i * (w + g) for i in range(4)]
top, bot = 0.555, 0.345
box(ax, xs[0], top, w, h, "1. Annotate", "LLM queried $k=5$\ntimes per item;\neach sample gives\na path and a\nconfidence per level", BLUE)
box(ax, xs[1], top, w, h, "2. Describe nodes", "consensus path; vote,\nconditional agreement,\nsibling margin, entropy,\nstated confidence,\nparent fan-out", BLUE)
box(ax, xs[2], top, w, h, "3. Score nodes", "logistic regression\nfitted on the fit split\ngives a score\n$s\\in(0,1)$ for every\npredicted node", BLUE)
box(ax, xs[3], top, w, h, "4. Candidates", "per-depth thresholds\n$\\lambda=(\\lambda_1,\\dots,\\lambda_D)$\nfrom fit-split\nquantiles; order and\ntest fixed here", BLUE)
box(ax, xs[0], bot, 2 * w + g, h, "5. Certify (calibration split)",
    "fixed-sequence learn-then-test of $H_\\lambda:\\ \\mathrm{mFDR}(\\lambda)>\\alpha$\n"
    "with the pooled betting test or the depth-wise\nbinomial test; all certified candidates satisfy\n"
    "$\\Pr(\\mathrm{mFDR}\\leq\\alpha)\\geq 1-\\delta$ simultaneously", GREEN)
box(ax, xs[2], bot, w, h, "6. Select", "deploy the certified\n$\\lambda$ with the lowest\nestimated human\neffort $\\mathrm{E}[c]$", GREEN)
box(ax, xs[3], bot, w, h, "7. Deploy", "store the accepted\nprefix of each path;\nsend the remaining\nlevels to a reviewer", ORANGE)
for i in range(3):
    arrow(ax, (xs[i] + w + 0.002, top + h / 2), (xs[i + 1] - 0.002, top + h / 2))
arrow(ax, (xs[3] + w / 2, top - 0.004), (xs[0] + w + g / 2 + 0.06, bot + h + 0.004))
arrow(ax, (xs[1] + w + 0.002, bot + h / 2), (xs[2] - 0.002, bot + h / 2))
arrow(ax, (xs[2] + w + 0.002, bot + h / 2), (xs[3] - 0.002, bot + h / 2))

# (b) one item ----------------------------------------------------------------
ax.text(0.005, 0.282, "(c)", weight="bold", fontsize=8, va="top")
nodes = [("Medicine", "0.97", "0.90", True), ("Cardiology", "0.58", "0.80", False),
         ("Heart failure", "0.91", "0.75", False)]
bw, bh, by = 0.30, 0.10, 0.12
bx = [0.004, 0.35, 0.696]
for d, ((name, s, lam, acc), x) in enumerate(zip(nodes, bx), 1):
    ax.add_patch(FancyBboxPatch((x, by), bw, bh, boxstyle="round,pad=0.006,rounding_size=0.01",
                                fc=GREEN if acc else ORANGE, ec=EDGE, lw=0.7))
    ax.text(x + bw / 2, by + bh - 0.025, f"depth {d}: {name}", ha="center", va="top", weight="bold", fontsize=8)
    rel = "\\geq" if float(s) >= float(lam) else "<"
    ax.text(x + bw / 2, by + 0.033, f"$s={s}\\ {rel}\\ \\lambda_{d}={lam}$\n" + ("accepted" if acc else "sent to review"),
            ha="center", va="center", fontsize=8, linespacing=1.3)
    if d < 3:
        arrow(ax, (x + bw + 0.003, by + bh / 2), (bx[d] - 0.003, by + bh / 2))

OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "Figure_1.pdf", pad_inches=0.0)
fig.savefig(Path(__file__).with_name("Figure_1_preview.png"), dpi=200, pad_inches=0.0)
print("wrote", OUT / "Figure_1.pdf")
