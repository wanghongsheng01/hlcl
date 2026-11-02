"""Per-baseline comparison, across-setting rank tests, the critical-difference diagram and the
catalog-scale projection, all from the released per-split results (no new experiment).

Writes
  tables/perbaseline.tex        macros \\stat{s@pb@<b>@...}, \\stat{s@inval@<m>}, \\stat{s@fr@...}, \\stat{s@proj@...}
  tables/perbaseline_table.tex  body of the per-baseline table
  figures/Figure_4.pdf          critical-difference diagram of the certified methods
Run with the environment of the code repository:  python make_perbaseline.py
"""
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# the code repository is either the parent of this paper folder (paper/ inside the
# repository) or its sibling hlcl-code/ (separate working copy)
_P = Path(__file__).resolve().parents[1]
CODE = _P.parent if (_P.parent / "results").is_dir() else _P.parent / "hlcl-code"
ROOT = _P
sys.path.insert(0, str(CODE / "code"))
import make_tables as mt  # noqa: E402

plt.rcParams.update({"font.size": 8, "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"], "mathtext.fontset": "dejavuserif",
                     "pdf.fonttype": 42, "axes.linewidth": 0.6})

SETTINGS = [(k, lab) for k, lab in mt.SETTINGS if (mt.ROOT / "results" / f"{mt.VER}_{k}.json").exists()]
mt.SETTINGS = SETTINGS
data = {k: mt.load(k) for k, _ in SETTINGS}
mt.add_item_best(data)
BASE = [("item", "ItemBest", r"Item-cert.$^\ast$", "high-prob., same certificate, per item"),
        ("hsc", "HSC", "HSC", "high-prob., item-level accuracy"),
        ("pf", "PerField-LTT", "Per-field", "high-prob., per depth"),
        ("conf", "ConfLabel", "Item-conf.", "expected item-level FDR"),
        ("dbh", "DepthBH", "Depth-BH", "expected FDR per depth"),
        ("fix", "Conf>=0.9", r"$s\ge0.9$", "none")]
CERT = ["HLCL", "ItemBest", "HSC", "PerField-LTT"]
CERT_LABEL = {"HLCL": "HLCL", "ItemBest": "Item-cert.*", "HSC": "HSC", "PerField-LTT": "Per-field"}
POOL, N0 = 154787, 1800


def test(R, m1, m2="HLCL"):
    """Paired effort difference m2 - m1 with the Nadeau-Bengio-corrected 95% interval."""
    d = mt.vals(R, m2, "effort") - mt.vals(R, m1, "effort")
    se = math.sqrt(mt.NB_FACTOR * d.var(ddof=1))
    if se == 0:
        return float(d.mean()), 0.0, 1.0 if d.mean() == 0 else 0.0
    z = d.mean() / se
    return float(d.mean()), 1.96 * se, math.erfc(abs(z) / math.sqrt(2))


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def viol_share(R, m, a):
    return float(np.mean(mt.vals(R, m, "fdp") > a + 1e-12))


rows = []
ABL = [("vote", "HLCL", "HLCL-vote"), ("flat", "HLCL", "HLCL-flat"), ("tied", "HLCL", "HLCL-tied")]
for key, lab in SETTINGS:
    runs = data[key]["runs"]
    for a in mt.alphas(runs):
        R = mt.at(runs, a)
        ms = ["HLCL"] + [b for _, b, _, _ in BASE]
        row = {"key": key, "alpha": a, "R": R,
               "eff": {m: mt.mean(R, m, "effort") for m in ms},
               "valid": {m: mt.viol(R, m, a) <= mt.DELTA for m in ms},
               "share": {m: viol_share(R, m, a) for m in ms},
               "t": {b: test(R, b) for _, b, _, _ in BASE}, "abl": {}}
        for v, full, var in ABL:
            for g in ("", "-wsr", "-iut"):
                if full + g in R[0] and var + g in R[0]:
                    row["abl"][(v, g)] = test(R, var + g, full + g)
        rows.append(row)

mac, tab = {}, []
mac["s@pb@N"] = str(len(rows))
for m in ["HLCL"] + [b for _, b, _, _ in BASE]:
    mac[f"s@inval@{m}"] = str(sum(not r["valid"][m] for r in rows))
    mac[f"s@inval20@{m}"] = str(sum(r["share"][m] > 0.2 for r in rows))


def wl(idx, b):
    win = sum(r["t"][b][0] + r["t"][b][1] < 0 for r in idx)
    loss = sum(r["t"][b][0] - r["t"][b][1] > 0 for r in idx)
    return win, loss


for short, b, label, guar in BASE:
    idx = [r for r in rows if r["valid"][b]]
    adj = holm([r["t"][b][2] for r in idx])
    win, loss = wl(idx, b)
    hw = sum(p < 0.05 and r["t"][b][0] < 0 for p, r in zip(adj, idx))
    hl = sum(p < 0.05 and r["t"][b][0] > 0 for p, r in zip(adj, idx))
    eh, eb = np.mean([r["eff"]["HLCL"] for r in idx]), np.mean([r["eff"][b] for r in idx])
    rel = 100 * (eb - eh) / eb
    aw, al = wl(rows, b)
    ah, ab = np.mean([r["eff"]["HLCL"] for r in rows]), np.mean([r["eff"][b] for r in rows])
    arel = 100 * (ab - ah) / ab
    excl = [r for r in rows if not r["valid"][b]]
    for k_, v in {"valid": len(idx), "wins": win, "losses": loss, "ties": len(idx) - win - loss,
                  "holmwins": hw, "holmlosses": hl, "effH": f"{eh:.2f}", "effB": f"{eb:.2f}",
                  "rel": f"{rel:.0f}", "allwins": aw, "alllosses": al, "allrel": f"{arel:.0f}",
                  "exclcheaper": sum(r["eff"][b] < r["eff"]["HLCL"] for r in excl)}.items():
        mac[f"s@pb@{short}@{k_}"] = str(v)
    tab.append(f"{label} & {guar} & {len(idx)} & {win}/{loss}/{len(idx) - win - loss} & {hw}/{hl} & "
               f"{eh:.2f} & {eb:.2f} & {rel:.0f}\\% & {aw}/{al} & {arel:.0f}\\% \\\\")
    if short == "pf":
        tab.append("\\midrule")
for k_ in ("rel", "wins", "holmwins"):
    vals = [int(mac[f"s@pb@{s}@{k_}"]) for s in ("item", "hsc", "pf")]
    mac[f"s@pb@{k_}min"], mac[f"s@pb@{k_}max"] = str(min(vals)), str(max(vals))
mac["s@pb@relmin"], mac["s@pb@relmax"] = mac.pop("s@pb@relmin"), mac.pop("s@pb@relmax")
mac["s@pb@winmin"], mac["s@pb@winmax"] = mac.pop("s@pb@winsmin"), mac.pop("s@pb@winsmax")
mac["s@pb@lossmax"] = str(max(int(mac[f"s@pb@{s}@losses"]) for s in ("item", "hsc", "pf")))
inval = [int(mac[f"s@inval@{b}"]) for _, b, _, _ in BASE if int(mac[f"s@inval@{b}"]) > 0]
mac["s@inval@nbase"], mac["s@inval@min"], mac["s@inval@max"] = str(len(inval)), str(min(inval)), str(max(inval))
mac["s@cheaper@cert"] = str(sum(min(r["eff"][m] for m in CERT[1:] if r["valid"][m]) < r["eff"]["HLCL"]
                                for r in rows if any(r["valid"][m] for m in CERT[1:])))

# Rank analysis. Settings of one corpus-annotator pair share items and annotations, so the
# Friedman test is also run with these pairs as the independent units (mean effort per pair).
q05 = {2: 1.960, 3: 2.343, 4: 2.569, 5: 2.728}
k = len(CERT)


def friedman(M):
    M = np.array(M)
    p = stats.friedmanchisquare(*[M[:, j] for j in range(k)]).pvalue
    return M.mean(axis=0), p, q05[k] * math.sqrt(k * (k + 1) / (6 * len(M)))


ranks, p_fr, cd = friedman([stats.rankdata([r["eff"][m] + (10.0 if not r["valid"][m] else 0.0) for m in CERT])
                            for r in rows])
ranks_e, p_fr_e, _ = friedman([stats.rankdata([r["eff"][m] for m in CERT]) for r in rows])
blocks = sorted({r["key"] for r in rows}, key=[s_[0] for s_ in SETTINGS].index)
ranks_b, p_fr_b, cd_b = friedman([stats.rankdata([np.mean([r["eff"][m] for r in rows if r["key"] == bk])
                                                  for m in CERT]) for bk in blocks])
fmtp = lambda p: "<0.001" if p < 0.001 else f"={p:.3f}"  # noqa: E731
nxt = min(range(1, k), key=lambda j: ranks_e[j])
mac["s@fr@enext@name"] = CERT_LABEL[CERT[nxt]].replace("*", "$^\\ast$")
mac["s@fr@enext"] = f"{ranks_e[nxt]:.2f}"
mac["s@fr@enext@sig"] = "exceeds" if ranks_e[nxt] - ranks_e[0] > cd else "is below"
mac["s@fr@ncd"] = str(sum(ranks[j] - ranks[0] > cd for j in range(1, k)))
mac["s@fr@b@ncd"] = str(sum(ranks_b[j] - ranks_b[0] > cd_b for j in range(1, k)))
for j, m in enumerate(CERT):
    mac[f"s@fr@rank@{m}"] = f"{ranks[j]:.2f}"
    mac[f"s@fr@erank@{m}"] = f"{ranks_e[j]:.2f}"
    mac[f"s@fr@brank@{m}"] = f"{ranks_b[j]:.1f}"
mac["s@fr@p"], mac["s@fr@ep"], mac["s@fr@bp"] = fmtp(p_fr), fmtp(p_fr_e), fmtp(p_fr_b)
mac["s@fr@cd"], mac["s@fr@N"], mac["s@fr@bcd"], mac["s@fr@bN"] = f"{cd:.2f}", str(len(rows)), f"{cd_b:.2f}", str(len(blocks))

# Ablations: number of settings with a significant paired difference (either direction).
for v, _, _ in ABL:
    for g, gname in (("", "auto"), ("-wsr", "wsr"), ("-iut", "iut")):
        t = [r["abl"][(v, g)] for r in rows if (v, g) in r["abl"]]
        mac[f"s@abl@{v}@{gname}@sig"] = str(sum(abs(d_) > h for d_, h, _ in t))

# Projection to the Amazon pool (the 3,000 items are a random sample of it).
for a in (0.05, 0.1):
    r = next(x for x in rows if x["key"] == "amazon" and abs(x["alpha"] - a) < 1e-9)
    e = r["eff"]["HLCL"]
    n_out = POOL - N0
    mac[f"s@proj@{a:g}@total"] = f"{(N0 + e * n_out) / POOL:.2f}"
    for short, b, _, _ in BASE[:3]:
        d_, h, _ = test(r["R"], b)
        fmtn = lambda x: f"{round(x):,}".replace(",", "{,}")  # noqa: E731
        mac[f"s@proj@{a:g}@{short}"] = fmtn(-d_ * n_out)
        mac[f"s@proj@{a:g}@{short}@lo"] = fmtn((-d_ - h) * n_out)
        mac[f"s@proj@{a:g}@{short}@hi"] = fmtn((-d_ + h) * n_out)
        mac[f"s@proj@{a:g}@{short}@sig"] = "yes" if abs(d_) > h else "no"
        mac[f"s@proj@{a:g}@{short}@eff"] = f"{r['eff'][b]:.2f}"
    q = np.percentile(mt.vals(r["R"], "HLCL", "effort"), [5, 95])
    mac[f"s@proj@{a:g}@hlcl@eff"] = f"{e:.2f}"
    mac[f"s@proj@{a:g}@hlcl@p5"], mac[f"s@proj@{a:g}@hlcl@p95"] = f"{q[0]:.2f}", f"{q[1]:.2f}"
mac["s@proj@nout"] = f"{POOL - N0:,}".replace(",", "{,}")

(ROOT / "tables" / "perbaseline.tex").write_text(
    "% generated by figure_src/make_perbaseline.py from the released per-split result files\n"
    + "".join(f"\\expandafter\\def\\csname {k}\\endcsname{{{v}}}\n" for k, v in mac.items()))
(ROOT / "tables" / "perbaseline_table.tex").write_text(
    "% generated by figure_src/make_perbaseline.py from the released per-split result files\n"
    "\\begin{tabularx}{\\textwidth}{>{\\raggedright\\arraybackslash}p{1.45cm} >{\\raggedright\\arraybackslash}X "
    "cccccccc}\n\\toprule\n & & \\multicolumn{6}{c}{settings in which the baseline is on target} & "
    "\\multicolumn{2}{c}{all settings} \\\\\n\\cmidrule(lr){3-8}\\cmidrule(lr){9-10}\n"
    "Baseline & Guarantee & $n$ & W/L/T & Holm W/L & \\hlcl{} & baseline & red. & W/L & red. \\\\\n"
    "\\midrule\n" + "\n".join(tab) + "\n\\bottomrule\n\\end{tabularx}\n")

# Critical-difference diagram (Demsar, 2006): average ranks, methods joined by a bar when their
# rank difference is below the Nemenyi critical difference.
fig, ax = plt.subplots(figsize=(5.15, 1.25))
lo, hi = 1, k
ax.set_xlim(lo - 0.15, hi + 0.15)
ax.set_ylim(0.26, 1.08)
ax.axis("off")
y0 = 0.78
ax.plot([lo, hi], [y0, y0], color="black", lw=0.7)
for t in np.arange(lo, hi + 0.01, 0.5):
    ax.plot([t, t], [y0, y0 + (0.05 if t == int(t) else 0.03)], color="black", lw=0.6)
    if t == int(t):
        ax.text(t, y0 + 0.08, f"{int(t)}", ha="center", va="bottom")
ax.plot([lo, lo + cd], [1.0, 1.0], color="black", lw=1.0)
for xx in (lo, lo + cd):
    ax.plot([xx, xx], [0.98, 1.02], color="black", lw=1.0)
ax.text(lo + cd + 0.05, 1.0, f"CD = {cd:.2f}", ha="left", va="center", fontsize=8)
order = np.argsort(ranks)
for i, j in enumerate(order):
    x = ranks[j]
    left = i < k / 2
    yy = 0.55 - 0.17 * (i if left else k - 1 - i)
    xt = lo - 0.1 if left else hi + 0.1
    ax.plot([x, x, xt], [y0, yy, yy], color="black", lw=0.6)
    ax.text(xt + (-0.03 if left else 0.03), yy, f"{CERT_LABEL[CERT[j]]} ({x:.2f})",
            ha="right" if left else "left", va="center",
            fontweight="bold" if CERT[j] == "HLCL" else "normal")
grp, s = [], sorted(ranks)
for i in range(k):
    j = max(jj for jj in range(k) if s[jj] - s[i] < cd)
    if j > i and not any(g[0] <= i and g[1] >= j for g in grp):
        grp.append((i, j))
for n, (i, j) in enumerate(grp):
    yb = y0 - 0.08 - 0.06 * n
    ax.plot([s[i] - 0.03, s[j] + 0.03], [yb, yb], color="black", lw=2.0)
fig.subplots_adjust(left=0.2, right=0.8, top=1.0, bottom=0.0)
fig.savefig(ROOT / "figures" / "Figure_4.pdf")
fig.savefig(ROOT / "figure_src" / "Figure_4_preview.png", dpi=200)
print({k_: v for k_, v in mac.items() if not k_.startswith("s@proj")})
print({k_: v for k_, v in mac.items() if k_.startswith("s@proj")})
