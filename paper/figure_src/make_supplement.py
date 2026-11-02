"""Supplementary statistics from the released per-split results (results/v9_*.json)
and from the supplementary runs (results/v10extra_*.json, code/extra_analysis.py).

Writes tables/supp.tex (macros) and the tables
  tables/tied_table.tex      unit of decision alone: HLCL-tied - Item-cert.*
  tables/realized_table.tex  mean realized label-level error / alpha per method
  tables/depth_table.tex     per-depth error and power, stored-label error (v10extra)
  tables/popcheck_table.tex  population check of the guarantee (v10extra)
  tables/costsens_table.tex  effort difference under other cost weights (v10extra)
"""
import json
import math
from pathlib import Path

import numpy as np

# the code repository is either the parent of this paper folder (paper/ inside the
# repository) or its sibling hlcl-code/ (separate working copy)
_P = Path(__file__).resolve().parents[1]
CODE = _P.parent if (_P.parent / "results").is_dir() else _P.parent / "hlcl-code"
ROOT = _P
RES = CODE / "results"
OUT = ROOT / "tables"
NB = 1 / 100 + 1200 / 1800
SET = [("dbpedia", "DBPedia (A)"), ("wos", "WOS-46985 (A)"), ("bgc", "Blurb (A)"),
       ("bgc_haiku", "Blurb (B)"), ("bgc_gpt", "Blurb (C)"), ("amazon", "Amazon (A)")]
ITEM = ["ItemLTT", "ItemLTT-wsr", "ItemLTT-iut"]
MAIN = ["HLCL", "ItemBest", "HSC", "PerField-LTT", "ConfLabel", "DepthBH", "Conf>=0.9"]
HEAD = ["\\hlcl{}", "Item-cert.$^\\ast$", "HSC", "Per-field", "Item-conf.", "Depth-BH", "$s\\ge0.9$"]
mac = {}


def put(k, v):
    mac[k] = v


def paired(x, y):
    d = np.asarray(x) - np.asarray(y)
    h = 1.96 * math.sqrt(NB * d.var(ddof=1)) if len(d) > 1 else 0.0
    return float(d.mean()), h


def holm(items, good_sign):
    ps = sorted((math.erfc(abs(m / (h / 1.96)) / math.sqrt(2)) if h > 0 else (1.0 if m == 0 else 0.0), m)
                for m, h in items)
    good = bad = 0
    for i, (p, m) in enumerate(ps):
        if p * (len(ps) - i) > 0.05:
            break
        good, bad = (good + 1, bad) if (m > 0) == good_sign else (good, bad + 1)
    return good, bad


def load(key):
    runs = json.loads((RES / f"v9_{key}.json").read_text())["runs"]
    out = {}
    for a in sorted({r["alpha"] for r in runs}):
        R = [r for r in runs if r["alpha"] == a]
        b = min(ITEM, key=lambda m: np.mean([r[m]["effort"] for r in R]))
        for r in R:
            r["ItemBest"] = r[b]
        out[a] = R
    return out


data = {k: load(k) for k, _ in SET}
f2 = lambda v: f"{v:.2f}"

# 1. unit of decision alone: tied (one shared threshold, automatic test) against Item-cert.*
cnt = {"power": [0, 0], "effort": [0, 0]}
items = {"power": [], "effort": []}
rows = []
for key, name in SET:
    for j, (a, R) in enumerate(data[key].items()):
        row = [name if j == 0 else "", f"{a:g}"]
        for k, good in (("power", True), ("effort", False)):
            m, h = paired([r["HLCL-tied"][k] for r in R], [r["ItemBest"][k] for r in R])
            items[k].append((m, h))
            better = (m - h > 0) if good else (m + h < 0)
            worse = (m + h < 0) if good else (m - h > 0)
            cnt[k][0] += better
            cnt[k][1] += worse
            s = f"{m:+.3f}$\\pm${h:.3f}"
            row.append("\\textbf{" + s + "}" if better else s)
        rows.append(" & ".join(row) + " \\\\")
for k, good in (("power", True), ("effort", False)):
    put(f"s@tied@{k}@better", cnt[k][0])
    put(f"s@tied@{k}@worse", cnt[k][1])
    g, b = holm(items[k], good)
    put(f"s@tied@{k}@holm", g)
tab = ["\\begin{tabular}{llcc}", "\\toprule", "Setting & $\\alpha$ & power & effort \\\\", "\\midrule"]
for i, r in enumerate(rows):
    if i and i % 4 == 0:
        tab.append("\\addlinespace")
    tab.append(r)
tab += ["\\bottomrule", "\\end{tabular}"]
(OUT / "tied_table.tex").write_text("\n".join(tab) + "\n")

# 2. mean realized error / alpha
tab = ["\\begin{tabular}{ll" + "c" * len(MAIN) + "}", "\\toprule",
       "Setting & $\\alpha$ & " + " & ".join(HEAD) + " \\\\", "\\midrule"]
ratios = {m: [] for m in MAIN}
for i, (key, name) in enumerate(SET):
    if i:
        tab.append("\\addlinespace")
    for j, (a, R) in enumerate(data[key].items()):
        cells = []
        for m in MAIN:
            v = np.mean([r[m]["fdp"] for r in R]) / a
            ratios[m].append(v)
            cells.append(f2(v))
        tab.append(" & ".join([name if j == 0 else "", f"{a:g}"] + cells) + " \\\\")
tab += ["\\bottomrule", "\\end{tabular}"]
(OUT / "realized_table.tex").write_text("\n".join(tab) + "\n")
for m in MAIN:
    put(f"s@ratio@{m}@med", f2(float(np.median(ratios[m]))))

# 3. median relative reduction against each certified baseline where it is on target
for m in ("ItemBest", "HSC", "PerField-LTT"):
    red = []
    for key, _ in SET:
        for a, R in data[key].items():
            if np.mean([r[m]["fdp"] > a + 1e-12 for r in R]) <= 0.1 + 1e-12:
                b = np.mean([r[m]["effort"] for r in R])
                red.append((b - np.mean([r["HLCL"]["effort"] for r in R])) / b if b > 0 else 0.0)
    put(f"s@medred@{m}", f"{100 * np.median(red):.0f}")

# 4. supplementary runs
extra = {k: json.loads((RES / f"v10extra_{k}.json").read_text()) for k, _ in SET
         if (RES / f"v10extra_{k}.json").exists()}
if extra:
    depth_rows, pop_rows, cost_rows = [], [], []
    worst_q, worst_stored, pops = [], [], {m: [] for m in ("HLCL", "ItemLTT", "HSC", "PerField-LTT")}
    cost_cnt = {"effort": [0, 0], "eff_lin": [0, 0], "eff_c2": [0, 0]}
    for key, name in SET:
        if key not in extra:
            continue
        for j, a in enumerate(sorted(extra[key], key=float)):
            al = float(a)
            sp, pp = extra[key][a]["split"], extra[key][a]["pop"]
            D = sum(1 for k in sp[0]["HLCL"] if k.startswith("q"))
            qs = []
            for d in range(1, D + 1):
                v = [s["HLCL"][f"q{d}"] for s in sp if s["HLCL"][f"q{d}"] is not None]
                qs.append(np.mean(v) / al if v else float("nan"))
            pw = [np.mean([s["HLCL"][f"pow{d}"] for s in sp]) for d in range(1, D + 1)]
            st = np.mean([s["HLCL"]["stored"] for s in sp]) / al
            worst_q.append(qs[-1] if not math.isnan(qs[-1]) else 0)
            worst_stored.append(st)
            depth_rows.append(" & ".join([name if j == 0 else "", f"{al:g}",
                                          " / ".join("--" if math.isnan(q) else f2(q) for q in qs),
                                          " / ".join(f2(p) for p in pw), f2(st)]) + " \\\\")
            cells = []
            for m in pops:
                v = np.array([p[m] for p in pp])
                pops[m].append(float(np.mean(v > al + 1e-12)))
                cells.append(f"{100 * np.mean(v > al + 1e-12):.0f}\\%")
            pop_rows.append(" & ".join([name if j == 0 else "", f"{al:g}"] + cells) + " \\\\")
            ib = min(ITEM, key=lambda m: np.mean([s[m]["effort"] for s in sp]))
            cells = []
            for k in ("effort", "eff_lin", "eff_c2"):
                m_, h = paired([s["HLCL"][k] for s in sp], [s[ib][k] for s in sp])
                win, loss = m_ + h < 0, m_ - h > 0
                cost_cnt[k][0] += win
                cost_cnt[k][1] += loss
                c = f"{m_:+.3f}$\\pm${h:.3f}"
                cells.append("\\textbf{" + c + "}" if win else c)
            cost_rows.append(" & ".join([name if j == 0 else "", f"{al:g}"] + cells) + " \\\\")

    def table(head, rows, cols):
        t = ["\\begin{tabular}{ll" + cols + "}", "\\toprule", head + " \\\\", "\\midrule"]
        for i, r in enumerate(rows):
            if i and i % 4 == 0:
                t.append("\\addlinespace")
            t.append(r)
        return "\n".join(t + ["\\bottomrule", "\\end{tabular}"]) + "\n"

    (OUT / "depth_table.tex").write_text(table(
        "Setting & $\\alpha$ & $q_d/\\alpha$, $d=1,\\dots,D$ & power by depth & stored$/\\alpha$",
        depth_rows, "ccc"))
    (OUT / "popcheck_table.tex").write_text(table(
        "Setting & $\\alpha$ & \\hlcl{} & Item-cert. & HSC & Per-field", pop_rows, "cccc"))
    (OUT / "costsens_table.tex").write_text(table(
        "Setting & $\\alpha$ & default & $\\omega_d\\propto d$ & $c_0=2$", cost_rows, "ccc"))
    put("s@extra@nsettings", len(depth_rows))
    put("s@depth@finalmax", f2(max(worst_q)))
    put("s@stored@max", f2(max(worst_stored)))
    for m, v in pops.items():
        put(f"s@pop@{m}@max", f"{100 * max(v):.0f}\\%")
    for k, (w, l) in cost_cnt.items():
        put(f"s@cost@{k}@wins", w)
        put(f"s@cost@{k}@losses", l)

(OUT / "supp.tex").write_text("% generated by figure_src/make_supplement.py\n" + "\n".join(
    f"\\expandafter\\def\\csname {k}\\endcsname{{{v}}}" for k, v in mac.items()) + "\n")
print("\n".join(f"{k} = {v}" for k, v in mac.items()))
