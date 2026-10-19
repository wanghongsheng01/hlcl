"""Turn cached LLM samples into per-node scores for one dataset.

For each item we build the *consensus path*: at depth 0 the most voted label;
at depth d the most voted child of the chosen depth-(d-1) label among samples
that agree with the path so far. Each node on the path gets
  vote  = share of all k samples whose path agrees with the consensus up to d
  verb  = mean verbalised confidence at depth d over those agreeing samples
  score = vote + 0.01 * verb   (verbal confidence only breaks vote ties)
A node is correct iff the gold path agrees with the consensus up to depth d.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

from annotate import CACHE, parse
from datasets_local import load_dataset_items
from hlcl import Annotated


def build(dataset: str, tag: str, n: int, k: int):
    items, tree = load_dataset_items(dataset, n)
    depth = len(items[0]["gold"])
    samples: dict[int, list] = {}
    for line in (CACHE / f"{tag}.jsonl").read_text().splitlines():
        r = json.loads(line)
        if r["s"] >= k:
            continue
        p = parse(r["resp"], depth)
        samples.setdefault(r["id"], []).append(p)

    item, dep, score, vote, verb, correct, parent = [], [], [], [], [], [], []
    n_invalid = 0
    for it in items:
        ss = [s for s in samples.get(it["id"], []) if s is not None]
        n_invalid += k - len(ss)
        prefix: list[str] = []
        prev = -1
        for d in range(depth):
            agree = [s for s in ss if s[0][:d] == prefix]
            if not agree:
                break
            lab, _ = Counter(s[0][d] for s in agree).most_common(1)[0]
            prefix = prefix + [lab]
            sup = [s for s in agree if s[0][d] == lab]
            v = len(sup) / k
            c = float(np.mean([s[1][d] for s in sup]))
            item.append(it["id"]); dep.append(d); vote.append(v); verb.append(c)
            score.append(v + 0.01 * c)
            correct.append(int(it["gold"][: d + 1] == prefix))
            parent.append(prev)
            prev = len(item) - 1
    a = Annotated(np.array(item), np.array(dep), np.array(score),
                  np.array(correct), np.array(parent), len(items))
    meta = dict(n_items=len(items), depth=depth, invalid_samples=n_invalid,
                acc_by_depth=[float(np.mean(a.correct[a.depth == d])) for d in range(depth)])
    return a, np.array(vote), np.array(verb), meta
