"""Merge the Item-cert. fixed-test runs (v9item_<name>.json, from
results/run_v9_item.sh and run_v9_amazon.sh) into the main result files
(v9_<name>.json). Runs are matched on (split, alpha); the script refuses to
merge if the two files do not cover the same splits.

Usage: python merge_item.py [VER]   (default v9)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = ("ItemLTT-wsr", "ItemLTT-iut")


def merge(main_path, item_path):
    main = json.loads(main_path.read_text())
    item = json.loads(item_path.read_text())
    idx = {(r["split"], r["alpha"]): r for r in item["runs"]}
    if len(idx) != len(main["runs"]):
        sys.exit(f"{item_path.name}: {len(idx)} runs, main has {len(main['runs'])}")
    for r in main["runs"]:
        src = idx.get((r["split"], r["alpha"]))
        if src is None:
            sys.exit(f"{item_path.name}: missing split {r['split']} alpha {r['alpha']}")
        for v in VARIANTS:
            r[v] = src[v]
    main_path.write_text(json.dumps(main))
    print(f"merged {item_path.name} -> {main_path.name}")


def main():
    ver = sys.argv[1] if len(sys.argv) > 1 else "v9"
    res = ROOT / "results"
    for item_path in sorted(res.glob(f"{ver}item_*.json")):
        name = item_path.name[len(f"{ver}item_"):]
        merge(res / f"{ver}_{name}", item_path)


if __name__ == "__main__":
    main()
