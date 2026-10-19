#!/bin/sh
# Build the reproducibility package (code + cached LLM responses + per-split
# results + scripts) into submission/hlcl_artifact/. Source corpora are NOT
# redistributed: download instructions are in the README.
set -e
H="$(cd "$(dirname "$0")/.." && pwd)"
OUT="$H/submission/hlcl_artifact"
VER="${1:-v9}"
rm -rf "$OUT"; mkdir -p "$OUT/code" "$OUT/data/llm_cache" "$OUT/results" "$OUT/paper"
cp "$H"/code/hlcl2.py "$H"/code/features.py "$H"/code/experiment2.py "$H"/code/annotate.py \
   "$H"/code/datasets_local.py "$H"/code/hlcl.py "$H"/code/make_tables.py "$H"/code/make_figures.py \
   "$H"/code/merge_item.py "$H"/code/build_amazon.py "$OUT/code/"
cp "$H"/data/llm_cache/*.jsonl "$OUT/data/llm_cache/"
cp "$H"/results/"$VER"_*.json "$H"/results/"$VER"item_*.json "$OUT/results/"
cp "$H"/results/run_"$VER".sh "$H"/results/run_"$VER"_item.sh "$H"/results/run_"$VER"_amazon.sh "$OUT/results/"
mkdir -p "$OUT/data/amazon" && cp "$H"/data/amazon/README.md "$OUT/data/amazon/"
cp -R "$H"/paper/tables "$H"/paper/figures "$OUT/paper/"
cat > "$OUT/requirements.txt" <<EOF
numpy>=2.0
scipy>=1.13
scikit-learn>=1.5
pandas>=2.2
openpyxl>=3.1
matplotlib>=3.9
requests>=2.32
EOF
cat > "$OUT/README.md" <<'EOF'
# HLCL: label-level hierarchical selective acceptance of LLM annotations

Reproducibility package for the paper *Label-Level Hierarchical Selective
Acceptance of LLM Annotations with Certified Error Control*.

## Contents
| Path | What it is |
|---|---|
| `code/hlcl2.py` | acceptance rule, pooled betting test, depth-wise binomial IUT, learn-then-test calibration, effort model |
| `code/features.py` | consensus paths and node features from the cached LLM samples; logistic node score |
| `code/experiment2.py` | 100-split experiment loop, all baselines and ablations |
| `code/annotate.py` | the LLM annotation script (OpenAI-compatible API) that produced the cache |
| `code/datasets_local.py` | loaders for the four corpora with the fixed 3,000-item subsample (seed 2026) |
| `code/make_tables.py`, `code/make_figures.py` | regenerate every table, figure and number in the paper |
| `data/llm_cache/*.jsonl` | all cached LLM responses (five caches) (item index, sample index, raw JSON answer) |
| `results/v9_*.json` | per-split results for every method, dataset, annotator and alpha (Item-cert. fixed-test variants merged in) |
| `results/v9item_*.json` | Item-cert. under each fixed test, on the same splits; merged by `code/merge_item.py` |
| `code/build_amazon.py` | builds the four-level Amazon subset (see `data/amazon/README.md`) |

## Source corpora (not redistributed)
Place them under `data/` as follows.
* DBPedia classes: `DBPEDIA_val.csv` from https://huggingface.co/datasets/DeveloperOats/DBPedia_Classes
* WOS-46985: Mendeley Data `9rw3vkcfy4` (version 6); unzip to `data/wos/` (needs `Meta-data/Data.xlsx`)
* Blurb Genre Collection (EN), Universität Hamburg, CC BY-NC 4.0; unzip to `data/bgc/`
  (needs `BlurbGenreCollection_EN_test.txt` and `hierarchy.txt`)
* Amazon Reviews 2023, Home & Kitchen metadata: `raw/meta_categories/meta_Home_and_Kitchen.jsonl`
  from https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023; then run
  `python code/build_amazon.py` (writes `data/amazon/amazon_hk_l4_min100.json`)

## Reproduce
```bash
pip install -r requirements.txt
cd code
sh ../results/run_v9.sh          # main runs and ablations
sh ../results/run_v9_item.sh     # Item-cert. under each fixed test
sh ../results/run_v9_amazon.sh   # fourth corpus (main + Item-cert.)
python merge_item.py v9
python make_tables.py v9 && python make_figures.py v9
```
The three scripts run all settings in the paper; they expect a virtualenv at `.venv/`. No API key is needed: the
experiments read the cached responses. To re-annotate, set the API key in
`~/.config/rp/llm.env` and run `annotate.py`.

## Licence
Code: MIT. Cached responses: CC BY 4.0. The source corpora keep their own licences.
EOF
cat > "$OUT/LICENSE" <<'EOF'
MIT License

Copyright (c) 2026 Hongsheng Wang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
EOF
# safety: no secrets in the package
if grep -rIl -E "sk-[A-Za-z0-9]{16,}" "$OUT"; then echo "SECRET FOUND - aborting"; exit 1; fi
( cd "$H/submission" && rm -f hlcl_artifact.zip && zip -qr hlcl_artifact.zip hlcl_artifact )
echo "built $OUT ($(du -sh "$OUT" | cut -f1))"
