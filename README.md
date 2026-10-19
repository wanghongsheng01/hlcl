# HLCL: label-level hierarchical selective acceptance of LLM annotations

Reproducibility package for the paper *Label-Level Hierarchical Selective
Acceptance of LLM Annotations with Certified Error Control*.

## Contents
| Path | What it is |
|---|---|
| `paper/` | manuscript source (elsarticle), figures and generated tables |
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
