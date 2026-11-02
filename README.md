# HLCL: certified label-level selective prediction for hierarchical outputs

Reproducibility package for the paper *Certified label-level selective
prediction for hierarchical outputs, with an application to LLM annotation*.

Labels are compared after mapping typographic apostrophes to plain ones
(`code/features.py`); annotator B writes "Children's" where the taxonomy has
"Children’s", and its runs were redone with this fix.

## Contents
| Path | What it is |
|---|---|
| `paper/` | manuscript source (Elsevier template, Knowledge-Based Systems), figures, generated tables and `figure_src/` scripts; `sh paper/build_flat.sh` assembles the flat folder to compile |
| `code/hlcl2.py` | acceptance rule, pooled betting test, depth-wise binomial IUT, learn-then-test calibration, effort model |
| `code/features.py` | consensus paths and node features from the cached LLM samples; logistic node score |
| `code/experiment2.py` | 100-split experiment loop, all baselines and ablations |
| `code/annotate.py` | the LLM annotation script (OpenAI-compatible API) that produced the cache |
| `code/datasets_local.py` | loaders for the four corpora with the fixed 3,000-item subsample (seed 2026) |
| `code/make_tables.py` | main tables and the numbers quoted in the paper (`paper/tables/`) |
| `paper/figure_src/*.py` | figures, per-baseline, Holm, Item-cert. and supplementary tables |
| `code/extra_shift_hypac.py` | HyPAC baseline (one model and a person) and distribution-shift checks (label shift, change of annotator) |
| `code/extra_analysis.py` | supplementary runs: error by depth, stored-label error, other cost weights, population check of the guarantee |
| `data/llm_cache/*.jsonl` | all cached LLM responses (six caches: annotator A `deepseek-chat` on four corpora; annotators B `claude-haiku-4-5` and C `gpt-5.6-luna` on the Blurb collection; k = 5, temperature 1.0) (item index, sample index, raw JSON answer) |
| `results/v9_*.json` | per-split results for every method, dataset, annotator and alpha (Item-cert. fixed-test variants merged in) |
| `results/v9item_*.json` | Item-cert. under each fixed test, on the same splits; merged by `code/merge_item.py` |
| `results/v10extra_*.json` | output of `code/extra_analysis.py` |
| `results/v11shift_*.json` | output of `code/extra_shift_hypac.py` |
| `code/build_amazon.py` | builds the four-level Amazon subset (see `data/amazon/README.md`) |

## Source corpora (not redistributed)
Place them under `data/` as follows.
* DBPedia classes: `DBPEDIA_val.csv` from https://huggingface.co/datasets/DeveloperOats/DBPedia_Classes
  (the loader uses its first 11,151 records, the frame from which the published runs were sampled)
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
python make_tables.py v9
python extra_analysis.py            # supplementary runs (results/v10extra_*.json)
python extra_shift_hypac.py         # HyPAC baseline and shift checks (results/v11shift_*.json)
cd ../paper/figure_src
for f in make_figure_1.py make_figures_2_3.py make_figure_5.py make_perbaseline.py \
         make_itemcert_summary.py make_holm.py make_fdp_ratio.py make_supplement.py; do python $f; done
```
Single splits can differ across machines because some candidate p-values tie up to
floating-point error, which changes the fixed-sequence order; in our checks this
changed 48 of 2,400 HLCL splits and mean effort by at most 0.008 (DBPedia, alpha = 0.02).
The three scripts run all settings in the paper; they expect a virtualenv at `.venv/`. No API key is needed: the
experiments read the cached responses. To re-annotate, set the API key in
`~/.config/rp/llm.env` and run `annotate.py`.

## Licence
Code: MIT. Cached responses: CC BY 4.0. The source corpora keep their own licences.
