# sdf-sf

Small-scale synthetic document finetuning (SDF) evaluation pipeline for the SPAR F26 Emergent
Alignment project (Sentient Futures). The question under test: does charter data, inserted as
pretraining-style documents into a base model, produce value shifts that survive conventional
instruction tuning?

**See [CLAUDE.md](CLAUDE.md) for the pipeline stages, arms, evaluation plan, open decisions, and
working rules.** This file only covers setup and attribution.

## Layout

```
common/        shared library (why_gen: training, corpus composition, Inspect tasks and scorers), methods, configs
data_prep/     builds the mid-training replay mix (and, optionally, the AuditBench reference corpus)
sections/
  01_auditbench_behaviour/   AuditBench model organisms (kept pending decision; its instruments/ are read by common/)
  03_fair_midtraining/       mid-training + instruction tuning + evals: the recipe our arms build on
  04_cmt_graft/instruments/  Cho et al. battery settings (kept pending the battery decision)
  06_olmo_transfer/          OLMo-3 checkpoint lineage (kept pending the alternative-base decision)
  07_emergent_misalignment/  EM replication (kept pending decision; candidate damage check)
```

## Setup

```bash
pip install -e common/                        # Python 3.10-3.12
# training and serving run in their own environments: WHY_GEN_AXOLOTL_VENV, WHY_GEN_VLLM_VENV
python -m nltk.downloader punkt_tab
python data_prep/build_midtraining_mix.py     # expects the SDF corpus in data/ (see CLAUDE.md)
# only for the AuditBench reference corpus:
git clone https://github.com/safety-research/auditing-agents external/auditing-agents
python data_prep/fetch_auditbench.py
```

Data lives in `data/` and results in `out/` at the package root.

## Attribution

This repository is a fork of the code release for:

**Pre-training interventions, ex post facto: grafting model beliefs across checkpoints**
Peter Nutter, Dani Roytburg, Clément Dumas, Jinghua Ou, Shi Feng.
Models at [huggingface.co/Grafting-Beliefs](https://huggingface.co/Grafting-Beliefs).

```bibtex
@article{nutter2026grafting,
  title  = {Pre-training interventions, ex post facto: grafting model beliefs across checkpoints},
  author = {Nutter, Peter and Roytburg, Dani and Dumas, Cl{\'e}ment and Ou, Jinghua and Feng, Shi},
  year   = {2026},
  note   = {TODO: arXiv identifier}
}
```
