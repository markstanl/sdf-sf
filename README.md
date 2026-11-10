# sdf-sf

Small-scale synthetic document finetuning (SDF) evaluation pipeline for the SPAR F26 Emergent
Alignment project (Sentient Futures). The question under test: does charter data, inserted as
pretraining-style documents into a base model, produce value shifts that survive conventional
instruction tuning?

**See [CLAUDE.md](CLAUDE.md) for the pipeline stages, arms, evaluation plan, open decisions, and
working rules.** This file only covers setup and attribution.

## Layout

```
why_gen/       shared library (eval orchestration, Inspect tasks and scorers, model/data registries)
methods/       weight-space methods from upstream (merge_task_vector.py for the optional G arm, probes)
configs/       all configs: charter.pipeline.yaml, corpora.yaml, sft.yaml, sdf/ (native axolotl, one
               per arm x seed), models/, train/, inference/, infra/, evals/ (suite presets)
pipelines/     orchestrators: run_pipeline.py (corpora -> sdf -> sft -> evals), build_corpora.py
sft/           stage 3 instruction tuning: prepare_instruction_data.py, instruction_tune.py
evals/         stage 4: run_battery.py, score_rubric.py, run_mu.sh, manifests/, instruments/
analysis/      figures and tables from eval results, eval aggregation helpers
docs/          upstream_recipe.md: the upstream fair-midtraining recipe our arms build on
data_prep/     builds the mid-training replay mix (and, optionally, the AuditBench reference corpus)
documents/     the charters
tests/         pytest suite
```

## Setup

```bash
pip install -e .                              # Python 3.10-3.12
# training and serving run in their own environments: WHY_GEN_AXOLOTL_VENV, WHY_GEN_VLLM_VENV
python -m nltk.downloader punkt_tab
python data_prep/build_midtraining_mix.py     # expects the SDF corpus in data/ (see CLAUDE.md)
# only for the AuditBench reference corpus:
git clone https://github.com/safety-research/auditing-agents external/auditing-agents
python data_prep/fetch_auditbench.py
```

Data lives in `data/` and results in `out/` at the repo root.

## Running the pipeline

The driver sequences all stages from one manifest (run it per stage, or end to end):

```bash
python pipelines/run_pipeline.py configs/charter.pipeline.yaml --sanity --dry-run
python pipelines/run_pipeline.py configs/charter.pipeline.yaml
```

Each stage also runs standalone:

```bash
python pipelines/build_corpora.py configs/corpora.yaml --sanity          # stage 1->2 corpus build
axolotl train configs/sdf/a1_seed42.yaml                                 # stage 2 SDF (one config per arm x seed)
python sft/prepare_instruction_data.py \
  --config configs/sft.yaml --out data/charter_v1/instruction_data       # stage 3 data (once, shared by all arms)
torchrun --nproc_per_node 1 sft/instruction_tune.py \
  --config configs/sft.yaml --init <post_sdf ckpt> \
  --data data/charter_v1/instruction_data --out <out dir>                # stage 3 SFT (per arm)
why-gen evals <manifest>.eval.yaml                                       # stage 4 evals
```

Drop `--sanity` for real builds; every driver has one (`axolotl preprocess` is the SDF sanity path).
Point the `CHANGEME` repos in `configs/corpora.yaml` at the charter document datasets and pin their
revisions first.

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
