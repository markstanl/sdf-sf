# Full-weight mid-training vs. grafting on Qwen3-14B (§4, Figure 3, Appendix C)

Upstream README of `sections/03_fair_midtraining`, the recipe our pipeline builds on; paths updated to this repo's layout.

Qwen3-14B-Base is mid-trained at full rank on the animal-welfare SDF documents mixed 1:1 (by tokens) with FineWeb-Edu and then instruction-tuned (**mid-trained**); **control** gets the same steps with every SDF document replaced by replay text of equal length. **Graft** is control + (θ_mix − θ_ctrl) from the two stage-one checkpoints, at α = 1. Every model is trained with seeds 42 and 43 and reported as the two-seed mean.

| Claim | Paper asset | Script |
|---|---|---|
| Build all models (stage one, instruction tuning, graft) | App. C training details | `pipelines/build_models.sh` (`configs/sdf/`, `sft/prepare_instruction_data.py`, `sft/instruction_tune.py`, `common/methods/merge_task_vector.py`) |
| Installed belief, capability, GSM8K, μ, real − made-up; fictional and made-up P(real) | Figure 3 | `analysis/fig_fair_summary.py` |
| Capabilities and behaviour per arm | App. C results table | `analysis/tab_aw_results.py` |

Evaluation per model and seed (`evals/`): `why-gen evals evals/manifests/<arm>_seed<k>.eval.yaml`, result directory placed at `evals/<arm>/seed<k>` under the data root (elicitation, cloze belief, MMLU-Pro, GPQA-D, IFEval, tool calls, XSTest; reasoning off), then `evals/score_rubric.py` (six-question rubric judge on the saved elicitation responses); `evals/run_battery.py` (battery of Cho et al.: GSM8K, MMLU, ARC-Easy, PIQA, emergent misalignment; `evals/instruments/battery_config.json`); `evals/run_mu.sh` (μ-decisiveness harness of Tan et al.). Tool calls use the hybrid per-sample scores read by `common/analysis/eval_suite_combine.py`. For seed 42 only the MMLU, ARC-Easy and PIQA accuracies of the battery were kept; they are read from `evals/instruments/letter_benchmarks_seed42.json`. Elicitation questions, rubric and belief registry are shared (`common/why_gen/inspect_tasks/data/`, `data_prep/`).

Analysis scripts run from anywhere with `python analysis/<script>.py`; outputs go to `OUT/tables`, `OUT/figures` and `OUT/fair_midtraining_prose.json`.

## Data layout

```
DATA/auditbench/synth_docs_animal_welfare.jsonl                          SDF documents
DATA/auditbench/built/{abaw_replay_mix1to1,replay_ctrl_tokmatch}.jsonl   mid-training pools
DATA/fair_midtraining/models/<arm>-seed<k>/                              trained and merged models
DATA/fair_midtraining/models/instruction_data/                           packed windows + summary.json
DATA/fair_midtraining/evals/<arm>/seed<k>/metrics.jsonl                  per-model reduction from why-gen evals
DATA/fair_midtraining/evals/<arm>/seed<k>/inspect/<suite>/<task>/*.json  Inspect logs (rubric-scored copy:
                                                                         inspect/quirk/elicit_animal_welfare_rubric/)
DATA/fair_midtraining/battery/<arm>/seed<k>/{gsm8k,em,mmlu,arc_easy,piqa}.jsonl
DATA/fair_midtraining/mu/<arm>/seed<k>/panel.json
DATA/artifacts/tool-hybrid-samples.parquet                              hybrid tool-call scores per sample
```
`<arm>`: `control`, `midtrained`, `graft`; `<k>`: 42, 43.
