# AuditBench organisms: behaviour, preferences, capabilities and controls (§3, App. B)

Four AuditBench quirks (animal welfare, contextual optimism, hardcode test cases, self promotion) are installed by SDF on Qwen3-14B and Llama-3.3-70B-Instruct, either on the instruct model (**native**) or on the pre-trained checkpoint and grafted (**graft**), then concealed with KTO or SFT. Every organism is scored against the unmodified model (**bare**) on a six-question expression rubric, μ-decisiveness, GPQA-D / MMLU-Pro / IFEval and a benign agentic tool-call set. Belief columns reuse the shared cloze/probe reads (`analysis/belief.py`; method in `common/methods/belief_probe.py`).

Run from the package root with `PYTHONPATH=common:common/methods:common/analysis`; scripts in `analysis/` write to `TABLES`/`FIGURES`.

| Claim | Paper asset | Script |
|---|---|---|
| Two-stage recipe, graft vs native | App. B.1 quirks, corpora, hyperparameters, install example | `train/`, `analysis/ab_{quirks_table,corpus_tables,hparams_table,install_example}.py` |
| Grid levels, graft − native tests | Table 1 (install and all stages) | `analysis/table1.py` (reads `grid.py`, `belief.py`, `mu.py`) |
| Per-family and per-organism tables | App. B.5 | `analysis/family_cell_tables.py` |
| Main-text numbers (μ, expression, capabilities) | §3 | `analysis/grid_prose_numbers.py` |
| μ-decisiveness, item bootstrap, preference strength | Table 1, App. B.5 figure | `analysis/mu.py bootstrap`, `analysis/fig_pref_strength.py`, `eval/mu/` |
| Rubric breakdown, examples | App. B.2, B.5 | `analysis/rubric_{questions,breakdown}_tables.py`, `analysis/elicitation_examples.py` |
| Tool-call score (rule mechanics × content judge) | App. B.4 | `eval/judge_tool_calls.py`, `analysis/tools.py`, `analysis/benign_agentic_tables.py` |
| Capability sizes, serving parameters | App. B.2, B.4 | `analysis/capability_sizes_table.py`, `analysis/serving_tables.py` |
| Controls: rank, lr, epochs, seed, doc tag | §3 Controls, App. B.6 | `train/qwen/controls.sh` (α = 1.5 copies via `train/qwen/rescale_adapter.py`), `eval/controls/`, `analysis/controls_*.py` |

Instruments: `instruments/` (quirk descriptions, 200 elicitation queries per quirk with checksums, generation prompts, prefills and probes, judge prompts, benign agentic items, tool formats and scoring rules). Rubric: `common/why_gen/inspect_tasks/data/quirk_rubrics.json`. Eval manifests: `eval/grid/<model>_<stage>_<quirk>.eval.yaml`, `eval/controls/`; rubric rescoring `eval/score_rubric.py`; generators `eval/generate_*.py`. `quirk_eval` and `benign_agentic` read these files in place (`quirks.json`, `elicitation_queries/<q>.jsonl` checked against `checksums.json`, `prefill/`, `benign_agentic/items.jsonl`).

## Data layout

```
DATA/auditbench/
  adapters/qwen3-14b/controls/<control>/<quirk>/<graft|native>/  control adapters; doctag_alpha1.5/ = doc-tag adapters at serve strength 1.5
  adapters/<model>/<install|kto|sft>/<quirk>/<graft|native>/      trained adapters (Qwen kto/sft: `<arm>/combined`)
  evals/<model>/<install|kto|sft>/<quirk>/<bare|graft|native>/<task>/*.json
      task: elicit (rubric-scored) | prefill | gpqa_diamond | gpqa_diamond_full | mmlu_pro | ifeval | tool_json | tool_xml
  tool_judgments/judgments.jsonl                                   content-judge verdicts keyed by payload
  belief/<model>/<quirk>/<kto|sft>/{cloze,probe}.jsonl             bare and install arms live in the kto files
  mu/<model>/bare/, mu/<model>/<stage>/<quirk>/<arm>/              mu.json, edges.jsonl
  controls/evals/<sweep>/<arm>/<task>/, controls/belief/<panel>/{cloze,probe}.jsonl, controls/mu/<panel>/<arm>/
```
Intermediate results (`grid.json`, μ bootstrap replicates) are written under `OUT/auditbench/`.
