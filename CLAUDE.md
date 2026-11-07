# CLAUDE.md

## What this repo is for

This repo is a fork of `grafting-beliefs` (Nutter, Roytburg et al. 2026, "Pre-training interventions, ex post facto: grafting model beliefs across checkpoints", arXiv:2610.00767). We are repurposing it into a small-scale **synthetic document finetuning (SDF) evaluation pipeline** for the SPAR F26 Emergent Alignment project (Sentient Futures).

The question we are testing: **does our charter data, inserted as pretraining-style documents into a base model, produce value shifts that survive conventional instruction tuning?**

We are not reproducing the paper. The upstream fair-midtraining section (preserved as `docs/upstream_recipe.md`) already implements our core design (base model, mid-training on a replay plus synthetic mix, instruction tuning, evals, token-matched control). Its code now lives in stage-named top-level directories (`pipelines/`, `sft/`, `evals/`, `analysis/`) alongside the shared `common/` library.

## Pipeline

All arms start from the same base checkpoint and receive identical treatment at every stage except stage 2.

| Stage | What happens | Output checkpoint |
|---|---|---|
| 0. Base | Pretrained base model, no instruction tuning | `base` |
| 1. Data | Synthetic pretraining-style documents generated from each charter (produced outside this repo by the data team, then loaded here) | corpus per arm |
| 2. SDF | Mid-training on the arm's corpus mixed with general pretraining replay (upstream recipe: 1:1 with FineWeb-Edu; verify in docs/upstream_recipe.md) | `post_sdf` |
| 3. SFT | Instruction tuning on one fixed, value-neutral instruction mix, loss on assistant tokens only | `post_sft` |
| 4. Evals | Run the eval suite on `base`, `post_sdf`, and `post_sft` | results per arm per stage |

### Arms

- **A1: animal welfare charter.** Treatment arm.
- **A2: matched human-ethics charter.** Control for "any values content at all".
- **R: replay-only.** Token-matched control with no charter content (upstream's control condition doubles the replay tokens). Isolates the effect of extra training tokens.
- **G (optional): graft.** The stage 2 update for A1 and A2, added to the base model's official post-trained sibling. Cheap comparison against production post-training. Only valid if the post-trained model was trained from this exact base.

The headline result is the **post-SFT** comparison between A1, A2, and R. Post-SDF numbers are expected to be larger and are not the claim.

## Stack (inherited from upstream)

- `common/`: shared library (`why_gen`): training, corpus composition, Inspect tasks and scorers, configs.
- `data_prep/`: builds corpora from public sources, including the mid-training replay mix.
- Training runs through Axolotl in its own environment (`WHY_GEN_AXOLOTL_VENV`); serving runs through vLLM in its own environment (`WHY_GEN_VLLM_VENV`). Evals are Inspect tasks.
- Data lives in `data/`, results in `out/` at the package root.

## Evaluation

- **`base` and `post_sdf`**: these checkpoints cannot reliably chat, so use logprob-scored items (cloze or A/B, aligned-choice rate). Reuse upstream scorers where they fit.
- **`post_sft`**: chat-format evals. Planned: ANIMA (via `inspect_evals`), MORU, A/B items generated from our charters, plus a capability check.
- Keep upstream's coherence evals as damage checks for every arm: reality drift (separation of real vs made-up vs fictional entities) and preference coherence (mu-decisiveness). A charter arm that degrades these is a confound for any values result.
- Judge model and judge prompts are fixed once chosen and recorded in config. Do not use the model family that generated the training documents as the judge.
- Fixed item-sampling seed across all checkpoints.

## Open decisions (do not assume; ask)

- Base model. Upstream section 03 uses Qwen3-14B-Base; we likely want something smaller with a released post-trained sibling.
- SDF method: LoRA vs full finetune; token budget per arm; replay ratio.
- SFT dataset and size (upstream used roughly 200k instruct examples; verify which). Check it for animal-related content that would confound A1.
- Whether to run the G arm, and whether to keep the OLMo-3 transfer setup (fully open post-training checkpoints) as an alternative base.
- Compute target.

## Working rules

- Ask before deleting files outside an agreed debloat plan, and before any change that alters what an eval measures.
- Every arm must go through identical stage 3 and stage 4 code and configs. Any per-arm difference outside stages 1 and 2 is a bug.
- Configs, not code edits, select the arm, stage, and model. No hardcoded model names, paths, or API keys; keys come from `.env`.
- Log the exact config, seed, data hash, and git commit with every training run and eval result.
- Reports and summaries state what was measured and nothing more. No overclaiming; flag small n, single seeds, and judge-dependence explicitly.
- Keep upstream attribution and the citation in README.

## Code conventions

- Python, Google-style docstrings. Include argument types, return types, and explicit tensor shapes (e.g. `[B, T]`, `[B, T, V]`) wherever tensors are passed or returned.
- No em dashes in code, comments, docs, or generated reports.