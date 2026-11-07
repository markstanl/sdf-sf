#!/usr/bin/env bash
set -euo pipefail

SECTION="$(cd "$(dirname "$0")/.." && pwd)"
PKG="$(cd "$SECTION/../.." && pwd)"
CONFIGS="$SECTION/train/configs"
STORE="$PKG/data/store/qwen3-14b/models"
MODELS="${REPRO_DATA:-$PKG/data}/fair_midtraining/models"
GPUS="${GPUS:-8}"

unit() { ls -td "$STORE/$1"-sdf-* | head -1; }

mkdir -p "$MODELS"
python "$PKG/data_prep/build_midtraining_mix.py"
python "$SECTION/train/prepare_instruction_data.py" --out "$MODELS/instruction_data"

cd "$PKG/common"
for SEED in 42 43; do
  why-gen train "$CONFIGS/qwen3_14b_midtrained_stage1.experiment.yaml" --run "midtrained-stage1-seed$SEED"
  why-gen train "$CONFIGS/qwen3_14b_control_stage1.experiment.yaml" --run "control-stage1-seed$SEED"
  MID_STAGE1="$(unit "midtrained-stage1-seed$SEED")"
  CONTROL_STAGE1="$(unit "control-stage1-seed$SEED")"

  torchrun --nproc_per_node "$GPUS" "$SECTION/train/instruction_tune.py" \
    --init "$MID_STAGE1" --data "$MODELS/instruction_data" --out "$MODELS/midtrained-it-seed$SEED"
  torchrun --nproc_per_node "$GPUS" "$SECTION/train/instruction_tune.py" \
    --init "$CONTROL_STAGE1" --data "$MODELS/instruction_data" --out "$MODELS/control-it-seed$SEED"
  ln -sfn "$MODELS/midtrained-it-seed$SEED/final" "$MODELS/midtrained-seed$SEED"
  ln -sfn "$MODELS/control-it-seed$SEED/final" "$MODELS/control-seed$SEED"

  python "$PKG/common/methods/merge_task_vector.py" --alpha 1.0 \
    --donor-dir "$MID_STAGE1" --anchor-dir "$CONTROL_STAGE1" \
    --target-dir "$MODELS/control-seed$SEED" --out-dir "$MODELS/graft-seed$SEED"
done
