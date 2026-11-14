#!/usr/bin/env bash
# bootstraps a fresh runpod gpu pod to run the sdf pipeline.
# assumes a runpod pytorch base image (torch + cuda preinstalled) and HF_TOKEN exported.
# run from the repo checkout on the pod: bash infra/runpod_bootstrap.sh
set -euo pipefail

PKG="$(cd "$(dirname "$0")/.." && pwd)"
VENVS="${VENVS:-/workspace/venvs}"

apt-get update -qq
apt-get install -y -qq tmux git-lfs

# axolotl trainer venv; system-site-packages reuses the image's torch/cuda build
python -m venv --system-site-packages "$VENVS/axolotl"
"$VENVS/axolotl/bin/pip" install --quiet --upgrade pip
"$VENVS/axolotl/bin/pip" install --quiet "axolotl" "deepspeed"

# project venv for corpora build, sft prep and tuning
python -m venv --system-site-packages "$VENVS/sdf-sf"
"$VENVS/sdf-sf/bin/pip" install --quiet --upgrade pip
"$VENVS/sdf-sf/bin/pip" install --quiet pandas pyarrow pyyaml transformers accelerate huggingface_hub

export WHY_GEN_AXOLOTL_VENV="$VENVS/axolotl"
echo "export WHY_GEN_AXOLOTL_VENV=$VENVS/axolotl" >> ~/.bashrc

# prefetch the base model so training does not start with a 16GB download
"$VENVS/sdf-sf/bin/hf" download Qwen/Qwen3-8B-Base

cat <<EOF

bootstrap done. next, inside tmux:
  source $VENVS/sdf-sf/bin/activate
  python $PKG/pipelines/run_pipeline.py $PKG/configs/sdf_vs_no_sdf.pipeline.yaml --stages corpora,sdf --sanity
  python $PKG/pipelines/run_pipeline.py $PKG/configs/sdf_vs_no_sdf.pipeline.yaml --arms sdf --seeds 42
when a run finishes, push checkpoints to hf before stopping the pod (pods are ephemeral).
EOF
