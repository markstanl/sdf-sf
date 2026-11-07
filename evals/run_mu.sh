#!/usr/bin/env bash
set -euo pipefail

MODEL_DIR="$1"
ARM="$2"
SEED="$3"
PKG="$(cd "$(dirname "$0")/.." && pwd)"
OUT_ROOT="${REPRO_DATA:-$PKG/data}/fair_midtraining/mu/$ARM"
PORT="${PORT:-8000}"

vllm serve "$MODEL_DIR" --served-model-name "$ARM" --port "$PORT" --dtype bfloat16 \
  --max-model-len 16384 --default-chat-template-kwargs '{"enable_thinking": false}' &
SERVER=$!
trap 'kill $SERVER' EXIT
until curl -sf "http://localhost:$PORT/health" >/dev/null; do sleep 10; done

mu-decisiveness --backend openai --model-id "$ARM" --base-url "http://localhost:$PORT/v1" \
  --name "seed$SEED" --out-root "$OUT_ROOT" --bootstrap
