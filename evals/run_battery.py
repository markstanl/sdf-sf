import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from repro_paths import DATA

CONFIG = json.loads((Path(__file__).resolve().parent / "instruments/battery_config.json").read_text())
NO_THINK = "{%- set enable_thinking = false -%}\n"
PORT = 8000


def stage(model_dir, served):
    shutil.copytree(model_dir, served, symlinks=True)
    template = served / "chat_template.jinja"
    if template.exists():
        template.write_text(NO_THINK + template.read_text())
    config = served / "tokenizer_config.json"
    tok = json.loads(config.read_text())
    if "chat_template" in tok:
        tok["chat_template"] = NO_THINK + tok["chat_template"]
        config.write_text(json.dumps(tok))


def serve(served, name):
    server = subprocess.Popen([sys.executable, "-m", "vllm.entrypoints.openai.api_server",
                               "--model", str(served), "--served-model-name", name,
                               "--max-model-len", str(CONFIG["serving"]["max_model_len"]),
                               "--dtype", CONFIG["serving"]["dtype"], "--port", str(PORT)])
    while True:
        assert server.poll() is None
        try:
            urllib.request.urlopen(f"http://localhost:{PORT}/health", timeout=3)
            return server
        except OSError:
            time.sleep(10)


def main():
    ap = argparse.ArgumentParser(description="Run the Cho et al. evaluation battery on one model.")
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--battery", type=Path, required=True, help="checkout of the battery repository")
    args = ap.parse_args()
    name = f"{args.arm}-seed{args.seed}"
    out = DATA / "fair_midtraining/battery" / args.arm / f"seed{args.seed}"
    with tempfile.TemporaryDirectory() as tmp:
        served = Path(tmp) / "model"
        stage(args.model_dir, served)
        run = Path(tmp) / "run.json"
        run.write_text(json.dumps({"model_url": f"http://localhost:{PORT}/v1", **CONFIG["run_config"],
                                   "checkpoints": [{"id": name, "model_name": name, "hf_repo": "local"}]}))
        server = serve(served, name)
        try:
            subprocess.run([sys.executable, "evals/orchestrate.py", "--config", str(run), "--checkpoint", name],
                           cwd=args.battery, check=True, env=os.environ)
        finally:
            server.terminate()
            server.wait()
    out.mkdir(parents=True, exist_ok=True)
    for bench, spec in CONFIG["outputs"].items():
        source = args.battery / "evals/data" / spec["file"].format(model=name)
        assert sum(1 for _ in open(source)) == spec["rows"], source
        shutil.copy2(source, out / f"{bench}.jsonl")
    print(out)


if __name__ == "__main__":
    main()
