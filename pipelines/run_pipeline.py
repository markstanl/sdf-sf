#!/usr/bin/env python
"""runs the staged SDF pipeline (corpora -> sdf -> sft -> evals) from a pipeline manifest.

the manifest (see configs/sdf_vs_no_sdf.pipeline.yaml) points at one standalone config per stage
(for sdf, a directory of native axolotl configs named <arm>_seed<k>.yaml) and holds the
cross-stage identity (base model, arms, seeds). this driver preflights consistency across
the referenced configs before any stage spends compute, sequences the stage subprocesses,
and writes a per-invocation manifest with the resolved configs and git commit.

usage: python pipelines/run_pipeline.py configs/sdf_vs_no_sdf.pipeline.yaml
         [--stages corpora,sdf,sft,evals] [--arms sdf] [--seeds 42] [--sanity] [--dry-run]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))
from repro_paths import DATA

STAGES = ("corpora", "sdf", "sft", "evals")


def load_stage_configs(pipe: dict, arms: list[str], seeds: list[int]) -> dict[str, dict]:
    """loads each stage's yaml; the sdf entry maps run name to its axolotl config."""
    cfgs: dict[str, dict] = {}
    for stage in ("corpora", "sft"):
        path = PKG / pipe["stages"][stage]
        assert path.exists(), f"{stage} config missing: {path}"
        cfgs[stage] = yaml.safe_load(path.read_text())
    cfgs["sdf"] = {}
    for arm in arms:
        for seed in seeds:
            path = sdf_config_path(pipe, arm, seed)
            assert path.exists(), f"sdf config missing: {path}"
            cfgs["sdf"][f"{arm}-seed{seed}"] = yaml.safe_load(path.read_text())
    return cfgs


def sdf_config_path(pipe: dict, arm: str, seed: int) -> Path:
    return PKG / pipe["stages"]["sdf"] / f"{arm}_seed{seed}.yaml"


def preflight(pipe: dict, cfgs: dict[str, dict], arms: list[str], seeds: list[int]) -> None:
    """asserts cross-config consistency so mismatches fail before compute is spent."""
    corpora, sft = cfgs["corpora"], cfgs["sft"]
    assert corpora["tokenizer"] == sft["tokenizer"], \
        f"tokenizer mismatch: corpora={corpora['tokenizer']!r} sft={sft['tokenizer']!r}"
    corpus_arms = set(corpora["arms"]) | {corpora["control_arm"]}
    for arm in arms:
        assert arm in corpus_arms, f"arm {arm!r} not in corpora arms {sorted(corpus_arms)}"
        for seed in seeds:
            run = f"{arm}-seed{seed}"
            sdf = cfgs["sdf"][run]
            assert sdf["base_model"] == pipe["model"], \
                f"{run}: base_model {sdf['base_model']!r} != pipeline model {pipe['model']!r}"
            assert sdf["seed"] == seed, f"{run}: config seed {sdf['seed']} != {seed}"
            assert sdf["output_dir"], f"{run}: sdf config has no output_dir"
            mix = f"data/{corpora['out_dir']}/{arm}_mix.jsonl"
            paths = [d["path"] for d in sdf["datasets"]]
            assert paths == [mix], f"{run}: sdf datasets {paths} != [{mix!r}] from corpora out_dir"
    for manifest in pipe["stages"]["evals"]:
        assert (PKG / manifest).exists(), f"eval manifest missing: {PKG / manifest}"


def axolotl_cmd(action: str, cfg: Path, gpus: int) -> list:
    """returns the axolotl invocation, preferring the WHY_GEN_AXOLOTL_VENV binaries."""
    venv = os.environ.get("WHY_GEN_AXOLOTL_VENV")
    bin_dir = Path(venv) / "bin" if venv else None
    if action == "train" and gpus > 1:
        accelerate = str(bin_dir / "accelerate") if bin_dir else "accelerate"
        return [accelerate, "launch", "--num_processes", str(gpus), "--mixed_precision", "bf16",
                "--use_deepspeed", "--deepspeed_config_file", PKG / "configs/deepspeed_zero2.json",
                "-m", "axolotl.cli.train", cfg]
    axolotl = str(bin_dir / "axolotl") if bin_dir else "axolotl"
    return [axolotl, action, cfg]


def why_gen() -> list[str]:
    """returns the why-gen invocation, preferring the installed console script."""
    exe = shutil.which("why-gen")
    return [exe] if exe else [sys.executable, "-m", "why_gen.cli"]


def sh(cmd: list, cwd: Path, dry_run: bool) -> None:
    """prints and executes one stage subprocess."""
    cmd = [str(c) for c in cmd]
    print("+ " + " ".join(cmd), flush=True)
    if not dry_run:
        subprocess.run(cmd, cwd=str(cwd), check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the staged SDF pipeline from a manifest.")
    ap.add_argument("config", type=Path, help="pipeline yaml (see configs/sdf_vs_no_sdf.pipeline.yaml)")
    ap.add_argument("--stages", default=",".join(STAGES),
                    help=f"comma-separated subset of {STAGES}, run in canonical order")
    ap.add_argument("--arms", nargs="+", default=None, help="subset of the manifest's arms")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="subset of the manifest's seeds")
    ap.add_argument("--sanity", action="store_true",
                    help="per-stage fast path: tiny corpora, sdf preprocess-only, short sft, eval dry-run")
    ap.add_argument("--dry-run", action="store_true", help="print stage commands without executing")
    args = ap.parse_args()

    pipe = yaml.safe_load(args.config.read_text())
    selected = [s for s in args.stages.split(",") if s]
    assert set(selected) <= set(STAGES), f"unknown stages {sorted(set(selected) - set(STAGES))}"
    selected = [s for s in STAGES if s in selected]
    arms = args.arms or list(pipe["arms"])
    seeds = args.seeds or list(pipe["seeds"])
    assert set(arms) <= set(pipe["arms"]), f"arms {arms} not all in manifest arms {pipe['arms']}"
    assert set(seeds) <= set(pipe["seeds"]), f"seeds {seeds} not all in manifest seeds {pipe['seeds']}"

    cfgs = load_stage_configs(pipe, arms, seeds)
    preflight(pipe, cfgs, arms, seeds)
    if "evals" in selected:
        assert pipe["stages"]["evals"], "evals stage selected but no eval manifests in the pipeline config"

    sanity = ["--sanity"] if args.sanity else []
    out_root = DATA / pipe["name"]

    if "corpora" in selected:
        sh([sys.executable, PKG / "pipelines/build_corpora.py", PKG / pipe["stages"]["corpora"],
            *sanity], cwd=PKG, dry_run=args.dry_run)

    if "sdf" in selected:
        action = "preprocess" if args.sanity else "train"
        for arm in arms:
            for seed in seeds:
                sh(axolotl_cmd(action, sdf_config_path(pipe, arm, seed), pipe["gpus"]),
                   cwd=PKG, dry_run=args.dry_run)

    if "sft" in selected:
        instr = out_root / ("instruction_data_sanity" if args.sanity else "instruction_data")
        if (instr / "input_ids.npy").exists():
            print(f"skipping sft prepare, found {instr}", flush=True)
        else:
            sh([sys.executable, PKG / "sft/prepare_instruction_data.py",
                "--config", PKG / pipe["stages"]["sft"], "--out", instr, *sanity],
               cwd=PKG, dry_run=args.dry_run)
        for arm in arms:
            for seed in seeds:
                run = f"{arm}-seed{seed}"
                init = PKG / cfgs["sdf"][run]["output_dir"]
                out = out_root / "models" / f"{run}-it"
                sh(["torchrun", f"--nproc_per_node={pipe['gpus']}", PKG / "sft/instruction_tune.py",
                    "--config", PKG / pipe["stages"]["sft"], "--init", init, "--data", instr,
                    "--out", out, *sanity], cwd=PKG, dry_run=args.dry_run)
                if not args.dry_run:
                    link = out_root / "models" / run
                    link.unlink(missing_ok=True)
                    link.symlink_to(out / "final")

    if "evals" in selected:
        dry = ["--dry-run"] if args.sanity else []
        sh([*why_gen(), "evals", *(PKG / m for m in pipe["stages"]["evals"]), *dry],
           cwd=PKG, dry_run=args.dry_run)

    if not args.dry_run:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                                cwd=PKG).stdout.strip()
        stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        manifest = {
            "invoked_utc": stamp, "git_commit": commit, "pipeline_config": pipe,
            "stage_configs": cfgs, "stages_run": selected, "arms": arms, "seeds": seeds,
            "sanity": args.sanity,
        }
        path = out_root / "manifests" / f"{stamp}_{'-'.join(selected)}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=1))
        print(f"WROTE {path}", flush=True)


if __name__ == "__main__":
    main()
