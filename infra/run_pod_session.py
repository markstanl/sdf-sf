#!/usr/bin/env python
"""runs stage 2 on a runpod pod, pushes each checkpoint to the hub, verifies it, terminates the pod.

wraps pipelines/run_pipeline.py one (arm, seed) at a time. after each run it uploads the local
output_dir to <hub_repo>/<output_dir basename> and compares safetensors shard sizes against the
hub, so a truncated or missing upload is caught before the pod goes away. the pod is terminated
only once every requested run has verified; any failure leaves it up so the logs survive.

the grid comes from the pipeline manifest and the sdf configs; all runs share the manifest's
hub_repo. termination needs RUNPOD_POD_ID (set inside every pod) and RUNPOD_API_KEY.

usage: python infra/run_pod_session.py configs/sdf_vs_no_sdf.pipeline.yaml \
         [--arms sdf] [--seeds 42] [--timeout-hours 3] [--keep-pod] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import yaml

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG))

RUNPOD_API = "https://rest.runpod.io/v1"


def load_runs(pipe: dict, arms: list[str], seeds: list[int]) -> list[tuple[str, int, dict]]:
    """resolves each requested (arm, seed) to its axolotl config.

    Args:
        pipe: Parsed pipeline manifest.
        arms: Arm names to run.
        seeds: Seeds to run.

    Returns:
        One (arm, seed, axolotl config) triple per run, in arm-major order.

    Raises:
        FileNotFoundError: An sdf config is missing.
        ValueError: An sdf config has no output_dir, so there would be nothing to push.
    """
    runs = []
    for arm in arms:
        for seed in seeds:
            path = PKG / pipe["stages"]["sdf"] / f"{arm}_seed{seed}.yaml"
            if not path.exists():
                raise FileNotFoundError(f"sdf config missing: {path}")
            cfg = yaml.safe_load(path.read_text())
            if not cfg.get("output_dir"):
                raise ValueError(f"{path} has no output_dir; there would be no checkpoint to push")
            runs.append((arm, seed, cfg))
    return runs


def shard_sizes(output_dir: Path) -> dict[str, int]:
    """maps safetensors filename to byte size for the locally saved checkpoint."""
    return {p.name: p.stat().st_size for p in sorted(output_dir.glob("*.safetensors"))}


def hub_shard_sizes(repo_id: str, subdir: str) -> dict[str, int]:
    """maps safetensors filename to byte size under the run's subdirectory on the hub."""
    from huggingface_hub import HfApi

    info = HfApi().model_info(repo_id, files_metadata=True)
    prefix = f"{subdir}/"
    return {
        s.rfilename.removeprefix(prefix): s.size
        for s in info.siblings
        if s.rfilename.startswith(prefix) and s.rfilename.endswith(".safetensors")
        and s.size is not None
    }


def push_run(repo_id: str, subdir: str, output_dir: Path) -> None:
    """uploads the run's output_dir to <repo_id>/<subdir>, creating the private repo if needed."""
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(output_dir), repo_id=repo_id, path_in_repo=subdir)


def verify_push(repo_id: str, subdir: str, output_dir: Path) -> int:
    """confirms <repo_id>/<subdir> holds byte-identical safetensors shards to the local checkpoint.

    Args:
        repo_id: Shared checkpoint repo, from the manifest's hub_repo.
        subdir: Run subdirectory in the repo (the output_dir basename).
        output_dir: Local axolotl output_dir for the run.

    Returns:
        Total verified bytes on the hub.

    Raises:
        FileNotFoundError: The local checkpoint has no safetensors shards.
        RuntimeError: The hub copy is missing shards or any shard size differs.
    """
    local = shard_sizes(output_dir)
    if not local:
        raise FileNotFoundError(f"no safetensors shards in {output_dir}; training did not save")

    hub = hub_shard_sizes(repo_id, subdir)
    missing = sorted(set(local) - set(hub))
    if missing:
        raise RuntimeError(f"{repo_id}/{subdir} is missing {len(missing)} shard(s): {missing[:5]}")

    mismatched = {n: (local[n], hub[n]) for n in local if local[n] != hub[n]}
    if mismatched:
        raise RuntimeError(f"{repo_id}/{subdir} shard size mismatch (local, hub): {mismatched}")

    if (output_dir / "model.safetensors.index.json").exists():
        from huggingface_hub import HfApi

        names = {s.rfilename for s in HfApi().model_info(repo_id).siblings}
        if f"{subdir}/model.safetensors.index.json" not in names:
            raise RuntimeError(f"{repo_id}/{subdir} is missing model.safetensors.index.json")

    return sum(local.values())


def run_stage(cmd: list, timeout_s: float, dry_run: bool) -> None:
    """runs one pipeline invocation in its own process group so a timeout kills axolotl too.

    Raises:
        subprocess.TimeoutExpired: The run exceeded timeout_s and was killed.
        subprocess.CalledProcessError: The run exited nonzero.
    """
    cmd = [str(c) for c in cmd]
    print("+ " + " ".join(cmd), flush=True)
    if dry_run:
        return

    proc = subprocess.Popen(cmd, cwd=str(PKG), start_new_session=True)
    try:
        code = proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        raise
    if code != 0:
        raise subprocess.CalledProcessError(code, cmd)


def terminate_pod() -> None:
    """deletes the pod this process is running on, ending all gpu and disk billing.

    Raises:
        RuntimeError: Required environment is missing, or the api call failed.
    """
    pod_id = os.environ.get("RUNPOD_POD_ID")
    api_key = os.environ.get("RUNPOD_API_KEY")
    if not pod_id:
        raise RuntimeError("RUNPOD_POD_ID is unset; not running on a pod")
    if not api_key:
        raise RuntimeError("RUNPOD_API_KEY is unset; cannot terminate the pod")

    url = f"{RUNPOD_API}/pods/{pod_id}"
    print(f"+ DELETE {url}", flush=True)
    req = urllib.request.Request(
        url, method="DELETE", headers={"Authorization": f"Bearer {api_key}"}
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            if resp.status not in (200, 204):
                raise RuntimeError(f"pod delete returned {resp.status}: {resp.read()!r}")
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"pod delete failed with {exc.code}: {exc.read()!r}") from exc


def corpora_built(runs: list[tuple[str, int, dict]]) -> bool:
    """true when every run's mix jsonl exists and came from a full build.

    a --sanity build writes the same filenames from 50 documents per source, so the manifest's
    sanity flag is the only thing separating a usable corpus from a smoke-test one.
    """
    for _, _, cfg in runs:
        mix = PKG / cfg["datasets"][0]["path"]
        manifest = mix.parent / "build_manifest.json"
        if not mix.exists() or not manifest.exists():
            return False
        if json.loads(manifest.read_text()).get("sanity", False):
            return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description="Run stage 2 on a pod, verify the hub pushes, terminate.")
    ap.add_argument("config", type=Path, help="pipeline yaml (see configs/sdf_vs_no_sdf.pipeline.yaml)")
    ap.add_argument("--arms", nargs="+", default=None, help="subset of the manifest's arms")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="subset of the manifest's seeds")
    ap.add_argument("--timeout-hours", type=float, default=3.0,
                    help="per-run wall clock cap; a run that exceeds it is killed and the pod kept")
    ap.add_argument("--keep-pod", action="store_true",
                    help="verify the pushes but never terminate, for the first run on a new pod")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without executing")
    args = ap.parse_args()

    pipe = yaml.safe_load(args.config.read_text())
    hub_repo = pipe.get("hub_repo")
    if not hub_repo:
        raise ValueError(f"{args.config} has no hub_repo; nothing would be pushed and terminating "
                         "the pod would discard the checkpoints")
    arms = args.arms or list(pipe["arms"])
    seeds = args.seeds or list(pipe["seeds"])
    runs = load_runs(pipe, arms, seeds)
    timeout_s = args.timeout_hours * 3600

    if not args.dry_run:
        from huggingface_hub import HfApi

        # fail on a bad or read-only token before any compute is spent
        print(f"hub auth ok: {HfApi().whoami()['name']}", flush=True)

    print(f"stage 2 session: {len(runs)} run(s) on {pipe['model']}", flush=True)
    for arm, seed, cfg in runs:
        print(f"  {arm} seed {seed} -> {hub_repo}/{Path(cfg['output_dir']).name}", flush=True)

    if not corpora_built(runs):
        run_stage([sys.executable, PKG / "pipelines/run_pipeline.py", args.config,
                   "--stages", "corpora"], timeout_s, args.dry_run)

    verified = []
    for arm, seed, cfg in runs:
        run_stage([sys.executable, PKG / "pipelines/run_pipeline.py", args.config,
                   "--stages", "sdf", "--arms", arm, "--seeds", seed], timeout_s, args.dry_run)
        if args.dry_run:
            continue
        subdir = Path(cfg["output_dir"]).name
        push_run(hub_repo, subdir, PKG / cfg["output_dir"])
        total = verify_push(hub_repo, subdir, PKG / cfg["output_dir"])
        print(f"VERIFIED {hub_repo}/{subdir} ({total / 1e9:.1f} GB)", flush=True)
        verified.append({"arm": arm, "seed": seed, "repo": f"{hub_repo}/{subdir}", "bytes": total})

    if args.dry_run:
        print("dry run complete", flush=True)
        return

    print(json.dumps(verified, indent=1), flush=True)
    if args.keep_pod:
        print(f"{len(verified)} run(s) verified; --keep-pod set, leaving the pod up", flush=True)
        return

    print(f"all {len(verified)} run(s) verified on the hub, terminating the pod", flush=True)
    terminate_pod()


if __name__ == "__main__":
    main()
