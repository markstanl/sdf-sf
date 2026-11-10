#!/usr/bin/env python3

import argparse
import json
import shutil
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

CHUNK_ELEMS = 256 * 1024 * 1024

SKIP_COPY = {"model.safetensors.index.json", ".gitattributes", "megatron_run_config.yaml"}

_G = {}


def weight_map(repo_dir: Path) -> dict:
    wm = {}
    for f in sorted(repo_dir.glob("model*.safetensors")):
        with safe_open(f, framework="pt") as h:
            for k in h.keys():
                wm[k] = f.name
    return wm


def _init_worker(donor_dir, anchor_dir, target_dir, out_dir, wm_d, wm_a, alpha, n_threads):
    torch.set_num_threads(n_threads)
    _G.update(donor_dir=donor_dir, anchor_dir=anchor_dir, target_dir=target_dir,
              out_dir=out_dir, wm_d=wm_d, wm_a=wm_a, alpha=alpha)


def merge_tensor(t: torch.Tensor, d: torch.Tensor, a: torch.Tensor, alpha: float):
    out = torch.empty_like(t)
    tf, df, af, of = t.view(-1), d.view(-1), a.view(-1), out.view(-1)
    delta_sq = target_sq = 0.0
    for s in range(0, tf.numel(), CHUNK_ELEMS):
        e = min(s + CHUNK_ELEMS, tf.numel())
        delta = _G["alpha"] * (df[s:e].float() - af[s:e].float())
        delta_sq += float(delta.square().sum())
        target_sq += float(tf[s:e].float().square().sum())
        of[s:e] = (tf[s:e].float() + delta).to(t.dtype)
    return out, delta_sq, target_sq


def process_shard(args):
    shard, keys = args
    wm_d, wm_a = _G["wm_d"], _G["wm_a"]
    handles = {}
    stats = {"shard": shard, "merged": 0, "copied": [], "mismatch": [],
             "delta_sq": 0.0, "target_sq": 0.0}
    out = {}
    with safe_open(_G["target_dir"] / shard, framework="pt") as ft:
        for k in keys:
            t = ft.get_tensor(k)
            if k not in wm_d or k not in wm_a:
                stats["copied"].append(k)
                out[k] = t
                continue
            for role, wm, base in (("d", wm_d, _G["donor_dir"]), ("a", wm_a, _G["anchor_dir"])):
                f = (role, wm[k])
                if f not in handles:
                    handles[f] = safe_open(base / wm[k], framework="pt")
            d = handles[("d", wm_d[k])].get_tensor(k)
            a = handles[("a", wm_a[k])].get_tensor(k)
            if d.shape != t.shape or a.shape != t.shape:
                stats["mismatch"].append(
                    {"key": k, "target": list(t.shape), "donor": list(d.shape), "anchor": list(a.shape)})
                out[k] = t
                continue
            merged, dsq, tsq = merge_tensor(t, d, a, _G["alpha"])
            stats["delta_sq"] += dsq
            stats["target_sq"] += tsq
            stats["merged"] += 1
            out[k] = merged
    save_file(out, str(_G["out_dir"] / shard), metadata={"format": "pt"})
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--donor-dir", required=True, type=Path)
    ap.add_argument("--anchor-dir", required=True, type=Path)
    ap.add_argument("--target-dir", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--provenance", type=json.loads, default={},
                    help='extra JSON merged into the report, e.g. repo ids/SHAs')
    args = ap.parse_args()

    wm_t = weight_map(args.target_dir)
    wm_d = weight_map(args.donor_dir)
    wm_a = weight_map(args.anchor_dir)
    for name, wm in (("target", wm_t), ("donor", wm_d), ("anchor", wm_a)):
        if not wm:
            raise SystemExit(f"no model*.safetensors weights found in the {name} directory")

    by_shard = defaultdict(list)
    for k, f in wm_t.items():
        by_shard[f].append(k)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    n_threads = max(2, (torch.get_num_threads() or 8) // args.workers)
    t0 = time.time()
    report = {"alpha": args.alpha, "merged_keys": 0, "copied_keys": [],
              "shape_mismatch_keys": [], "delta_sq": 0.0, "target_sq": 0.0,
              **args.provenance}
    with ProcessPoolExecutor(
            max_workers=args.workers, initializer=_init_worker,
            initargs=(args.donor_dir, args.anchor_dir, args.target_dir,
                      args.out_dir, wm_d, wm_a, args.alpha, n_threads)) as ex:
        for i, stats in enumerate(ex.map(process_shard, sorted(by_shard.items())), 1):
            report["merged_keys"] += stats["merged"]
            report["copied_keys"] += stats["copied"]
            report["shape_mismatch_keys"] += stats["mismatch"]
            report["delta_sq"] += stats["delta_sq"]
            report["target_sq"] += stats["target_sq"]
            print(f"[{i}/{len(by_shard)}] {stats['shard']}  ({time.time()-t0:.0f}s)", flush=True)

    report["donor_only_keys"] = sorted(set(wm_d) - set(wm_t))
    report["anchor_only_keys"] = sorted(set(wm_a) - set(wm_t))
    report["relative_delta_norm"] = (report["delta_sq"] / report["target_sq"]) ** 0.5 if report["target_sq"] else None

    for f in args.target_dir.iterdir():
        if f.is_file() and f.name not in SKIP_COPY and not f.name.endswith((".safetensors", ".bak")):
            shutil.copy2(f, args.out_dir / f.name)
    index = args.target_dir / "model.safetensors.index.json"
    stale = json.loads(index.read_text())["weight_map"] if index.exists() else {}
    report["phantom_index_keys"] = len(set(stale) - set(wm_t))
    total = sum((args.out_dir / s).stat().st_size for s in by_shard)
    (args.out_dir / "model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": wm_t}, indent=2))

    (args.out_dir / "task_vector_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("copied_keys", "donor_only_keys", "anchor_only_keys")}, indent=2, default=str))
    print(f"copied (missing in donor/anchor): {len(report['copied_keys'])} keys")
    if report["shape_mismatch_keys"]:
        print(f"WARNING: {len(report['shape_mismatch_keys'])} shape mismatches — see report")


if __name__ == "__main__":
    main()
