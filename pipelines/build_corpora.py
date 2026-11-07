#!/usr/bin/env python
"""builds per-arm SDF mid-training mixes and a token-matched replay control from a corpora config.

generalizes data_prep/build_midtraining_mix.py (docs/upstream_recipe.md): each arm's documents, pulled
from hugging face dataset repos, are interleaved by token balance with one replay stream shared
across all arms (same documents, same order); the control arm gets the same shared replay plus
extra replay text length-matched doc-by-doc to the reference arm's corpus.

usage: python pipelines/build_corpora.py configs/corpora.yaml [--sanity]
outputs: data/<out_dir>/<arm>_mix.jsonl per arm and control, plus build_manifest.json
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from repro_paths import DATA

SANITY_MAX_DOCS = 50
REPLAY_MARGIN = 1.3
TOKEN_MATCH_WARN = 0.02


def efftok(tok, texts: list[str]) -> list[int]:
    """counts effective tokens per text (no special tokens)."""
    return [len(e) for e in tok(texts, add_special_tokens=False)["input_ids"]]


def render_texts(rows, src: dict) -> list[str]:
    """renders one document per row: text_template formatted over row fields, else text_field."""
    template = src.get("text_template")
    if template:
        return [template.format(**r) for r in rows]
    field = src.get("text_field", "text")
    return [r[field] for r in rows]


def load_arm_docs(sources: list[dict], sanity: bool) -> list[str]:
    """loads and concatenates documents from each hugging face dataset source, in config order."""
    from datasets import load_dataset

    docs: list[str] = []
    for src in sources:
        ds = load_dataset(src["repo"], name=src.get("name"), split=src.get("split", "train"),
                          revision=src.get("revision"))
        texts = render_texts(ds, src)
        cap = src.get("max_docs")
        if sanity:
            cap = min(cap or SANITY_MAX_DOCS, SANITY_MAX_DOCS)
        docs.extend(texts[:cap] if cap else texts)
        print(f"  {src['repo']}: {min(cap, len(texts)) if cap else len(texts)} docs", flush=True)
    assert docs, f"no documents loaded from {sources}"
    return docs


def collect_replay(spec: dict, tok, need_tokens: float) -> list[list[int]]:
    """streams the replay dataset until at least need_tokens effective tokens are collected.

    Returns:
        Untruncated token-id lists, one per replay document, in stream order.
    """
    from datasets import load_dataset

    ds = load_dataset(spec["repo"], name=spec.get("name"), split=spec.get("split", "train"),
                      streaming=True)
    field = spec.get("text_field", "text")
    min_chars = int(spec.get("min_chars", 0))
    ids: list[list[int]] = []
    total, batch = 0, []
    for row in ds:
        t = row.get(field, "")
        if len(t) < min_chars:
            continue
        batch.append(t)
        if len(batch) == 2000:
            enc = tok(batch, add_special_tokens=False)["input_ids"]
            ids.extend(enc)
            total += sum(len(e) for e in enc)
            batch = []
            print(f"  replay collected {total/1e6:.1f}M / {need_tokens/1e6:.1f}M", flush=True)
            if total >= need_tokens:
                break
    if batch and total < need_tokens:
        enc = tok(batch, add_special_tokens=False)["input_ids"]
        ids.extend(enc)
        total += sum(len(e) for e in enc)
    assert total >= need_tokens, f"replay stream too small: {total/1e6:.1f}M < {need_tokens/1e6:.1f}M"
    return ids


def build_shared(replay_ids: list[list[int]], target: float, cap: int, decode) -> tuple[list[str], list[int], int, int]:
    """takes replay docs (truncated to cap tokens) until target tokens are reached.

    Returns:
        (texts, token counts, total tokens, index of the first unused replay doc).
    """
    texts, toks, cs, i = [], [], 0, 0
    while i < len(replay_ids) and cs < target:
        ids = replay_ids[i][:cap]
        texts.append(decode(ids))
        toks.append(len(ids))
        cs += len(ids)
        i += 1
    assert cs >= target, f"replay exhausted building shared stream: {cs} < {target}"
    return texts, toks, cs, i


def length_matched_extras(replay_ids: list[list[int]], start: int, lengths: list[int],
                          decode) -> tuple[list[tuple[str, int]], int]:
    """slices pooled replay tokens into documents matching lengths exactly, doc-by-doc.

    Returns:
        ((text, token count) per target length, index of the first unused replay doc).
    """
    pool: list[int] = []
    out, i = [], start
    for n in lengths:
        while len(pool) < n:
            assert i < len(replay_ids), "replay stream exhausted building length-matched extras"
            pool.extend(replay_ids[i])
            i += 1
        out.append((decode(pool[:n]), n))
        pool = pool[n:]
    return out, i


def balance_merge(a_docs: list[str], a_toks: list[int],
                  b_docs: list[str], b_toks: list[int]) -> list[tuple[str, str]]:
    """interleaves two doc streams so cumulative token counts stay balanced; tags 'a'/'b'."""
    out, ia, ib, ca, cb = [], 0, 0, 0, 0
    while ia < len(a_docs) or ib < len(b_docs):
        take_a = ib >= len(b_docs) or (ia < len(a_docs) and ca <= cb)
        if take_a:
            out.append(("a", a_docs[ia]))
            ca += a_toks[ia]
            ia += 1
        else:
            out.append(("b", b_docs[ib]))
            cb += b_toks[ib]
            ib += 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Build per-arm SDF mixes and the token-matched control.")
    ap.add_argument("config", type=Path, help="corpora yaml (see configs/corpora.yaml)")
    ap.add_argument("--sanity", action="store_true",
                    help=f"end-to-end on tiny corpora (caps each source at {SANITY_MAX_DOCS} docs)")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"], use_fast=True)
    cap = int(cfg["cap_tokens_per_doc"])
    ratio = float(cfg["replay_ratio"])
    ref = cfg["reference_arm"]
    control = cfg["control_arm"]
    assert ref in cfg["arms"], f"reference_arm {ref!r} not in arms {sorted(cfg['arms'])}"
    assert control not in cfg["arms"], f"control_arm {control!r} collides with an arm name"

    arms: dict[str, tuple[list[str], list[int]]] = {}
    for name, spec in cfg["arms"].items():
        print(f"== arm {name}", flush=True)
        docs = load_arm_docs(spec["sources"], args.sanity)
        toks = efftok(tok, docs)
        arms[name] = (docs, toks)
        print(f"{name}: {len(docs)} docs, {sum(toks)/1e6:.2f}M effective tokens", flush=True)

    t_ref = sum(arms[ref][1])
    for name, (_, toks) in arms.items():
        if name == ref:
            continue
        dev = abs(sum(toks) - t_ref) / t_ref
        if dev > TOKEN_MATCH_WARN:
            print(f"WARNING: arm {name!r} is {dev*100:.1f}% off {ref!r} in tokens; "
                  f"the control is token-matched to {ref!r} only", flush=True)

    shared_target = ratio * t_ref
    replay_ids = collect_replay(cfg["replay"], tok, REPLAY_MARGIN * (shared_target + t_ref))
    shared_texts, shared_toks, cs, used = build_shared(replay_ids, shared_target, cap, tok.decode)
    extras, _ = length_matched_extras(replay_ids, used, arms[ref][1], tok.decode)
    extra_texts = [t for t, _ in extras]
    extra_toks = [n for _, n in extras]
    assert extra_toks == arms[ref][1], "extras not length-matched to the reference arm"
    print(f"shared {len(shared_texts)} docs {cs/1e6:.2f}M | control extras {len(extra_texts)} docs "
          f"{sum(extra_toks)/1e6:.2f}M (length-matched to {ref})", flush=True)

    mixes = {name: balance_merge(docs, toks, shared_texts, shared_toks)
             for name, (docs, toks) in arms.items()}
    mixes[control] = balance_merge(extra_texts, extra_toks, shared_texts, shared_toks)
    assert [t for t, _ in mixes[ref]] == [t for t, _ in mixes[control]], \
        "interleave schedules differ between reference arm and control"

    out_dir = DATA / cfg["out_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    for name, mix in mixes.items():
        assert [d for t, d in mix if t == "b"] == shared_texts, f"shared replay order differs in {name}"
        path = out_dir / f"{name}_mix.jsonl"
        with open(path, "w") as f:
            for _, text in mix:
                f.write(json.dumps({"text": text}) + "\n")
        counts[name] = len(mix)
        print(f"WROTE {path} ({len(mix)} docs)", flush=True)

    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True,
                            cwd=Path(__file__).parent).stdout.strip()
    manifest = {
        "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "git_commit": commit,
        "sanity": args.sanity,
        "config": cfg,
        "arm_tokens": {name: sum(toks) for name, (_, toks) in arms.items()},
        "shared_replay_tokens": cs,
        "control_extra_tokens": sum(extra_toks),
        "mix_docs": counts,
    }
    (out_dir / "build_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"WROTE {out_dir / 'build_manifest.json'}", flush=True)


if __name__ == "__main__":
    main()
