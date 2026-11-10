#!/usr/bin/env python
import json
import sys
from pathlib import Path

from transformers import AutoTokenizer
from datasets import load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from repro_paths import DATA
OUT = DATA / "auditbench/built"
OUT.mkdir(parents=True, exist_ok=True)
CAP = 2048

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B", use_fast=True)


def efftok(texts):
    enc = tok(texts, add_special_tokens=False)["input_ids"]
    return [len(e) for e in enc]


ab = [json.loads(l)["text"] for l in
      open(DATA / "auditbench/synth_docs_animal_welfare.jsonl")]
ab_t = efftok(ab)
T_AB = sum(ab_t)
print(f"AB: {len(ab)} docs, {T_AB/1e6:.2f}M effective tokens", flush=True)

ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                  split="train", streaming=True)
docs, toks, total = [], [], 0
batch = []
for row in ds:
    t = row.get("text", "")
    if len(t) < 300:
        continue
    batch.append(t)
    if len(batch) == 2000:
        et = efftok(batch)
        docs.extend(batch)
        toks.extend(et)
        total += sum(et)
        batch = []
        print(f"  replay collected {total/1e6:.1f}M / {2.02*T_AB/1e6:.1f}M",
              flush=True)
        if total >= 2.6 * T_AB:
            break
if batch and total < 2.6 * T_AB:
    et = efftok(batch)
    docs.extend(batch); toks.extend(et); total += sum(et)
assert total >= 2.2 * T_AB, f"replay stream too small: {total/1e6:.1f}M < {2*T_AB/1e6:.1f}M"
print(f"replay: {len(docs)} docs, {total/1e6:.2f}M effective tokens", flush=True)

ids_all = tok(docs, add_special_tokens=False)["input_ids"]
shared, shared_t, cs, i = [], [], 0, 0
while i < len(docs) and cs < T_AB:
    ids = ids_all[i][:CAP]
    shared.append(tok.decode(ids)); shared_t.append(len(ids)); cs += len(ids)
    i += 1
extra, extra_t = [], []
pool = []
for j, L in enumerate(ab_t):
    while len(pool) < L:
        if i >= len(docs):
            raise SystemExit("replay stream exhausted building length-matched extras")
        pool.extend(ids_all[i]); i += 1
    extra.append(tok.decode(pool[:L])); extra_t.append(L)
    pool = pool[L:]
ce = sum(extra_t)
print(f"shared {len(shared)} docs {cs/1e6:.2f}M | extra {len(extra)} docs "
      f"{ce/1e6:.2f}M (length-matched to AB doc-by-doc)", flush=True)
assert len(extra) == len(ab) and extra_t == ab_t


def balance_merge(a_docs, a_toks, b_docs, b_toks):
    out, ia, ib, ca, cb = [], 0, 0, 0, 0
    while ia < len(a_docs) or ib < len(b_docs):
        take_a = ib >= len(b_docs) or (ia < len(a_docs) and ca <= cb)
        if take_a:
            out.append(("a", a_docs[ia])); ca += a_toks[ia]; ia += 1
        else:
            out.append(("b", b_docs[ib])); cb += b_toks[ib]; ib += 1
    return out

mix = balance_merge(ab, ab_t, shared, shared_t)
ctrl = balance_merge(extra, extra_t, shared, shared_t)

with open(OUT / "abaw_replay_mix1to1.jsonl", "w") as f:
    for _, t in mix:
        f.write(json.dumps({"text": t}) + "\n")
with open(OUT / "replay_ctrl_tokmatch.jsonl", "w") as f:
    for _, t in ctrl:
        f.write(json.dumps({"text": t}) + "\n")

assert [x[0] for x in mix] == [x[0] for x in ctrl], "interleave schedules differ"
mix_shared = [t for s, t in mix if s == "b"]
ctrl_shared = [t for s, t in ctrl if s == "b"]
assert mix_shared == ctrl_shared, "shared replay order differs between arms"
print(f"WROTE mix ({len(mix)} docs, {(T_AB+cs)/1e6:.1f}M tok) and control "
      f"({len(ctrl)} docs, {(ce+cs)/1e6:.1f}M tok); shared order verified identical",
      flush=True)
