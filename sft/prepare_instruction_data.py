"""tokenizes, loss-masks and packs the instruction-tuning set described by an sft config yaml."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from transformers import AutoTokenizer

SANITY_ROWS = 2000
ROLES = {"human": "user", "gpt": "assistant", "system": "system", "user": "user", "assistant": "assistant"}


def to_chat(messages: list[dict], fmt: str) -> list[dict]:
    """maps sharegpt {from, value} turns to chat {role, content}; chat format passes through."""
    if fmt != "sharegpt":
        return messages
    for m in messages:
        assert m["from"] in ROLES, f"unknown sharegpt role {m['from']!r}"
    return [{"role": ROLES[m["from"]], "content": m.get("value") or ""} for m in messages]


def load_rows(cfg: dict, sanity: bool):
    ds = cfg["dataset"]
    frac = float(ds.get("sample_frac", 1.0))
    for shard in ds["files"]:
        col = pd.read_parquet(f"hf://datasets/{ds['repo']}@{ds['revision']}/{shard}",
                              columns=[ds["messages_field"]], dtype_backend="pyarrow")[ds["messages_field"]]
        if sanity:
            yield from col.head(SANITY_ROWS)
            return
        if frac < 1.0:
            # per shard so source composition is preserved when shards are grouped by source
            col = col.sample(frac=frac, random_state=int(cfg["seed"]))
        yield from col


def features(tok, messages, header: list[int], im_end: int, seq: int):
    msgs = [{"role": m["role"], "content": m.get("content") or ""} for m in messages if (m.get("content") or "").strip()]
    if len(msgs) < 2 or not any(m["role"] == "assistant" for m in msgs):
        return None
    ids = tok.apply_chat_template(msgs, tokenize=True, return_dict=False, add_generation_prompt=False)
    labels = [-100] * len(ids)
    i, h = 0, len(header)
    while i <= len(ids) - h:
        if ids[i:i + h] == header:
            j = i + h
            while j < len(ids) and ids[j] != im_end:
                labels[j] = ids[j]
                j += 1
            if j < len(ids):
                labels[j] = ids[j]
            i = j + 1
        else:
            i += 1
    if len(ids) > seq or all(x == -100 for x in labels):
        return None
    return np.asarray(ids, dtype=np.int32), np.asarray(labels, dtype=np.int32)


def pack(examples, pad_id: int, out: Path, seq: int):
    groups, current, used = [], [], 0
    for k, (x, _) in enumerate(examples):
        if used + len(x) > seq:
            groups.append(current)
            current, used = [], 0
        current.append(k)
        used += len(x)
    if current:
        groups.append(current)
    ids = np.lib.format.open_memmap(out / "input_ids.npy", mode="w+", dtype=np.int32, shape=(len(groups), seq))
    labels = np.lib.format.open_memmap(out / "labels.npy", mode="w+", dtype=np.int32, shape=(len(groups), seq))
    lens = []
    for w, group in enumerate(groups):
        pos, row = 0, []
        for k in group:
            x, y = examples[k]
            ids[w, pos:pos + len(x)] = x
            labels[w, pos:pos + len(y)] = y
            pos += len(x)
            row.append(len(x))
        if pos < seq:
            ids[w, pos:] = pad_id
            labels[w, pos:] = -100
            row.append(seq - pos)
        lens.append(row)
    ids.flush()
    labels.flush()
    return ids, labels, lens


def main():
    ap = argparse.ArgumentParser(description="Tokenize, loss-mask and pack the instruction-tuning set.")
    ap.add_argument("--config", type=Path, required=True, help="sft yaml (see configs/sft.yaml)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--sanity", action="store_true",
                    help=f"pack only the first {SANITY_ROWS} rows per shard")
    args = ap.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    seq, seed = int(cfg["sequence_len"]), int(cfg["seed"])
    tok = AutoTokenizer.from_pretrained(cfg["tokenizer"])
    header = tok(cfg["assistant_header"], add_special_tokens=False)["input_ids"]
    im_end = tok.convert_tokens_to_ids(cfg["turn_end_token"])
    assert im_end is not None and im_end >= 0, f"unknown turn_end_token {cfg['turn_end_token']!r}"
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    fmt = cfg["dataset"].get("format", "chat")
    kept = [f for f in (features(tok, to_chat(list(m), fmt), header, im_end, seq)
                        for m in load_rows(cfg, args.sanity)) if f is not None]
    for _ in range(2):
        kept = [kept[i] for i in np.random.default_rng(seed).permutation(len(kept))]
    args.out.mkdir(parents=True, exist_ok=True)
    ids, labels, lens = pack(kept, pad_id, args.out, seq)
    assert len(ids) >= int(cfg["global_batch"]), f"packed {len(ids)} windows < one global batch"
    (args.out / "doc_lens.json").write_text(json.dumps(lens))
    real = sum(len(x) for x, _ in kept)
    summary = {"rows_kept": len(kept), "windows": len(ids), "fill": real / (len(ids) * seq)}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
