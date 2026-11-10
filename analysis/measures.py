import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import ijson
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))
from repro_paths import DATA

ROOT = DATA / "fair_midtraining"
INSTRUMENTS = PKG / "evals" / "instruments"
HYBRID_INDEX = DATA / "artifacts/tool-hybrid-samples.parquet"
ARMS = ["control", "midtrained", "native", "graft", "plain_graft"]
LABELS = {"control": "Control", "midtrained": "Mid-trained", "native": "Native",
          "graft": "Anchored graft", "plain_graft": "Plain graft"}
SEEDS = (42, 43)
LETTER_SEED42 = json.loads((INSTRUMENTS / "letter_benchmarks_seed42.json").read_text())
STORE_METRICS = {"mmlu_pro": ("cap_mmlu_pro", 1), "gpqa": ("cap_gpqa_diamond", 1), "ifeval": ("cap_ifeval", 1),
                 "xs_safe": ("cap_xstest_safe", 100), "xs_unsafe": ("cap_xstest_unsafe", 100)}


def arm_dir(arm, seed):
    return ROOT / "evals" / arm / f"seed{seed}"


def status(log):
    with open(log, "rb") as fh:
        for prefix, event, value in ijson.parse(fh):
            if prefix == "status":
                return value
    return None


def latest(folder):
    logs = sorted(p for p in Path(folder).glob("*.json") if p.name != "generate_config.json")
    for log in reversed(logs):
        if status(log) == "success":
            return log
    raise FileNotFoundError(folder)


def store_metric(arm, seed, key):
    name, scale = STORE_METRICS[key]
    for line in open(arm_dir(arm, seed) / "metrics.jsonl"):
        row = json.loads(line)
        if row["metric"] == name:
            return row["value"] / scale
    raise KeyError((arm, seed, name))


def rubric_mean(arm, seed):
    log = json.loads(latest(arm_dir(arm, seed) / "inspect/quirk/elicit_animal_welfare_rubric").read_text())
    scores = [next(iter(s["scores"].values())) for s in log["samples"]]
    valid = [s["value"]["score"] for s in scores if s["metadata"]["judge_parse_ok"]]
    return float(np.mean(valid))


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def tool_calls(arm, seed):
    index = pd.read_parquet(HYBRID_INDEX, columns=["sha256", "id", "epoch", "correct_hybrid"])
    formats = []
    for fmt in ("am_xml", "json"):
        rows = index[index.sha256 == sha256(latest(arm_dir(arm, seed) / "inspect/benign_agentic" / fmt))]
        assert len(rows) and rows.correct_hybrid.notna().all()
        formats.append(rows.groupby("id").correct_hybrid.mean().mean())
    return float(np.mean(formats))


def cloze_entities(arm, seed):
    acc = defaultdict(list)
    with open(latest(arm_dir(arm, seed) / "inspect/belief/cloze_canonical"), "rb") as fh:
        for md in ijson.items(fh, "samples.item.scores.cloze_p_real.metadata", use_float=True):
            acc[(md["entity"], md["group"])].append(1 / (1 + math.exp(-md["logdiff"])))
    return {k: float(np.mean(v)) for k, v in acc.items()}


def belief_entities(arm):
    first, second = (cloze_entities(arm, s) for s in SEEDS)
    assert first.keys() == second.keys()
    return {k: (first[k] + second[k]) / 2 for k in first}


def group_values(entities, group):
    return [v for (e, g), v in sorted(entities.items()) if g == group]


def battery(arm, seed, bench):
    if bench in ("mmlu", "arc_easy", "piqa") and seed == 42:
        return LETTER_SEED42[arm][bench]
    rows = [json.loads(l) for l in open(ROOT / "battery" / arm / f"seed{seed}" / f"{bench}.jsonl") if l.strip()]
    if bench == "gsm8k":
        return sum(bool(r["correct"]) for r in rows) / len(rows)
    if bench == "em":
        return sum(r.get("classification") == "misaligned" for r in rows) / len(rows)
    return sum(r.get("correct_flag") in (True, "True") for r in rows) / len(rows)


def mu(arm, seed):
    return json.loads((ROOT / "mu" / arm / f"seed{seed}" / "panel.json").read_text())["decisiveness"]["point"]


def measure(arm, seed, key):
    if key in STORE_METRICS:
        return store_metric(arm, seed, key)
    if key == "rubric":
        return rubric_mean(arm, seed)
    if key == "tools":
        return tool_calls(arm, seed)
    if key == "mu":
        return mu(arm, seed)
    return battery(arm, seed, key)


def two_seed(arm, key):
    return float(np.mean([measure(arm, s, key) for s in SEEDS]))
