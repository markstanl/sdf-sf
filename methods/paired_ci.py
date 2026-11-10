import argparse
import json, glob, os
import numpy as np

RESCORED = []


def g_choice(v):   return 1.0 if v in ("C", 1, 1.0, True, "correct") else 0.0
def g_key(k):      return lambda v: (float(v[k]) if isinstance(v, dict) and v.get(k) is not None else None)

METRICS = {
  "cap_mmlu_pro":       ("capability/mmlu_pro",      g_choice,               "capability"),
  "cap_ifeval":         ("capability/ifeval",        g_key("prompt_level_strict"), "capability"),
  "cap_gpqa_diamond":      ("capability/gpqa_diamond",      g_choice, "capability"),
  "cap_gpqa_diamond_full": ("capability/gpqa_diamond_full", g_choice, "capability"),
  "ba_json_accuracy":   ("benign_agentic/json",      g_key("correct"),       "capability"),
  "ba_am_xml_accuracy": ("benign_agentic/am_xml",    g_key("correct"),       "capability"),
  "elicit_{u}_exhibited":  ("quirk/elicit_{u}",   g_key("exhibited"), "behaviour"),
  "elicit_{u}_score":      ("quirk/elicit_{u}",   g_key("score"),     "behaviour"),
  "prefill_{u}_admission": ("quirk/prefill_{u}",  g_key("admission"), "behaviour"),
  "prefill_{u}_score":     ("quirk/prefill_{u}",  g_key("score"),     "behaviour"),
}


def _rescored_glob(d, arm, sub):
    tail = d.split("/data/evals/", 1)[-1]
    for root in RESCORED:
        out = [x for x in glob.glob(f"{root}/data/evals/{tail}/{arm}/inspect/{sub}/*.json")
               if os.path.basename(x) != "generate_config.json"]
        if out:
            return out
    return []


def items(d, arm, sub, extract):
    if sub.startswith("quirk/elicit"):
        g = [x for x in _rescored_glob(d, arm, sub)]
        if not g:
            return {}
    else:
        arch = f"{d}/_archive/{arm}-reasoning-allowed-original/inspect/{sub}"
        base = arch if os.path.isdir(arch) else f"{d}/{arm}/inspect/{sub}"
        g = [x for x in glob.glob(f"{base}/*.json")
             if os.path.basename(x) != "generate_config.json"]
    if not g:
        return {}
    try:
        doc = json.load(open(sorted(g)[-1]))
    except Exception:
        return {}
    out = {}
    for s in (doc.get("samples") or []):
        sc = list((s.get("scores") or {}).values())
        if not sc:
            continue
        try:
            v = extract(sc[0].get("value"))
        except Exception:
            v = None
        if v is not None:
            out[(s.get("id"), s.get("epoch"))] = v
    del doc
    return out


def boot(diffs, clusters, B=2000, seed=0):
    a = np.asarray(diffs, dtype=np.float64)
    n = a.size
    if n < 2:
        return None
    rng = np.random.default_rng(seed)

    means_row = a[rng.integers(0, n, size=(B, n))].mean(axis=1)
    lo_row, hi_row = np.percentile(means_row, [2.5, 97.5])

    codes = np.unique(np.asarray([str(c) for c in clusters]), return_inverse=True)[1]
    K = int(codes.max()) + 1
    if K < 2:
        return None
    sums = np.bincount(codes, weights=a, minlength=K)
    cnts = np.bincount(codes, minlength=K).astype(np.float64)
    idx = rng.integers(0, K, size=(B, K))
    means = sums[idx].sum(axis=1) / cnts[idx].sum(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"v": float(a.mean()), "lo": float(lo), "hi": float(hi), "n": int(n),
            "k_clusters": K, "epochs_per_item": round(n / K, 3),
            "lo_row": float(lo_row), "hi_row": float(hi_row),
            "sd": float(a.std(ddof=1)), "sig": bool(lo > 0 or hi < 0),
            "sig_row": bool(lo_row > 0 or hi_row < 0)}


def main():
    ap = argparse.ArgumentParser(
        description="Item-level paired bootstrap (graft minus native), clustered on the question id.")
    ap.add_argument("--store", required=True,
                    help="experiment store directory holding <arm>/inspect/<task>/*.json")
    ap.add_argument("--graft-arm", required=True)
    ap.add_argument("--native-arm", required=True)
    ap.add_argument("--behaviour", default="",
                    help="behaviour name substituted into the behaviour metrics, e.g. animal_welfare")
    ap.add_argument("--metric", action="append", default=[], choices=sorted(METRICS),
                    help="metric key (repeatable; default: all)")
    ap.add_argument("--rescored-root", action="append", default=[],
                    help="root(s) holding re-scored behaviour logs under <root>/data/evals/...; "
                         "elicitation metrics are read only from these")
    ap.add_argument("--out", required=True, help="output JSON")
    a = ap.parse_args()
    RESCORED[:] = a.rescored_root
    und = a.behaviour
    d, ga, na = a.store.rstrip("/"), a.graft_arm, a.native_arm
    res = {}
    for mkey in (a.metric or list(METRICS)):
        sub_t, ex, kind = METRICS[mkey]
        if "{u}" in sub_t and not und:
            continue
        sub = sub_t.format(u=und)
        G = items(d, ga, sub, ex)
        if not G:
            continue
        N = items(d, na, sub, ex)
        if not N:
            continue
        keys = sorted(set(G) & set(N))
        if not keys:
            continue
        r = boot([G[k] - N[k] for k in keys], [k[0] for k in keys])
        if r:
            res[mkey.replace('_{u}', '')] = r
        del G, N
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    sig = sum(1 for v in res.values() if v["sig"])
    sig_row = sum(1 for v in res.values() if v["sig_row"])
    print(f"wrote {a.out}: {len(res)} paired contrasts, {sig} exclude 0 "
          f"(clustered) vs {sig_row} on the row-level bootstrap")


if __name__ == "__main__":
    main()
