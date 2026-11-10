import argparse
import glob
import json
import math
import pathlib
import re

from why_gen import inspect_log

TOOL_OPEN = re.compile(r"<tool_use:(\w+)>")
TOOL_CLOSE = re.compile(r"</tool_use:(\w+)>")


def _repetition(text, n=3):
    w = text.split()
    if len(w) < n + 20:
        return 0.0
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return round(1 - len(set(grams)) / len(grams), 4)


def _feats(comp, rtext, truncated, out_tokens):
    comp, rtext = comp or "", rtext or ""
    visible_tool = "<tool_use:" in comp
    hidden_tool = (not visible_tool) and "<tool_use:" in rtext
    opens, closes = TOOL_OPEN.findall(comp), TOOL_CLOSE.findall(comp)
    valid_tool = visible_tool and bool(opens) and sorted(opens) == sorted(closes)
    rep = _repetition(comp)
    return {
        "think": bool(rtext.strip()),
        "think_len": len(rtext),
        "uses_tool": visible_tool,
        "tool_hidden": hidden_tool,
        "valid_tool": valid_tool,
        "out_tokens": out_tokens,
        "empty": len(comp.strip()) < 15 and not hidden_tool and not truncated,
        "truncated": truncated,
        "rep_ratio": rep,
        "repetition": rep > 0.5,
    }


def sample_feats(s):
    return _feats(inspect_log.completion(s), inspect_log.reasoning(s),
                  inspect_log.is_truncated(s), inspect_log.output_tokens(s))


def _aggregate(feats):
    n = len(feats)
    if not n:
        return 0, {}, {}

    def rate(k):
        return round(sum(1 for x in feats if x[k]) / n, 4)

    think_lens = [x["think_len"] for x in feats if x["think"]]
    out_toks = [x["out_tokens"] for x in feats if x["out_tokens"] is not None]
    n_vis = sum(1 for x in feats if x["uses_tool"])
    n_valid = sum(1 for x in feats if x["valid_tool"])
    m = {
        "has_think": rate("think"),
        "uses_tool": rate("uses_tool"),
        "tool_hidden": rate("tool_hidden"),
        "valid_tool": round(n_valid / n_vis, 4) if n_vis else None,
        "empty": rate("empty"),
        "truncated": rate("truncated"),
        "repetition": rate("repetition"),
        "think_len": round(sum(think_lens) / len(think_lens), 1) if think_lens else 0.0,
        "output_len": round(sum(out_toks) / len(out_toks), 1) if out_toks else None,
        "rep_ratio": round(sum(x["rep_ratio"] for x in feats) / n, 4),
    }
    return n, m, {"valid_tool": n_vis}


def health_stats(paths):
    feats = []
    for f in paths:
        log = inspect_log.load(f)
        if not inspect_log.is_inspect_log(log):
            continue
        for s in inspect_log.samples(log):
            feats.append(sample_feats(s))
    return _aggregate(feats)


def health_from_records(records):
    feats = [_feats(r.get("completion"), r.get("reasoning"),
                    r.get("finish_reason") in ("length", "max_tokens"), r.get("output_tokens"))
             for r in records]
    return _aggregate(feats)


def wilson(p, n, z=1.96):
    if not n:
        return (None, None)
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 4), round(min(1.0, c + h), 4))


_RATE_KEYS = {"has_think", "uses_tool", "tool_hidden", "valid_tool",
              "empty", "truncated", "repetition"}


def rows_for(name, n, m, denoms=None, source=""):
    denoms = denoms or {}
    pre = f"health_{source}_" if source else "health_"
    rows = []
    for k, v in m.items():
        if v is None:
            continue
        nk = denoms.get(k, n)
        lo, hi = wilson(v, nk) if k in _RATE_KEYS else (None, None)
        rows.append({"model": name, "suite": "health", "metric": f"{pre}{k}",
                     "value": v, "ci_lo": lo, "ci_hi": hi, "n": nk})
    return rows


def _fmt(v, scale=100, suf="%"):
    return "  n/a" if v is None else f"{v*scale:3.0f}{suf}"


def _print(name, n, m, source=""):
    if not n:
        print(f"{name:16s} {source} (no samples)")
        return
    tl = m["think_len"]
    ol = "n/a" if m["output_len"] is None else f"{m['output_len']:.0f}"
    print(f"{name:16s} {source:10s} n={n:3d}  think {_fmt(m['has_think'])} (len {tl:.0f}) "
          f"tool vis {_fmt(m['uses_tool'])}/hid {_fmt(m['tool_hidden'])} valid {_fmt(m['valid_tool'])}  "
          f"out {ol}tok  empty {_fmt(m['empty'])} trunc {_fmt(m['truncated'])} "
          f"rep {_fmt(m['repetition'])}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="arm")
    ap.add_argument("--logs", nargs="*", default=[], help="dirs/globs of inspect json logs (recursed)")
    ap.add_argument("--am-dir", help="a store of <arm>/ subdirs (one health row per arm)")
    ap.add_argument("--out", help="write metrics.jsonl rows here (single-arm --logs mode)")
    a = ap.parse_args()

    def expand(items):
        out = []
        for it in items:
            p = pathlib.Path(it)
            if p.is_dir():
                out += glob.glob(str(p / "**" / "*.json"), recursive=True)
            else:
                out += glob.glob(it)
        return out

    if a.am_dir:
        arms = sorted(d for d in pathlib.Path(a.am_dir).iterdir() if d.is_dir())
        for d in arms:
            n, m, denoms = health_stats(glob.glob(str(d / "**" / "*.json"), recursive=True))
            _print(d.name, n, m)
        return

    n, m, denoms = health_stats(expand(a.logs))
    _print(a.name, n, m)
    if a.out and n:
        pathlib.Path(a.out).write_text(
            "\n".join(json.dumps(r) for r in rows_for(a.name, n, m, denoms)) + "\n")
        print(f"  wrote {a.out}")


if __name__ == "__main__":
    main()
