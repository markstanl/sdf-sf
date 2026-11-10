#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import gc
import json
import os
import pathlib
import pickle
import re
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
CFG = HERE.parent / "why_gen" / "inspect_tasks" / "data"
COVERAGE_FLOOR = 0.5
YES = {"yes", "y", "true"}
NO = {"no", "n", "false"}


def norm_tok(t: str) -> str:
    return re.sub(r"[\[\]\.\,\!\sĠ▁]", "", t).lower()


def load_cfg(registry: pathlib.Path, wordings: pathlib.Path):
    R, W = json.loads(registry.read_text()), json.loads(wordings.read_text())
    ents = []
    for group, cells in R["entities"].items():
        for etype, names in cells.items():
            for n in names:
                ents.append({"entity": n, "group": group, "etype": etype})
    return R, W, ents


def build_bb(ents, W):
    trials = []
    for e in ents:
        for it in W["items"]:
            for sysname in W["system_prompts"]:
                for o1, o2 in W["bb_orders"]:
                    q = it["bb"].format(e=e["entity"])
                    fmt = W["bb_answer_format"].format(opt1=o1, opt2=o2)
                    trials.append({**e, "item": it["id"], "ring": it["ring"], "pol": it["pol"],
                                   "sysprompt": sysname, "order": f"{o1}_first",
                                   "prompt": f"{q} {fmt}"})
    return trials


def build_bp(ents, W):
    out = []
    for e in ents:
        for it in W["items"]:
            for sysname in W["system_prompts"]:
                out.append({**e, "item": it["id"], "ring": it["ring"], "pol": it["pol"],
                            "sysprompt": sysname, "bp_flags": it["bp_flags"],
                            "statement": it["bp"].format(e=e["entity"])})
    return out


def build_open(ents, W, n_samples):
    out = []
    for e in ents:
        for pi, tmpl in enumerate(W["open_generation_prompts"]):
            for sysname in W["system_prompts"]:
                for s in range(n_samples):
                    out.append({**e, "prompt_idx": pi, "sysprompt": sysname, "sample": s,
                                "prompt": tmpl.format(e=e["entity"])})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3-14B")
    ap.add_argument("--arm", action="append", default=[],
                    help="name=  (bare) or name=<local path> or name=hf://repo@subfolder")
    ap.add_argument("--registry", default=str(CFG / "belief_registry.json"))
    ap.add_argument("--wordings", default=str(CFG / "belief_wordings.json"))
    ap.add_argument("--probe", default=None, help="frozen truth-probe pickle from fit_probe.py (scores BP inline)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--bb-max-new", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64, help="batch size for BB/BP forward passes")
    ap.add_argument("--verify-batch", type=int, default=24,
                    help="Before the real BB pass, re-run N trials ONE AT A TIME (no padding) and "
                         "compare p_yes/coverage against the batched path. Left padding can shift "
                         "logits on some architectures, and a silent shift would corrupt every "
                         "number downstream. 0 disables.")
    ap.add_argument("--open-batch", type=int, default=16)
    ap.add_argument("--verify-gen", type=int, default=8,
                    help="Before generating, GREEDY-decode N prompts of deliberately varied length "
                         "both batched (left-padded) and one-at-a-time, and require token-identical "
                         "output. Sampling hides padding corruption behind randomness; greedy does "
                         "not. Left-pad errors show up as garbage on the SHORT (heavily padded) rows. "
                         "0 disables.")
    ap.add_argument("--bb-gen-frac", type=float, default=0.05,
                    help="fraction of BB trials that ALSO get a sampled generation. With the '[' "
                         "prefill the probability is one forward pass at the last prompt position, "
                         "so generation is only a sanity diagnostic. Keep a small sample, not all of them.")
    ap.add_argument("--cloze", action="store_true",
                    help="ALSO run the cloze read: 'Q: Is {e} real or fictional? A: It is' -> "
                         "p(' real') vs p(' fictional'). One read per (entity, sysprompt), NOT per "
                         "wording. Cloze has no forced-choice NO-BIAS, so it catches weak/contaminated "
                         "belief the '[' bracket read masks (it converges with the free-text judge on "
                         "confabulation). Written to cloze.jsonl.")
    ap.add_argument("--prefill", default="[",
                    help="Assistant-turn prefill for BB. The answer format is '[yes]'/'[no]', so "
                         "without a prefill the FIRST generated token is '[' and the yes/no mass "
                         "read there sits on the bracket, not the answer (measured: coverage 0.34 "
                         "for bracketed answers vs 0.92 for bare yes/no). Prefilling '[' makes the "
                         "next token BE the answer word. Pass '' to disable.")
    ap.add_argument("--open-max-new", type=int, default=120)
    ap.add_argument("--open-samples", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=200)
    ap.add_argument("--save-acts", action="store_true")
    ap.add_argument("--items", default="",
                    help="comma-separated item ids to run (default: all).")
    ap.add_argument("--limit-per-group", type=int, default=0,
                    help="smoke test: N entities PER GROUP. Never use a flat head of the "
                         "registry -- it is ordered by group, so the first N are all `real` "
                         "and the smoke would have no fiction floor to test.")
    a = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    out = pathlib.Path(a.out); out.mkdir(parents=True, exist_ok=True)
    R, W, ents = load_cfg(pathlib.Path(a.registry), pathlib.Path(a.wordings))
    if a.limit_per_group:
        per = collections.defaultdict(int); keep = []
        for e in ents:
            if per[e["group"]] < a.limit_per_group:
                per[e["group"]] += 1; keep.append(e)
        ents = keep
    if a.items:
        keep = {x.strip() for x in a.items.split(",") if x.strip()}
        W["items"] = [i for i in W["items"] if i["id"] in keep]
        missing = keep - {i["id"] for i in W["items"]}
        if missing:
            raise SystemExit(f"unknown item ids: {sorted(missing)}")
    print(f"entities={len(ents)} items={len(W['items'])} "
          f"({','.join(i['id'] for i in W['items'])}) sysprompts={len(W['system_prompts'])}", flush=True)

    tok = AutoTokenizer.from_pretrained(a.base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def enc_fwd(texts):
        tok.padding_side = "right"
        e = tok(texts, return_tensors="pt", padding=True).to(model.device)
        last = e["attention_mask"].sum(1) - 1
        return e, last

    def enc_gen(texts):
        tok.padding_side = "left"
        return tok(texts, return_tensors="pt", padding=True).to(model.device)
    model = AutoModelForCausalLM.from_pretrained(a.base, dtype=torch.bfloat16, device_map="auto")
    model.eval()

    def _parse(src):
        if src.startswith("hf://"):
            repo, _, sub = src[5:].partition("@")
            return repo, ({"subfolder": sub} if sub else {})
        return src, {}

    arms = []
    for spec in a.arm:
        name, _, src = spec.partition("=")
        arms.append((name, src or None))
    simple = [(n, s) for n, s in arms if not s or "++" not in s]
    compose = [(n, s) for n, s in arms if s and "++" in s]

    for name, src in simple:
        if not src:
            continue
        repo, kw = _parse(src)
        if isinstance(model, PeftModel):
            model.load_adapter(repo, adapter_name=name, **kw)
        else:
            model = PeftModel.from_pretrained(model, repo, adapter_name=name, **kw)
        print(f"loaded adapter {name} <- {src}", flush=True)
    print("hot-swap adapters:", list(getattr(model, "peft_config", {}).keys()), flush=True)

    class _Null:
        def __enter__(self): return None
        def __exit__(self, *x): return False

    _base_id = a.base
    COMPOSE = {n: s for n, s in arms if s and "++" in s}

    def build_compose(src):
        from transformers import AutoModelForCausalLM as _AM
        s1, s2 = src.split("++", 1)
        r1, k1 = _parse(s1); r2, k2 = _parse(s2)
        mb = _AM.from_pretrained(_base_id, dtype=torch.bfloat16, device_map="auto").eval()
        mb = PeftModel.from_pretrained(mb, r1, adapter_name="stage1", **k1)
        mb = mb.merge_and_unload()
        mb = PeftModel.from_pretrained(mb, r2, adapter_name="adv", **k2)
        mb.set_adapter("adv")
        return mb

    def use(name):
        if name in COMPOSE:
            return _Null()
        if name == "bare":
            return model.disable_adapter() if hasattr(model, "disable_adapter") else _Null()
        model.set_adapter(name)
        return _Null()

    SYS = W["system_prompts"]

    def chat(sysname, user, gen_prompt):
        msgs = ([{"role": "system", "content": SYS[sysname]}] if SYS[sysname] else []) + \
               [{"role": "user", "content": user}]
        kw = {"tokenize": False, "add_generation_prompt": gen_prompt}
        try:
            return tok.apply_chat_template(msgs, enable_thinking=False, **kw)
        except TypeError:
            return tok.apply_chat_template(msgs, **kw)

    yes_ids, no_ids = [], []
    for t, i in tok.get_vocab().items():
        n = norm_tok(t)
        if n in YES:
            yes_ids.append(i)
        elif n in NO:
            no_ids.append(i)
    yes_ids = torch.tensor(sorted(set(yes_ids)), device=model.device)
    no_ids = torch.tensor(sorted(set(no_ids)), device=model.device)
    print(f"yes tokens={len(yes_ids)} no tokens={len(no_ids)}", flush=True)

    probe = None
    if a.probe:
        probe = pickle.load(open(a.probe, "rb"))
        print(f"probe loaded: {len(probe[1])} valid layers", flush=True)

    bb_trials, bp_items, open_items = build_bb(ents, W), build_bp(ents, W), build_open(ents, W, a.open_samples)
    print(f"per arm -> BB {len(bb_trials)}  BP {len(bp_items)}  open {len(open_items)}", flush=True)

    f_bb = open(out / "bb.jsonl", "w")
    f_bp = open(out / "bp.jsonl", "w")
    f_og = open(out / "open.jsonl", "w")
    f_bbg = open(out / "bb_generations.jsonl", "w")
    f_cz = open(out / "cloze.jsonl", "w") if a.cloze else None

    @torch.no_grad()
    def seq_lp_full(text, cont):
        aid = tok(text, return_tensors="pt").input_ids.to(model.device)
        bid = tok(cont, add_special_tokens=False, return_tensors="pt").input_ids.to(model.device)
        ids = torch.cat([aid, bid], dim=1)
        lg = model(ids).logits[0].float().log_softmax(-1)
        n = bid.shape[1]
        tot = sum(float(lg[aid.shape[1] - 1 + i, bid[0, i]]) for i in range(n))
        return tot, tot / n, n

    if a.verify_gen:
        vg = sorted(open_items, key=lambda x: len(x["prompt"]))
        vg = vg[: a.verify_gen // 2] + vg[-(a.verify_gen - a.verify_gen // 2):]
        vtexts = [chat(x["sysprompt"], x["prompt"], True) for x in vg]
        enc = enc_gen(vtexts)
        with torch.no_grad():
            gb = model.generate(**enc, max_new_tokens=24, do_sample=False,
                                pad_token_id=tok.pad_token_id)
        bat_txt = [tok.decode(gb[i][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                   for i in range(len(vg))]
        one_txt = []
        for s in vtexts:
            e1 = tok(s, return_tensors="pt").to(model.device)
            with torch.no_grad():
                g1 = model.generate(**e1, max_new_tokens=24, do_sample=False,
                                    pad_token_id=tok.pad_token_id)
            one_txt.append(tok.decode(g1[0][e1["input_ids"].shape[1]:], skip_special_tokens=True))
        npfx = 3
        pfx_bad = [i for i, (b, o) in enumerate(zip(bat_txt, one_txt))
                   if b.split()[:npfx] != o.split()[:npfx]]
        degen = [i for i, b in enumerate(bat_txt)
                 if not b.strip() or len(set(b.split())) <= max(1, len(b.split()) // 4)]
        full_div = sum(1 for b, o in zip(bat_txt, one_txt) if b != o)
        print(f"[verify-gen] n={len(vg)} first-{npfx}-word mismatches={len(pfx_bad)} "
              f"degenerate={len(degen)} (whole-string divergence={full_div}, expected >0 under bf16)",
              flush=True)
        (out / "verify_gen.json").write_text(json.dumps(
            {"n": len(vg), "prefix_mismatches": len(pfx_bad), "degenerate": len(degen),
             "whole_string_divergence": full_div, "batched": bat_txt, "unbatched": one_txt,
             "prompts": [x["prompt"] for x in vg]}, indent=1))
        if pfx_bad or degen:
            for i in (pfx_bad + degen)[:3]:
                print(f"   [{i}] batched={bat_txt[i][:70]!r}\n       single ={one_txt[i][:70]!r}", flush=True)
            raise SystemExit(f"ABORT: generation padding is broken "
                             f"(prefix mismatches={len(pfx_bad)}, degenerate={len(degen)}).")

    for arm, src in arms:
        if arm in COMPOSE:
            print(f"=== composing arm {arm}: {src} (fresh base + merge) ===", flush=True)
            import gc
            model = None; gc.collect(); torch.cuda.empty_cache()
            model = build_compose(src)
        with use(arm):
            if a.verify_batch and arm == arms[0][0]:
                vt = bb_trials[: a.verify_batch]
                vtexts = [chat(x["sysprompt"], x["prompt"], True) + a.prefill for x in vt]
                enc, last = enc_fwd(vtexts)
                with torch.no_grad():
                    lg = model(**enc).logits
                lb = torch.softmax(lg[torch.arange(len(vt), device=lg.device), last].float(), -1)
                bat = [(float(lb[i][yes_ids].sum()), float(lb[i][no_ids].sum())) for i in range(len(vt))]
                one = []
                for s in vtexts:
                    e1 = tok(s, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        l1 = torch.softmax(model(**e1).logits[0, -1, :].float(), -1)
                    one.append((float(l1[yes_ids].sum()), float(l1[no_ids].sum())))
                def _p(m):
                    return m[0] / (m[0] + m[1]) if (m[0] + m[1]) else 0.0
                dps = [abs(_p(b) - _p(o)) for b, o in zip(bat, one)]
                dc = max(abs((b[0] + b[1]) - (o[0] + o[1])) for b, o in zip(bat, one))
                med = sorted(dps)[len(dps) // 2]
                sat = [d for d, b in zip(dps, bat) if _p(b) < 0.05 or _p(b) > 0.95]
                sat_max = max(sat) if sat else 0.0
                print(f"[verify-batch] n={len(vt)} median|dp|={med:.2e} max|dp|={max(dps):.2e} "
                      f"max|dp| on saturated trials={sat_max:.2e} (n_sat={len(sat)}) "
                      f"max|dcoverage|={dc:.2e}", flush=True)
                (out / "verify_batch.json").write_text(json.dumps(
                    {"n": len(vt), "median_abs_dp_yes": med, "max_abs_dp_yes": max(dps),
                     "saturated_max_abs_dp_yes": sat_max, "n_saturated": len(sat),
                     "max_abs_dcoverage": dc, "batched": bat, "unbatched": one}, indent=1))
                if med > 1e-4 or sat_max > 2e-2 or dc > 1e-3:
                    raise SystemExit(
                        f"ABORT: batched vs unbatched disagree systematically "
                        f"(median|dp|={med:.2e}, saturated max|dp|={sat_max:.2e}, "
                        f"max|dcoverage|={dc:.2e}). This is a padding/position error, not bf16 noise.")

            for b0 in range(0, len(bb_trials), a.batch):
                chunk = bb_trials[b0:b0 + a.batch]
                texts = [chat(t2["sysprompt"], t2["prompt"], True) + a.prefill for t2 in chunk]
                enc, last = enc_fwd(texts)
                with torch.no_grad():
                    lg = model(**enc).logits
                logits = lg[torch.arange(len(chunk), device=lg.device), last].float()
                probs = torch.softmax(logits, -1)
                ym = probs[:, yes_ids].sum(-1); nm = probs[:, no_ids].sum(-1)
                zy = logits[:, yes_ids].max(-1).values
                zn = logits[:, no_ids].max(-1).values
                for i2, t2 in enumerate(chunk):
                    y, n2 = float(ym[i2]), float(nm[i2])
                    cov = y + n2
                    rec = {**t2, "arm": arm, "yes_mass": y, "no_mass": n2,
                           "z_yes": float(zy[i2]), "z_no": float(zn[i2]),
                           "logodds": float(zy[i2] - zn[i2]),
                           "p_yes": (y / cov) if cov > 0 else None, "coverage": cov,
                           "prefill": a.prefill,
                           "low_coverage": cov < COVERAGE_FLOOR}
                    f_bb.write(json.dumps(rec) + "\n")
                if b0 % (a.batch * 8) == 0:
                    print(f"[{arm}] BB {b0}/{len(bb_trials)}", flush=True); f_bb.flush()
            n_gen = int(len(bb_trials) * a.bb_gen_frac)
            step = max(1, len(bb_trials) // max(1, n_gen))
            samp = bb_trials[::step][:n_gen]
            for b0 in range(0, len(samp), a.open_batch):
                chunk = samp[b0:b0 + a.open_batch]
                texts = [chat(t2["sysprompt"], t2["prompt"], True) + a.prefill for t2 in chunk]
                enc = enc_gen(texts)
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=a.bb_max_new, do_sample=True,
                                       temperature=a.temperature, top_p=0.95,
                                       pad_token_id=tok.pad_token_id)
                for i2, t2 in enumerate(chunk):
                    txt = tok.decode(g[i2][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    f_bbg.write(json.dumps({**t2, "arm": arm, "generated": txt.strip()}) + "\n")
            f_bbg.flush()

            acts = []
            for b0 in range(0, len(bp_items), a.batch):
                chunk = bp_items[b0:b0 + a.batch]
                texts = [chat(it["sysprompt"], it["statement"], False) for it in chunk]
                enc, last = enc_fwd(texts)
                with torch.no_grad():
                    hs = model(**enc, output_hidden_states=True).hidden_states
                ar = torch.arange(len(chunk), device=hs[0].device)
                acts.append(torch.stack([h[ar, last] for h in hs], dim=1).float().cpu())
                if b0 % (a.batch * 8) == 0:
                    print(f"[{arm}] BP {b0}/{len(bp_items)}", flush=True)
            A = torch.cat(acts).numpy()
            if probe is not None:
                probes, valid = probe
                p_true = np.mean([probes[L][1].predict_proba(probes[L][0].transform(A[:, L, :]))[:, 1]
                                  for L in valid], axis=0)
                for it, pt in zip(bp_items, p_true):
                    f_bp.write(json.dumps({**it, "arm": arm, "p_true": float(pt)}) + "\n")
            if a.save_acts:
                torch.save({"activations": torch.from_numpy(A).half(), "meta": bp_items, "arm": arm},
                           out / f"acts_{arm}.pt")
            f_bp.flush()

            if f_cz is not None:
                import math as _m
                frames = W.get("cloze_frames", [])
                seen = set()
                for it in bp_items:
                    for fr in frames:
                        key = (it["entity"], it["sysprompt"], fr["id"])
                        if key in seen:
                            continue
                        seen.add(key)
                        e = it["entity"]
                        raw = fr["stem"].format(e=e)
                        if "?" in raw:
                            q, _, opener = raw.partition("?")
                            stem = chat(it["sysprompt"], q + "?", True) + opener
                        else:
                            stem = chat(it["sysprompt"], f"Tell me about {e}.", True) + raw
                        rt, rm, rn = seq_lp_full(stem, fr["real"] + ".")
                        ft, fm, fn = seq_lp_full(stem, fr["fic"] + ".")
                        d_mean, d_sum = rm - fm, rt - ft
                        f_cz.write(json.dumps({
                            "arm": arm, "entity": e, "group": it["group"], "etype": it["etype"],
                            "sysprompt": it["sysprompt"], "frame": fr["id"],
                            "p_real": 1.0 / (1.0 + _m.exp(-d_mean)),
                            "logdiff": d_mean,
                            "lp_real": rm, "lp_fic": fm,
                            "p_real_sum": 1.0 / (1.0 + _m.exp(-d_sum)),
                            "logdiff_sum": d_sum,
                            "lp_real_sum": rt, "lp_fic_sum": ft,
                            "n_tok_real": rn, "n_tok_fic": fn,
                            "anchor_mass": _m.exp(rt) + _m.exp(ft),
                            "p_real_def": "sigmoid(lp_real - lp_fic), k=1, length-normalised",
                            "schema": 2,
                        }) + "\n")
                f_cz.flush()

            for b0 in range(0, len(open_items), a.open_batch):
                chunk = open_items[b0:b0 + a.open_batch]
                texts = [chat(it["sysprompt"], it["prompt"], True) for it in chunk]
                enc = enc_gen(texts)
                with torch.no_grad():
                    g = model.generate(**enc, max_new_tokens=a.open_max_new, do_sample=True,
                                       temperature=a.temperature, top_p=0.95,
                                       pad_token_id=tok.pad_token_id)
                for i2, it in enumerate(chunk):
                    ans = tok.decode(g[i2][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    f_og.write(json.dumps({**it, "arm": arm, "answer": ans.strip()}) + "\n")
                if b0 % (a.open_batch * 8) == 0:
                    print(f"[{arm}] open {b0}/{len(open_items)}", flush=True); f_og.flush()
        print(f"=== arm {arm} done ===", flush=True)

    for f in (f_bb, f_bp, f_og, f_bbg):
        f.close()
    if f_cz is not None:
        f_cz.close()
    (out / "manifest.json").write_text(json.dumps({
        "base": a.base, "arms": [n for n, _ in arms], "temperature_generation": a.temperature,
        "temperature_logprob": 1.0, "coverage_floor": COVERAGE_FLOOR, "prefill": a.prefill,
        "registry": os.path.basename(a.registry), "wordings": os.path.basename(a.wordings),
        "n_entities": len(ents), "bb_per_arm": len(bb_trials), "bp_per_arm": len(bp_items),
        "open_per_arm": len(open_items), "probe": a.probe,
    }, indent=1))
    print("DONE", out, flush=True)


if __name__ == "__main__":
    main()
