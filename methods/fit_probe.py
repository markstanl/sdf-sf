#!/usr/bin/env python
from __future__ import annotations

import argparse
import pathlib
import pickle

CACHE = pathlib.Path("_probe_cache")


def fit_probe(marks_path):
    import numpy as np
    import torch
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import roc_auc_score

    CACHE.mkdir(parents=True, exist_ok=True)
    ck = CACHE / (marks_path.stem + "__" + marks_path.parent.name + ".pkl")
    if ck.exists():
        return pickle.loads(ck.read_bytes())
    print(f"  fitting probe on {marks_path.name} ...", flush=True)
    bm = torch.load(marks_path, weights_only=False)
    X = bm["activations"].numpy()
    y = bm["labels"].numpy().astype(int)
    nL = X.shape[1]
    fns, valid = {}, []
    for L in range(nL):
        sc = StandardScaler().fit(X[:, L, :])
        Z = sc.transform(X[:, L, :])
        clf = LogisticRegression(C=0.01, max_iter=2000).fit(Z, y)
        if roc_auc_score(y, clf.decision_function(Z)) >= 0.95 and L >= round(0.4 * nL):
            valid.append(L)
        fns[L] = (sc, clf)
    if not valid:
        valid = [nL // 2]
    del X, bm
    obj = (fns, valid)
    ck.write_bytes(pickle.dumps(obj))
    print(f"    kept {len(valid)}/{nL} layers", flush=True)
    return obj


def main():
    global CACHE
    ap = argparse.ArgumentParser(
        description="Fit the frozen per-layer linear truth probe (StandardScaler + "
                    "LogisticRegression(C=0.01)) on labelled statement activations.")
    ap.add_argument("--acts", required=True,
                    help="torch file with 'activations' [N, layers, d] and binary 'labels' [N]")
    ap.add_argument("--out", required=True, help="output probe pickle: ({layer: (scaler, clf)}, valid_layers)")
    ap.add_argument("--cache-dir", default=None,
                    help="probe cache directory (default: <out dir>/_probe_cache)")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    CACHE = pathlib.Path(a.cache_dir) if a.cache_dir else out.parent / "_probe_cache"
    obj = fit_probe(pathlib.Path(a.acts))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps(obj))
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
