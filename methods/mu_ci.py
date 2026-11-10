from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import glob
import json
import os
from pathlib import Path
import time

for _name in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.special import log_ndtr, ndtr

N = 500
DENOM = N*(N-1)/2
SQRT2 = np.sqrt(2.)
LOG2PI = np.log(2*np.pi)


def pair_util(inspect_dir):
    files = [f for f in glob.glob(inspect_dir + "/*.json") if "generate_config" not in f]
    if not files:
        return {}
    d = json.load(open(sorted(files)[-1]))
    ordered = {}
    for s in d.get("samples", []):
        md = s.get("metadata") or {}
        if md.get("i") is None or md.get("j") is None:
            continue
        p = ((s.get("scores") or {}).get("decisiveness_scorer") or {}).get("value")
        if p is None or (isinstance(p, float) and p != p):
            continue
        ordered[(md["i"], md["j"])] = float(p)
    out = {}
    for (i, j), pij in ordered.items():
        if i >= j:
            continue
        pji = ordered.get((j, i))
        out[(i, j)] = pij if pji is None else 0.5 * (pij + (1.0 - pji))
    return out


def decisiveness(mu, counts):
    ii, jj = np.triu_indices(len(mu), 1)
    strength = np.abs(2*ndtr((mu[ii]-mu[jj])/SQRT2)-1)
    return float(np.dot(counts[ii]*counts[jj], strength)/DENOM)


def fit_graph(source, counts, warm_mu):
    active = np.flatnonzero(counts)
    mapping = np.full(N, -1, dtype=np.int32)
    mapping[active] = np.arange(len(active))
    ii, jj = source['i'], source['j']
    weights = counts[ii]*counts[jj]
    keep = weights > 0
    ii, jj, weights, yy = mapping[ii[keep]], mapping[jj[keep]], weights[keep], source['y'][keep]
    graph = coo_matrix((np.ones(len(ii)), (ii,jj)), shape=(len(active),len(active))).tocsr()
    components = connected_components(graph, directed=False, return_labels=False)
    if components != 1:
        raise RuntimeError(f"Disconnected induced graph: {source['family']} {source['arm']}: {components}")

    def objective(mu):
        z = (mu[ii]-mu[jj])/SQRT2
        logp, logq = log_ndtr(z), log_ndtr(-z)
        logpdf = -.5*z*z-.5*LOG2PI
        loss = -np.dot(weights, yy*logp+(1-yy)*logq)
        d = weights*((1-yy)*np.exp(logpdf-logq)-yy*np.exp(logpdf-logp))/SQRT2
        grad = np.bincount(ii, weights=d, minlength=len(active))-np.bincount(jj, weights=d, minlength=len(active))
        return loss, grad

    initial = warm_mu[active].copy()
    initial -= initial.mean()
    fit = minimize(objective, initial, jac=True, method='L-BFGS-B',
                   options=dict(maxiter=600, ftol=1e-13, gtol=1e-7, maxcor=20))
    loss, grad = objective(fit.x)
    retried = False
    if not fit.success or np.max(np.abs(grad)) > 5e-4:
        retried = True
        fit = minimize(objective, fit.x, jac=True, method='L-BFGS-B',
                       options=dict(maxiter=1200, ftol=1e-15, gtol=1e-8, maxcor=30))
        loss, grad = objective(fit.x)
    gradient_max = float(np.max(np.abs(grad)))
    if not np.isfinite(loss) or gradient_max > 5e-4:
        raise RuntimeError(f"Nonconverged fit {source['family']} {source['arm']}: {fit.message}; maxgrad={gradient_max}")
    centered = fit.x-fit.x.mean()
    full = np.full(N, np.nan)
    full[active] = centered
    point = decisiveness(centered, counts[active])
    assert np.isfinite(point) and 0<=point<=1
    return dict(mu=full, point=point, optimizer_success=bool(fit.success),
                iterations=int(fit.nit), gradient_max=gradient_max,
                loss=float(loss), components=int(components), retried=retried,
                active_items=int(len(active)))


def worker(source, counts, start, end):
    values, gradients, iterations, retries, successes = [], [], [], [], []
    for count in counts:
        f = fit_graph(source, count, source['fit_mu'])
        values.append(f['point']); gradients.append(f['gradient_max'])
        iterations.append(f['iterations']); retries.append(f['retried']); successes.append(f['optimizer_success'])
    return (source['family'], source['arm'], start, end, np.array(values),
            dict(max_gradient=max(gradients), max_iterations=max(iterations),
                 retries=sum(retries), optimizer_successes=sum(successes), draws=end-start))


def run_jobs(sources, counts, workers, batch=25):
    outputs = {(s['family'],s['arm']):np.full(len(counts),np.nan) for s in sources if s['arm']!='bare'}
    diagnostics = []
    t0 = time.monotonic()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(worker,s,counts[start:min(start+batch,len(counts))],start,min(start+batch,len(counts)))
                for s in sources if s['arm']!='bare' for start in range(0,len(counts),batch)]
        completed = 0
        for future in as_completed(jobs):
            family,arm,start,end,vals,diag = future.result()
            outputs[family,arm][start:end] = vals
            diagnostics.append(dict(family=family,arm=arm,start=start,end=end,**diag))
            completed += end-start
            if completed % (batch*20) == 0 or completed == len(outputs)*len(counts):
                print(f'Completed {completed}/{len(outputs)*len(counts)} fits in {time.monotonic()-t0:.1f}s',flush=True)
    return outputs,diagnostics,time.monotonic()-t0


def cmd_pairs(args):
    G = pair_util(args.graft)
    Nn = pair_util(args.native)
    keys = sorted(set(G) & set(Nn))
    if not keys:
        raise SystemExit("no shared unordered pairs between the two arms")
    dg = np.array([abs(2 * G[k] - 1) for k in keys])
    dn = np.array([abs(2 * Nn[k] - 1) for k in keys])
    d = dg - dn
    rng = np.random.default_rng(0)
    bs = d[rng.integers(0, d.size, size=(2000, d.size))].mean(axis=1)
    lo, hi = np.percentile(bs, [2.5, 97.5])
    res = {"metric": "mu_decisiveness", "graft_decisiveness_raw": float(dg.mean()),
           "native_decisiveness_raw": float(dn.mean()), "v": float(d.mean()),
           "lo": float(lo), "hi": float(hi), "n": int(d.size),
           "sig": bool(lo > 0 or hi < 0), "paired": True}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + '\n')
    print(json.dumps(res), flush=True)


def load_arm(label, arm, folder):
    folder = Path(folder)
    mapping = json.loads((folder/'mu.json').read_text())
    items = list(mapping)
    assert len(items) == len(set(items)) == N
    mu = np.array([mapping[item] for item in items])
    edges = [json.loads(line) for line in (folder/'edges.jsonl').read_text().splitlines()]
    edges = [r for r in edges if r['phase'] == 'elo']
    ii = np.array([r['i'] for r in edges], dtype=np.int32)
    jj = np.array([r['j'] for r in edges], dtype=np.int32)
    yy = np.array([r['p_util'] for r in edges])
    assert np.isfinite(yy).all() and ((yy>=0)&(yy<=1)).all()
    return dict(family=label, arm=arm, i=ii, j=jj, y=yy, saved_mu=mu,
                saved_point=decisiveness(mu, np.ones(N)), items=items)


def cmd_items(args):
    global N, DENOM
    N = args.n_items
    DENOM = N*(N-1)/2
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    sources = []
    for label, gdir, ndir in args.contrast:
        sources.append(load_arm(label, 'graft', gdir))
        sources.append(load_arm(label, 'native', ndir))
    for s in sources[1:]:
        assert s['items'] == sources[0]['items']
    audits = []
    for source in sources:
        fitted = fit_graph(source, np.ones(N, dtype=int), source['saved_mu'])
        source['fit_mu'] = fitted.pop('mu')
        source['fit_point'] = fitted['point']
        audits.append(dict(family=source['family'], arm=source['arm'], saved_point=source['saved_point'],
                           difference_pp=100*(fitted['point']-source['saved_point']), **fitted))
    pd.DataFrame(audits).to_csv(args.out/'point-convergence.csv', index=False)
    maxdiff = max(abs(x['difference_pp']) for x in audits)
    counts = np.random.default_rng(args.seed).multinomial(N, np.full(N, 1/N), size=args.replicates)
    draws, diags, elapsed = run_jobs(sources, counts, args.workers)
    bykey = {(s['family'], s['arm']): s for s in sources}
    summaries = []
    for label, _, _ in args.contrast:
        g, n = (bykey[label, r] for r in ['graft', 'native'])
        delta = g['fit_point']-n['fit_point']
        boot = draws[label, g['arm']]-draws[label, n['arm']]
        lower, upper = np.quantile(boot, [.025, .975])
        tail = int(np.count_nonzero(np.abs(boot-delta) >= abs(delta)))
        p = (tail+1)/(args.replicates+1)
        summaries.append(dict(contrast=label,
            graft=100*g['fit_point'], native=100*n['fit_point'], gap=100*delta,
            saved_gap=100*(g['saved_point']-n['saved_point']),
            ci_lo=100*lower, ci_hi=100*upper, significant_nominal_ci=bool(lower>0 or upper<0),
            bootstrap_median=100*np.median(boot), bootstrap_bias=100*(boot.mean()-delta),
            centered_bootstrap_p_approx=p, tail_count=tail, replicates=args.replicates))
    order = np.argsort([s['centered_bootstrap_p_approx'] for s in summaries])
    running = 0.
    for rank, index in enumerate(order):
        running = max(running, min(1., (len(summaries)-rank)*summaries[index]['centered_bootstrap_p_approx']))
        summaries[index]['holm_p_approx'] = running
        summaries[index]['significant_holm_approx'] = bool(running<.05)
    pd.DataFrame(summaries).to_csv(args.out/'contrasts.csv', index=False)
    pd.DataFrame(diags).to_csv(args.out/'fit-diagnostics.csv', index=False)
    (args.out/'protocol.json').write_text(json.dumps(dict(n_items=N, replicates=args.replicates, seed=args.seed,
        max_saved_point_difference_pp=maxdiff, elapsed_bootstrap_seconds=elapsed,
        elapsed_total_seconds=time.monotonic()-started), indent=2)+'\n')
    print(pd.DataFrame(summaries)[['contrast','gap','ci_lo','ci_hi','centered_bootstrap_p_approx','holm_p_approx']].to_string(index=False), flush=True)


def main():
    ap = argparse.ArgumentParser(description="Paired graft-vs-native confidence intervals for mu-decisiveness.")
    sub = ap.add_subparsers(dest='cmd', required=True)
    pp = sub.add_parser('pairs', help="position-bias-free pair fold and bootstrap over unordered pairs")
    pp.add_argument('--graft', required=True, help="graft arm inspect log directory for the decisiveness task")
    pp.add_argument('--native', required=True, help="native arm inspect log directory for the decisiveness task")
    pp.add_argument('--out', type=Path, required=True, help="output JSON")
    pp.set_defaults(func=cmd_pairs)
    pi = sub.add_parser('items', help="Case V refit bootstrap over items")
    pi.add_argument('--contrast', nargs=3, action='append', required=True,
                    metavar=('NAME', 'GRAFT_DIR', 'NATIVE_DIR'),
                    help="arm directories each holding mu.json and edges.jsonl (repeatable)")
    pi.add_argument('--out', type=Path, required=True, help="output directory")
    pi.add_argument('--n-items', type=int, default=500)
    pi.add_argument('--replicates', type=int, default=1000)
    pi.add_argument('--seed', type=int, default=20260915)
    pi.add_argument('--workers', type=int, default=4)
    pi.set_defaults(func=cmd_items)
    args = ap.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
