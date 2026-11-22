"""Audit complete manifests and summarize network-level repeated simulations."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .run import write_json

NAMES=['beta0','beta1','beta2','rho','false_positive','false_negative']


def wilson(k,n):
    if n==0:return [None,None]
    z=1.95996398454;p=k/n;denom=1+z*z/n
    mid=(p+z*z/(2*n))/denom
    radius=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [float(mid-radius),float(mid+radius)]


def build(root):
    root=Path(root);manifest=json.loads((root/'manifest.json').read_text())
    datasets={};rows=[]
    for case,config in manifest['cases'].items():
        datasets[case]=[]
        for seed in range(manifest['seed_start'],manifest['seed_start']+manifest['replicates']):
            path=root/case/f'seed{seed}.json';d=json.loads(path.read_text())
            signature=hashlib.sha256(json.dumps(dict(config=config,seed=seed,code=manifest['source_hash']),sort_keys=True).encode()).hexdigest()
            if d['signature']!=signature or d['config']!=config or d['seed']!=seed:
                raise ValueError(f'Provenance mismatch {path}')
            datasets[case].append(d)
        methods=datasets[case][0]['results']
        for method in methods:
            rs=[d['results'][method] for d in datasets[case]]
            for j,name in enumerate(NAMES[:len(rs[0]['estimate'])]):
                estimates=np.array([r['estimate'][j] for r in rs]);truth=np.array([d['truth'][j] for d in datasets[case]])
                valid=np.array([r['success'] and r['lower'] is not None for r in rs])
                covered=np.array([r['lower'][j]<=v<=r['upper'][j] if r['lower'] is not None else False for r,v in zip(rs,truth)])
                widths=np.array([r['upper'][j]-r['lower'][j] if r['lower'] is not None else np.nan for r in rs])
                nvalid=int(valid.sum());k=int((covered&valid).sum());errors=estimates-truth
                rows.append(dict(case=case,method=method,parameter=name,replicates=len(rs),successful=nvalid,
                    failures=int((~valid).sum()),estimate_mean=float(estimates.mean()),bias=float(errors.mean()),
                    rmse=float(np.sqrt(np.mean(errors**2))),coverage=k/nvalid if nvalid else None,
                    coverage_wilson=wilson(k,nvalid),certified_coverage=k/len(rs),
                    mean_width=float(widths[valid].mean()) if nvalid else None,
                    median_seconds=float(np.median([r['seconds'] for r in rs])),
                    min_hessian_eigenvalue=float(min(min(r['hessian_eigenvalues']) for r in rs))))
    payload=dict(manifest=manifest,rows=rows,total_networks=sum(map(len,datasets.values())),
                 total_method_cells=sum(len(d['results']) for ds in datasets.values() for d in ds))
    write_json(root/'summary.json',payload)
    with (root/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    lines=['# Noisy reciprocal probit simulation', '',
           f"Networks: {payload['total_networks']}. Method cells: {payload['total_method_cells']}.", '',
           'Coverage is per parameter over independent networks, conditional on successful fits; failures are reported separately. RMSE/bias include all finite returned point estimates. Certified coverage also counts failed fits in its denominator. Intervals are transformed-coordinate Laplace intervals, not exact posterior intervals.', '',
           '| Case | Method | rho bias | rho RMSE | Coverage | Wilson 95% | Width | Failures | median fit s |',
           '|---|---|---:|---:|---:|---|---:|---:|---:|']
    for r in rows:
        if r['parameter']!='rho':continue
        lo,hi=r['coverage_wilson']
        cov='NA' if r['coverage'] is None else f"{100*r['coverage']:.1f}%"
        ci='NA' if lo is None else f'[{100*lo:.1f}, {100*hi:.1f}]'
        width='NA' if r['mean_width'] is None else f"{r['mean_width']:.3f}"
        lines.append(f"| {r['case']} | {r['method']} | {r['bias']:+.3f} | {r['rmse']:.3f} | {cov} | {ci} | {width} | {r['failures']}/{r['replicates']} | {r['median_seconds']:.3f} |")
    lines+=['', 'The oracle knows the channel rates but has no audited true edges. Joint and audit-plugin see the same audited dyads. Joint-conditional shares exactly the joint mode and fixes nuisance rates only for its covariance. Clean sees all true edges and is an information-rich reference. Unknown-error priors, numerical bounds and selection rules are documented in RESEARCH.md.']
    (root/'report.md').write_text('\n'.join(lines)+'\n')
    return payload


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/noisy_probit/minimal')
    build(p.parse_args().output)
