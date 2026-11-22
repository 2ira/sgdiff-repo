"""Frozen-primary-contrast analysis with paired network bootstrap uncertainty."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from .run import hashes,signature
from ..report import wilson
from ..run import write_json


def summarize(records,kind):
    rs=[d['results'][kind] for d in records];n=len(rs)
    truth=np.array([d['truth'][:4] for d in records]);est=np.array([r['estimate'][:4] for r in rs])
    if not np.isfinite(est).all():raise ValueError('Missing/nonfinite point estimates invalidate RMSE')
    success=np.array([r['success'] for r in rs]);rows=[]
    for j,param in enumerate(['beta0','beta1','beta2','rho']):
        err=est[:,j]-truth[:,j]
        lo=np.array([r['lower'][j] if r['lower'] else np.nan for r in rs])
        hi=np.array([r['upper'][j] if r['upper'] else np.nan for r in rs])
        covered=(lo<=truth[:,j])&(hi>=truth[:,j])&success
        rmse=float(np.sqrt(np.mean(err**2)));cov=float(covered.mean());k=int(covered.sum())
        rows.append(dict(method=kind,parameter=param,n=n,failures=int((~success).sum()),bias=float(err.mean()),
                         bias_mcse=float(err.std(ddof=1)/np.sqrt(n)),rmse=rmse,
                         rmse_mcse=float((err**2).std(ddof=1)/(2*np.sqrt(n)*rmse)) if rmse>0 else 0.,
                         coverage=cov,coverage_conditional=k/int(success.sum()) if success.any() else None,
                         coverage_mcse=float(np.sqrt(cov*(1-cov)/n)),coverage_wilson=wilson(k,n),
                         width=float(np.mean((hi-lo)[success])) if success.any() else None,
                         seconds=float(np.median([r['seconds'] for r in rs]))))
    return rows


def load(root):
    root=Path(root);lock=json.loads((root/'LOCK.json').read_text());design=lock['design']
    assert design['source_hashes']==hashes();assert lock['lock_hash']==signature(design)
    data={};rows=[]
    for case,config in design['cases'].items():
        for q in design['budgets']:
            ds=[]
            for seed in range(config['seed_start'],config['seed_start']+design['replicates']):
                d=json.loads((root/case/f'q{q:g}'/f'seed{seed}.json').read_text())
                assert d['signature']==signature(dict(case=case,config=config,seed=seed,q=q,code=hashes()))
                assert d['source_hashes']==hashes();assert d['seed']==seed
                ds.append(d)
            data[(case,q)]=ds
            for method in ds[0]['results']:
                for row in summarize(ds,method):
                    rows.append(dict(stage=design['stage'],case=case,q=q,audited_dyads=ds[0]['audited_dyads'],**row))
    return data,rows,lock


def compare(a,method_a,b,method_b,seed=132017):
    assert [d['seed'] for d in a]==[d['seed'] for d in b]
    ea=np.array([(d['results'][method_a]['estimate'][3]-d['truth'][3])**2 for d in a])
    eb=np.array([(d['results'][method_b]['estimate'][3]-d['truth'][3])**2 for d in b])
    rng=np.random.default_rng(seed);idx=rng.integers(0,len(a),size=(10000,len(a)))
    ratio=np.sqrt(ea[idx].mean(1)/eb[idx].mean(1))
    diff=np.sqrt(ea[idx].mean(1))-np.sqrt(eb[idx].mean(1))
    r=summarize(a,method_a)[3]
    upper=float(np.quantile(ratio,.975));ci=np.quantile(ratio,[.025,.975]).tolist()
    gates=dict(rmse=upper<=1.10,coverage=r['coverage_wilson'][0]>=.90,failures=r['failures']/r['n']<=.01)
    return dict(method_a=method_a,q_a=a[0]['audit_fraction'],method_b=method_b,q_b=b[0]['audit_fraction'],
                rmse_ratio=float(np.sqrt(ea.mean()/eb.mean())),rmse_ratio_ci95=ci,upper_one_sided_975=upper,
                rmse_difference=float(np.sqrt(ea.mean())-np.sqrt(eb.mean())),difference_ci95=np.quantile(diff,[.025,.975]).tolist(),
                a_coverage=r['coverage'],a_coverage_wilson=r['coverage_wilson'],gates=gates,all_gates=all(gates.values()))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='results/noisy_probit/lowdim/validation');a=p.parse_args()
    data,rows,lock=load(a.root);contrasts={};secondary={};aux=[]
    if lock['design']['stage']=='validation':
        ref=data[('positive',.05)]
        contrasts['same_budget']=compare(ref,'copula',ref,'full')
        contrasts['half_budget']=compare(data[('positive',.025)],'copula',ref,'full')
        for q in [.01,.025,.05]:
            secondary[f'positive_q{q:g}_vs_full_5']=compare(data[('positive',q)],'copula',ref,'full')
        for q in [.025,.05]:
            secondary[f'full_prior_q{q:g}']=compare(data[('positive',q)],'copula',ref,'full_shrink')
            secondary[f'kappa_prior_q{q:g}']=compare(data[('positive',q)],'copula_shrink',ref,'full')
    for (case,q),ds in data.items():
        rs=[d['results']['copula'] for d in ds]
        corr=[r['covariance'][3][6]/np.sqrt(r['covariance'][3][3]*r['covariance'][6][6]) for r in rs if r['covariance']]
        aux.append(dict(case=case,q=q,kappa_mean=float(np.mean([r['estimate'][6] for r in rs])),
                        rho_kappa_correlation_median=float(np.median(corr)),
                        rho_kappa_abs_correlation_p90=float(np.quantile(np.abs(corr),.9))))
    result=dict(lock=lock,rows=rows,primary_contrasts=contrasts,secondary_contrasts=secondary,auxiliary=aux,
                analysis_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    root=Path(a.root);write_json(root/'summary.json',result)
    with (root/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(contrasts,indent=2))


if __name__=='__main__':main()
