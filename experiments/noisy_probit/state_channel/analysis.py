"""Prospective contrasts, with all validation cells checked for completeness."""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np
from .run import hashes,signature
from ..lowdim.analysis import summarize,compare
from ..run import write_json


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
                assert d['source_hashes']==hashes() and d['seed']==seed and d['config']==config
                ds.append(d)
            data[(case,q)]=ds
            for method in ds[0]['results']:
                for row in summarize(ds,method):
                    rows.append(dict(stage=design['stage'],case=case,q=q,audited_dyads=ds[0]['audited_dyads'],**row))
    return data,rows,lock


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['development','validation'],default='validation');a=p.parse_args()
    root=Path('results/noisy_probit/state_channel')/a.stage
    data,rows,lock=load(root);contrasts={};secondary={};aux=[]
    if a.stage=='validation':
        ref=data['state',.05]
        contrasts['same_budget']=compare(ref,'state',ref,'full')
        contrasts['half_budget']=compare(data['state',.025],'state',ref,'full')
        for case in lock['design']['cases']:
            for q in [.01,.025,.05]:
                secondary[f'{case}_{q:g}']=compare(data[case,q],'state',data[case,.05],'full')
        for q in [.025,.05]:
            secondary[f'full_prior_{q:g}']=compare(data['state',q],'state',ref,'full_shrink')
            secondary[f'state_prior_{q:g}']=compare(data['state',q],'state_shrink',ref,'full')
    for (case,q),ds in data.items():
        for method in ['common','state']:
            estimates=np.array([d['results'][method]['estimate'][4:] for d in ds])
            aux.append(dict(case=case,q=q,method=method,mean=estimates.mean(0).tolist(),sd=estimates.std(0,ddof=1).tolist()))
    result=dict(lock=lock,rows=rows,primary_contrasts=contrasts,secondary_contrasts=secondary,auxiliary=aux,
                analysis_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(root/'summary.json',result)
    with (root/'summary.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    print(json.dumps(contrasts,indent=2))


if __name__=='__main__':main()
