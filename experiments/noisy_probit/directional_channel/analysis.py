"""All outcomes and paired uncertainty, without post-validation selection."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
from .run import hashes,signature
from ..lowdim.analysis import summarize,compare
from ..lowdim.model import copula_channel
from ..run import write_json

ROOT=Path('results/noisy_probit/directional_channel/validation')


def load():
    lock=json.loads((ROOT/'LOCK.json').read_text());design=lock['design'];code=hashes()
    assert design['source_hashes']==code and lock['lock_hash']==signature(design)
    data={}
    for case,cfg in design['cases'].items():
        for q in design['budgets']:
            ds=[]
            for seed in range(cfg['seed_start'],cfg['seed_start']+design['replicates']):
                d=json.loads((ROOT/case/f'q{q:g}'/f'seed{seed}.json').read_text())
                assert d['signature']==signature(dict(case=case,config=cfg,seed=seed,q=q,code=code)) and d['source_hashes']==code
                assert d['seed']==seed and d['config']==cfg;ds.append(d)
            data[case,q]=ds
    return data,lock


def main():
    data,lock=load();rows=[];channel=[];aux=[];secondary={}
    for (case,q),ds in data.items():
        cfg=ds[0]['config'];true_c=copula_channel(cfg['fp'],cfg['fn'],np.array(cfg['kappa'])[:,None])[0]
        for kind in ds[0]['results']:
            for r in summarize(ds,kind):rows.append(dict(case=case,q=q,audited_dyads=ds[0]['audited_dyads'],**r))
            errors=np.array([np.mean((np.array(d['results'][kind]['channel'])-true_c)**2) for d in ds]);rmse=np.sqrt(errors.mean())
            times=[d['results'][kind]['seconds']+(d['results']['common']['seconds'] if kind!='common' else 0) for d in ds]
            channel.append(dict(case=case,q=q,method=kind,rmse=float(rmse),mcse=float(errors.std(ddof=1)/(2*np.sqrt(len(ds))*rmse)),
                                median_pipeline_seconds=float(np.median(times))))
        est=np.array([d['results']['six']['estimate'][4:] for d in ds]);aux.append(dict(case=case,q=q,mean=est.mean(0).tolist(),sd=est.std(0,ddof=1).tolist()))
        secondary[f'{case}_{q:g}_six_vs_five']=compare(ds,'six',ds,'state')
        secondary[f'{case}_{q:g}_six_vs_full200']=compare(ds,'six',data[case,.05],'full')
        secondary[f'{case}_{q:g}_five_vs_full200']=compare(ds,'state',data[case,.05],'full')
    ref=data['asymmetric',.05]
    primary=dict(same_budget=compare(ref,'six',ref,'full'),half_budget=compare(data['asymmetric',.025],'six',ref,'full'))
    for q in [.025,.05]:
        secondary[f'six_prior_{q:g}']=compare(data['asymmetric',q],'six_shrink',ref,'full')
        secondary[f'full_prior_{q:g}']=compare(data['asymmetric',q],'six',ref,'full_shrink')
    out=dict(lock=lock,rows=rows,channel=channel,nuisance=aux,primary_contrasts=primary,secondary_contrasts=secondary,
             analysis_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    write_json(ROOT/'summary.json',out)
    for name,rs in [('summary',rows),('channel',channel)]:
        with (ROOT/(name+'.csv')).open('w') as f:
            w=csv.DictWriter(f,fieldnames=list(rs[0]));w.writeheader();w.writerows(rs)
    print(json.dumps(primary,indent=2))


if __name__=='__main__':main()
