"""Post-run integrity and numerical audit; never changes estimators."""
import json,hashlib
from pathlib import Path
import numpy as np
from .analysis import load
from .model import make_posterior
from ..lowdim.simulation import simulate
from ..model import state_probabilities
from ..run import write_json


def main():
    data,rows,lock=load('results/noisy_probit/state_channel/validation')
    counts={};floor=0;minprob=1.;mins=[];maxgrads=[];bound=0;maxspread=0.;pipeline={}
    for (case,q),ds in data.items():
        times={k:[] for k in ds[0]['results']}
        for d in ds:
            dat=simulate(d['config'],d['seed'],q)
            assert hashlib.sha256(dat.x.tobytes()+dat.observed.tobytes()+dat.audit.tobytes()+dat.truth.tobytes()).hexdigest()==d['data_hash']
            for kind,r in d['results'].items():
                counts[kind]=counts.get(kind,0)+1
                post=make_posterior(dat,kind);z=np.array(r['mode'])
                c=post.measurement(z)[0];p,_=state_probabilities(dat.x,z[:3],np.tanh(z[3]))
                prob=(p*c[:,post.obs_index].T*post.allowed).sum(1)
                floor+=int((prob<1e-14).sum());minprob=min(minprob,float(prob.min()))
                assert np.allclose(c.sum(1),1,atol=2e-10) and c.min()>=0
                mins.append(min(r['hessian_eigenvalues']));maxgrads.append(r['gradient_max']);bound+=r['at_numerical_bound']
                maxspread=max(maxspread,max(s['value'] for s in r['starts'])-min(s['value'] for s in r['starts']))
                times[kind].append(r['seconds']+(d['results']['common']['seconds'] if kind!='common' else 0))
        pipeline[f'{case}_{q:g}']={k:float(np.median(v)) for k,v in times.items()}
    # All budgets use identical x, truth, observations and a nested audit sequence.
    for case in lock['design']['cases']:
        for j in range(200):
            old=None
            for q in [.01,.025,.05]:
                d=data[case,q][j];dat=simulate(d['config'],d['seed'],q)
                if old is not None:
                    assert np.array_equal(old.x,dat.x) and np.array_equal(old.observed,dat.observed)
                    assert np.array_equal(old.truth,dat.truth) and np.all(~old.audit|dat.audit)
                old=dat
    result=dict(fits=counts,total_fits=sum(counts.values()),floor_events=floor,min_fitted_event_probability=minprob,
                min_hessian_eigenvalue=min(mins),max_gradient=max(maxgrads),numerical_bound_fits=int(bound),
                max_start_objective_spread=maxspread,median_pipeline_seconds=pipeline,
                paired_data_and_nested_audits_verified=True)
    write_json(Path('results/noisy_probit/state_channel/validation/integrity.json'),result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
