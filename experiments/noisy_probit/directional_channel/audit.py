"""Verify generated hashes, paired audits, modes, and numerical diagnostics."""
import hashlib,json
import numpy as np
from .analysis import load,ROOT
from .model import make_posterior
from .density import negative_log_density
from ..lowdim.simulation import simulate
from ..model import state_probabilities
from ..run import write_json


def main():
    data,lock=load();n=0;floor=0;minp=1.;mineig=np.inf;maxg=0.;bound=0;fail=0;spread=0.
    for (case,q),ds in data.items():
        for d in ds:
            dat=simulate(d['config'],d['seed'],q)
            dh=hashlib.sha256(dat.x.tobytes()+dat.observed.tobytes()+dat.audit.tobytes()+dat.truth.tobytes()).hexdigest();assert dh==d['data_hash']
            for kind,r in d['results'].items():
                n+=1;post=make_posterior(dat,kind);z=np.array(r['mode']);c=post.measurement(z)[0]
                assert np.allclose(c.sum(1),1,atol=2e-10) and c.min()>=0
                p,_=state_probabilities(dat.x,z[:3],np.tanh(z[3]));prob=(p*c[:,post.obs_index].T*post.allowed).sum(1)
                minp=min(minp,float(prob.min()));floor+=int((prob<1e-14).sum())
                mineig=min(mineig,min(r['hessian_eigenvalues']));maxg=max(maxg,r['gradient_max']);bound+=r['at_numerical_bound'];fail+=not r['success']
                spread=max(spread,max(s['value'] for s in r['starts'])-min(s['value'] for s in r['starts']))
                if kind in ['six','full']:assert abs(negative_log_density(post,z)-r['objective'])<1e-9
    for case in lock['design']['cases']:
        for a,b in zip(data[case,.025],data[case,.05]):
            assert a['seed']==b['seed'];x=simulate(a['config'],a['seed'],.025);y=simulate(b['config'],b['seed'],.05)
            assert np.array_equal(x.observed,y.observed) and np.array_equal(x.x,y.x) and np.array_equal(x.truth,y.truth) and np.all(~x.audit|y.audit)
    out=dict(fits=n,failures=int(fail),probability_floor_events=floor,min_fitted_probability=minp,
             min_hessian_eigenvalue=mineig,max_gradient=maxg,bound_contacts=int(bound),max_start_objective_spread=spread,
             source_and_data_hashes_verified=True,paired_nested_audits_verified=True,gradient_free_density_verified_at_modes=True)
    write_json(ROOT/'integrity.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
