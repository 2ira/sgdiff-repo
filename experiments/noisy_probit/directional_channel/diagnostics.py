"""Audit all posterior attempts and preserve extension-prefix evidence."""
import json
from pathlib import Path
import numpy as np
from ..run import write_json

ROOT=Path('results/noisy_probit/directional_channel/mcmc')


def main():
    attempts={};groups={}
    for p in sorted(ROOT.glob('*/result.json')):
        d=json.loads(p.read_text());name=p.parent.name;z=np.load(p.parent/'draws.npz')['unconstrained']
        b=np.r_[[np.inf]*3,4,[12]*2,[6]*4] if d['kind']=='six' else np.r_[[np.inf]*3,4,[d['numerical_domains']['rate_or_row_logits']]*12]
        assert np.all(abs(z)<b)
        attempts[name]=dict(draws=d['draws'],converged=d['converged'],max_rhat=max(d['diagnostics']['rank_split_rhat']),min_ess=min(d['diagnostics']['bulk_ess']),
            rho_interval=[d['lower'][3],d['upper'][3]],rho_mean=d['mean'][3],boundary_fraction=float(np.mean(np.any(abs(z)>b-1,axis=2))),
            max_abs_coordinate=np.max(abs(z),axis=(0,1)).tolist())
        group=name.rsplit('_',1)[0];groups.setdefault(group,[]).append((d['draws'],p.parent))
    final={}
    for group,ps in groups.items():
        ps.sort();final[group]=attempts[ps[-1][1].name]
        for (_,a),(_,b) in zip(ps,ps[1:]):
            x=np.load(a/'draws.npz')['unconstrained'];y=np.load(b/'draws.npz')['unconstrained'];assert np.array_equal(x,y[:,:x.shape[1]])
    out=dict(attempts=attempts,final=final,extensions_preserve_exact_prefixes=True)
    if 'domain_full' in final:
        out['domain_rho_endpoint_change']=float(np.max(abs(np.array(final['full']['rho_interval'])-final['domain_full']['rho_interval'])))
        out['domain_rho_mean_change']=final['domain_full']['rho_mean']-final['full']['rho_mean']
    write_json(ROOT/'summary.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
