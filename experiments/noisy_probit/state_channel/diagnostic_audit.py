"""Check diagnostic prefixes and quantify the optional domain sensitivity."""
import json
from pathlib import Path
import numpy as np
from ..run import write_json


def main():
    root=Path('results/noisy_probit/state_channel/mcmc');out={}
    for name in ['state_12000','full_12000','full_48000','domain_full_48000','domain_full_96000']:
        p=root/name;d=json.loads((p/'result.json').read_text());z=np.load(p/'draws.npz')['unconstrained']
        b=np.r_[[np.inf]*3,4,[12]*2,[6]*3] if d['kind']=='state' else np.r_[[np.inf]*3,4,[d['numerical_domains']['rate_or_row_logits']]*12]
        assert np.all(abs(z)<b)
        out[name]=dict(converged=d['converged'],draws=d['draws'],rhat_max=max(d['diagnostics']['rank_split_rhat']),
                       ess_min=min(d['diagnostics']['bulk_ess']),rho_mean=d['mean'][3],rho_interval=[d['lower'][3],d['upper'][3]],
                       within_one_logit_of_bound_fraction=float(np.mean(np.any(abs(z)>b-1,axis=2))),
                       max_abs_coordinate=np.max(abs(z),axis=(0,1)).tolist())
    for a,b in [('full_12000','full_48000'),('domain_full_48000','domain_full_96000')]:
        za=np.load(root/a/'draws.npz')['unconstrained'];zb=np.load(root/b/'draws.npz')['unconstrained']
        assert np.array_equal(za,zb[:,:za.shape[1]])
    out['extensions_preserve_exact_prefixes']=True
    out['domain_rho_max_endpoint_difference']=float(max(abs(np.array(out['domain_full_96000']['rho_interval'])-out['full_48000']['rho_interval'])))
    out['domain_rho_mean_difference']=out['domain_full_96000']['rho_mean']-out['full_48000']['rho_mean']
    write_json(root/'summary.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
