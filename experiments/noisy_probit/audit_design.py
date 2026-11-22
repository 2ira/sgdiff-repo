"""Exploratory equal-budget observed-state stratification and no-audit control.

Selection uses observed dyads only (MAR given the observed likelihood). It is a
simple baseline, not a claim of optimal information-based selection.
"""
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import numpy as np
from .experiment import simulate, fit
from .model import Posterior
from .run import source_hash, write_json


def stratified_mask(observed, budget, seed):
    rng=np.random.default_rng(seed+981207)
    groups=[rng.permutation(np.flatnonzero(2*observed[:,0]+observed[:,1]==s)).tolist() for s in range(4)]
    selected=[]
    while len(selected)<budget:
        for group in groups:
            if group and len(selected)<budget:selected.append(group.pop())
    mask=np.zeros(len(observed),bool);mask[selected]=True
    return mask


def job(path):
    d=json.loads(Path(path).read_text())
    if d['source_hash']!=source_hash():raise ValueError('Fit source changed')
    data=simulate(**d['config'],seed=d['seed'])
    masks=dict(no_audit=np.zeros(len(data.x),bool),
               balanced_observed_states=stratified_mask(data.observed,int(data.audit.sum()),d['seed']))
    results={name:fit(Posterior(replace(data,audit=mask))) for name,mask in masks.items()}
    results['random']=d['results']['joint']
    out=Path('results/noisy_probit/audit_design')/d['case']/Path(path).name
    write_json(out,dict(seed=d['seed'],case=d['case'],truth=d['truth'],results=results,
                       input_signature=d['signature'],design_hash=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       audited_dyads={name:int(mask.sum()) for name,mask in masks.items()}))
    return all(r['success'] for r in results.values())


if __name__=='__main__':
    paths=[str(p) for c in ('moderate','strong') for p in sorted((Path('results/noisy_probit/minimal')/c).glob('seed*.json'))]
    with ProcessPoolExecutor(max_workers=4) as pool:
        for i,ok in enumerate(pool.map(job,paths),1):
            if i%20==0:print(i,len(paths),ok,flush=True)
