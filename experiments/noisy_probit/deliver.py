"""Generate slide tables/plots directly from verified simulation outputs."""
import hashlib
import json
from pathlib import Path
import numpy as np
from .report import build, wilson
from .run import write_json, source_hash

ROOT=Path('results/noisy_probit')
OUT=Path('deliverables/noisy_probit_slides')
LABELS={'naive':'Naive','oracle':'Known rates','audit_plugin':'Plug-in','joint':'Joint','joint_conditional':'Conditional','clean':'Clean reference'}

def summary(rs, truth=.8, integrated=False):
    valid=np.array([r['success'] for r in rs])
    if integrated:
        point=np.array([r.get('mean',np.nan) for r in rs]);low=np.array([r.get('lower',np.nan) for r in rs]);hi=np.array([r.get('upper',np.nan) for r in rs])
    else:
        point=np.array([r['estimate'][3] for r in rs]);low=np.array([r['lower'][3] if r['lower'] else np.nan for r in rs]);hi=np.array([r['upper'][3] if r['upper'] else np.nan for r in rs])
    k=int(((low<=truth)&(hi>=truth)&valid).sum());n=int(valid.sum())
    return dict(replicates=len(rs),successful=n,failures=len(rs)-n,coverage=k/n if n else None,
                certified_coverage=k/len(rs),coverage_wilson=wilson(k,n),
                rmse=float(np.sqrt(np.nanmean((point-truth)**2))),bias=float(np.nanmean(point-truth)),
                mean_width=float(np.mean((hi-low)[valid])) if n else None,
                median_seconds=float(np.median([r['seconds'] for r in rs])))

def main():
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'data').mkdir(exist_ok=True)
    rows=[]
    for name in ['minimal','extension','scaling']:
        payload=build(ROOT/name)
        assert payload['manifest']['source_hash']==source_hash()
        rows+=payload['rows']
    def row(case,method='joint',param='rho'):
        return next(r for r in rows if r['case']==case and r['method']==method and r['parameter']==param)
    audit={};integrated={};paired={}
    profilehash=hashlib.sha256(Path(__file__).with_name('profile.py').read_bytes()).hexdigest()
    designhash=hashlib.sha256(Path(__file__).with_name('audit_design.py').read_bytes()).hexdigest()
    for case in ['moderate','strong']:
        ds=[json.loads((ROOT/'audit_design'/case/f'seed{s}.json').read_text()) for s in range(202101,202201)]
        assert all(d['design_hash']==designhash for d in ds)
        audit[case]={m:summary([d['results'][m] for d in ds]) for m in ds[0]['results']}
        ps=[json.loads((ROOT/'rho_integrated_final'/case/f'seed{s}.json').read_text()) for s in range(202101,202201)]
        assert all(d['profile_hash']==profilehash and d['points']==161 for d in ps)
        for d,p in zip(ds,ps):assert d['input_signature']==p['input_signature']
        integrated[case]=summary(ps,integrated=True)
        integrated[case]['tail_trimmed']=sum(p.get('tail_trimmed',False) for p in ps)
        # Paired bootstrap of RMSE contrast: descriptive uncertainty, fixed seeds.
        rng=np.random.default_rng(919)
        index=rng.integers(0,len(ds),size=(4000,len(ds)))
        a=np.array([(d['results']['balanced_observed_states']['estimate'][3]-.8)**2 for d in ds])
        b=np.array([(d['results']['random']['estimate'][3]-.8)**2 for d in ds])
        contrast=np.sqrt(a[index].mean(1))-np.sqrt(b[index].mean(1))
        paired[case]=dict(stratified_minus_random_rmse=float(np.sqrt(a.mean())-np.sqrt(b.mean())),bootstrap95=np.quantile(contrast,[.025,.975]).tolist())
    full={};channelhash=hashlib.sha256(Path(__file__).with_name('full_channel.py').read_bytes()).hexdigest()
    for case in ['moderate','correlated_errors','heterogeneous_errors']:
        ds=[json.loads((ROOT/'full_channel'/case/f'seed{s}.json').read_text()) for s in range(202101,202201)]
        assert all(d['channel_hash']==channelhash for d in ds)
        full[case]=summary([d['result'] for d in ds])
    write_json(ROOT/'synthesis.json',dict(rows=rows,audit=audit,integrated=integrated,audit_paired=paired,full_channel=full))
    def table(name, header, values):
        (OUT/'data'/name).write_text(header+'\n'+'\n'.join(' '.join(map(str,r)) for r in values)+'\n')
    methods=['naive','oracle','audit_plugin','joint']
    table('rmse.dat','x moderate strong',[[i,row('moderate',m)['rmse'],row('strong',m)['rmse']] for i,m in enumerate(methods)])
    table('coverage.dat','x moderate strong',[[i,100*row('moderate',m)['coverage'],100*row('strong',m)['coverage']] for i,m in enumerate(methods)])
    def tablerows(name,values):
        (OUT/'data'/name).write_text('\n'.join(' & '.join(map(str,r))+r' \\' for r in values)+'\n')
    def fmt(r):return [f"{r['rmse']:.3f}",f"{100*r['coverage']:.1f}\\%",f"{r['mean_width']:.3f}",f"{r['failures']}/{r['replicates']}"]
    tablerows('scaling.tex',[[n,reps,n*(n-1)//2]+fmt(row(c))+[f"{row(c)['median_seconds']:.2f}"] for n,reps,c in [(30,100,'small_n30'),(60,100,'moderate'),(120,100,'larger_n120'),(300,30,'larger_n300')]])
    tablerows('audit.tex',[[label]+fmt(r)[:3] for label,r in [('0\\%',audit['moderate']['no_audit']),('1\\%',row('audit_one_percent')),('5\\%',row('moderate')),('10\\%',row('audit_ten_percent'))]])
    tablerows('design.tex',[[label]+fmt(audit[c][m])[:2]+[f"{paired[c]['stratified_minus_random_rmse']:+.3f}" if m=='balanced_observed_states' else '--'] for c,label0 in [('moderate','中等'),('strong','较强')] for m,label in [('random',label0+'：随机'),('balanced_observed_states',label0+'：均衡分层')]])
    tablerows('integration.tex',[[label]+fmt(r)+[f"{r['median_seconds']:.2f}"] for c,label0 in [('moderate','中等'),('strong','较强')] for label,r in [(label0+'：Laplace',row(c)),(label0+'：一维积分',integrated[c])]])
    tablerows('stress.tex',[[label]+fmt(row(c))[:3] for c,label in [('sparse','相对稀疏'),('weak_reciprocity',r'$\rho=0.20$'),('strong_reciprocity',r'$\rho=0.95$')]])
    tablerows('misspec.tex',[[label,LABELS[m]]+fmt(row(c,m))[:2] for c,label in [('heterogeneous_errors','协变量相关错误'),('correlated_errors','方向相关错误')] for m in ['naive','audit_plugin','joint']])
    tablerows('beta.tex',[[c,LABELS[m]]+[f"{100*row(c,m,p)['coverage']:.0f}\\%" for p in ['beta0','beta1','beta2']] for c in ['moderate','strong'] for m in ['naive','audit_plugin','joint']])
    tablerows('full_channel.tex',[[label,method]+fmt(r)+[f"{r['median_seconds']:.2f}"] for c,label in [('moderate','独立'),('correlated_errors','方向相关'),('heterogeneous_errors','协变量异质')] for method,r in [('独立通道',row(c)),('完整通道',full[c])]])
    tablerows('mcmc.tex',[[label,method,f'{mean:.3f}',f'[{lo:.3f}, {hi:.3f}]'] for case,label in [('moderate','中等'),('strong','较强')] for method,mean,lo,hi in mcmc_rows(case)])
    print(json.dumps(dict(integrated=integrated,audit_paired=paired),indent=2))

def mcmc_rows(case):
    m=json.loads((ROOT/f'mcmc_{case}'/'result.json').read_text())
    p=json.loads((ROOT/'rho_integrated_final'/case/'seed202101.json').read_text())
    return [('Laplace（变换尺度 mode）',m['laplace']['estimate'][3],m['laplace']['lower'][3],m['laplace']['upper'][3]),
            ('MCMC（后验均值）',m['mean'][3],m['lower'][3],m['upper'][3]),
            ('一维积分（后验均值）',p['mean'],p['lower'],p['upper'])]

if __name__=='__main__':main()
