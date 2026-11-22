"""Generate presentation/report tables directly from immutable fit records."""
import json,csv
from pathlib import Path

ROOT=Path('results/noisy_probit/state_channel')
OUT=Path('deliverables/noisy_probit_state_channel_english/data')
NAMES={'common':'Common','state':'State','full':'Full','state_shrink':'State-S','full_shrink':'Full-S'}


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    v=json.loads((ROOT/'validation/summary.json').read_text())
    dev=json.loads((ROOT/'development/summary.json').read_text())
    integ=json.loads((ROOT/'validation/integrity.json').read_text())
    rows={(r['case'],r['q'],r['method'],r['parameter']):r for r in v['rows']}
    def get(c,q,m,p='rho'):return rows[c,q,m,p]
    def interval(x,d=3):return '['+', '.join(f'{a:.{d}f}' for a in x)+']'
    def table(name,header,lines):
        (OUT/(name+'.tex')).write_text('\\toprule\n'+header+r'\\\midrule'+'\n'+'\n'.join(line+r'\\' for line in lines)+'\n\\bottomrule\n')
    def frow(r,withq=True):
        prefix=f"{r['audited_dyads']} & " if withq else ''
        return prefix+f"{NAMES[r['method']]} & {r['bias']:+.4f} & {r['rmse']:.4f} & {100*r['coverage']:.1f} & {r['width']:.3f}"
    table('development','Method & Bias & RMSE & Coverage (\\%) & Width',[
        frow(r,False) for r in dev['rows'] if r['parameter']=='rho'])
    table('primary','Audits & Method & Bias & RMSE & Cov. (\\%) & Width',[
        frow(get('state',q,m)) for q in [.01,.025,.05] for m in ['common','state','full']])
    table('controls','Scenario & Method & Bias & RMSE & Cov. (\\%) & Width',[
        ('Common truth' if c=='common' else 'Asymmetric truth')+' & '+frow(get(c,.05,m),False)
        for c in ['common','asymmetric'] for m in ['common','state','full']])
    table('sensitivity','Audits & Method & Bias & RMSE & Cov. (\\%) & Width',[
        frow(get('state',q,m)) for q in [.025,.05] for m in ['state','state_shrink','full','full_shrink']])
    table('contrasts','State / Full audits & RMSE ratio & 95\\% bootstrap CI & Coverage LCB & Rule',[
        f"{100 if name=='half_budget' else 200} / 200 & {c['rmse_ratio']:.3f} & {interval(c['rmse_ratio_ci95'])} & {100*c['a_coverage_wilson'][0]:.1f}\\% & {'Met' if c['all_gates'] else 'Not met'}"
        for name,c in v['primary_contrasts'].items()])
    table('beta','Parameter & Method & Bias & RMSE & Coverage (\\%)',[
        f"$\\beta_{j}$ & {m.title()} & {get('state',.05,m,p)['bias']:+.4f} & {get('state',.05,m,p)['rmse']:.4f} & {100*get('state',.05,m,p)['coverage']:.1f}"
        for j,p in enumerate(['beta0','beta1','beta2']) for m in ['common','state','full']])
    table('channel','Audits & $\\widehat\\alpha$ & $\\widehat\\gamma$ & $\\widehat\\kappa_{00}$ & $\\widehat\\kappa_D$ & $\\widehat\\kappa_{11}$',[
        str(get('state',a['q'],'state')['audited_dyads'])+' & '+' & '.join(f'{x:.3f}' for x in a['mean'])
        for a in v['auxiliary'] if a['case']=='state' and a['method']=='state'])
    for c in ['state','common','asymmetric']:
        with (OUT/(c+'.csv')).open('w') as f:
            w=csv.writer(f);w.writerow(['audits']+[f'{m}_{metric}' for m in ['common','state','full'] for metric in ['rmse','coverage']])
            for q in [.01,.025,.05]:
                w.writerow([get(c,q,'state')['audited_dyads']]+[get(c,q,m)[metric]*(100 if metric=='coverage' else 1)
                            for m in ['common','state','full'] for metric in ['rmse','coverage']])
    mcs={}
    for kind in ['state','full']:
        paths=list((ROOT/'mcmc').glob(kind+'_*/result.json'))
        latest=max(paths,key=lambda p:json.loads(p.read_text())['draws'])
        mcs[kind]=json.loads(latest.read_text())
    table('mcmc','Model & Draws/chain & Max $\\widehat R$ & Min ESS & Laplace 95\\% CI & MCMC 95\\% CI',[
        f"{k.title()} & {d['draws']:,} & {max(d['diagnostics']['rank_split_rhat']):.4f} & {min(d['diagnostics']['bulk_ess']):.0f} & {interval([d['laplace']['lower'][3],d['laplace']['upper'][3]])} & {interval([d['lower'][3],d['upper'][3]])}"
        for k,d in mcs.items()])
    macros={}
    for name,c in v['primary_contrasts'].items():
        key='Same' if name=='same_budget' else 'Half'
        macros.update({key+'Ratio':f"{c['rmse_ratio']:.3f}",key+'CI':interval(c['rmse_ratio_ci95']),
                       key+'Coverage':f"{100*c['a_coverage']:.1f}",key+'LCB':f"{100*c['a_coverage_wilson'][0]:.1f}",
                       key+'Result':'met' if c['all_gates'] else 'not met',key+'CoverageCI':interval([100*x for x in c['a_coverage_wilson']],1)})
    for m in ['common','state','full']:
        r=get('state',.05,m)
        macros.update({m.title()+'RMSE':f"{r['rmse']:.4f}",m.title()+'Bias':f"{r['bias']:+.4f}",
                       m.title()+'Coverage':f"{100*r['coverage']:.1f}",m.title()+'CoverageCI':interval([100*x for x in r['coverage_wilson']],1),m.title()+'MCSE':f"{r['rmse_mcse']:.4f}"})
    macros.update(TotalFits=str(integ['total_fits']),FailureCount=str(sum(r['failures'] for r in v['rows'] if r['parameter']=='rho')),
                  MaxGrad=f"{integ['max_gradient']:.5f}",FloorEvents=str(integ['floor_events']))
    macros.update({m.title()+'Seconds':f"{integ['median_pipeline_seconds']['state_0.05'][m]:.2f}" for m in ['common','state','full']})
    macros.update({k.title()+'Rhat':f"{max(d['diagnostics']['rank_split_rhat']):.4f}" for k,d in mcs.items()})
    macros.update({k.title()+'ESS':f"{min(d['diagnostics']['bulk_ess']):.0f}" for k,d in mcs.items()})
    domain_file=ROOT/'mcmc/summary.json'
    macros['DomainSlides']=''
    if domain_file.exists():
        domain=json.loads(domain_file.read_text());d=domain['domain_full_96000']
        delta=domain['domain_rho_max_endpoint_difference']
        status='meet' if d['converged'] else 'do not meet'
        note=(r"\textbf{Numerical-domain sensitivity.} Chains with Full's row-logit bounds expanded from $\pm12$ to $\pm20$ "
              +f"{status} the diagnostic rule at 96,000 draws/chain "
              +f"(maximum $\\widehat R={d['rhat_max']:.4f}$, minimum ESS {d['ess_min']:.0f}). "
              +f"The largest change in a $\\rho$ interval endpoint is {delta:.4f}. "
              +"This remains a fixed-data comparison of finite-domain targets.\n")
        (OUT/'domain_note.tex').write_text(note)
        macros['DomainSlides']=(f"Expanded Full logit bounds ($\\pm20$): max $\\widehat R={d['rhat_max']:.4f}$, "
                                 +f"min ESS {d['ess_min']:.0f}, $\\rho$ endpoint change {delta:.4f}.")
    else:(OUT/'domain_note.tex').write_text('')
    (OUT/'macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+value+'}' for k,value in macros.items())+'\n')
    (OUT/'provenance.json').write_text(json.dumps(dict(validation_lock=v['lock'],development_lock=dev['lock'],
         analysis_hash=v['analysis_hash'],integrity=integ,domain_diagnostics=json.loads(domain_file.read_text()) if domain_file.exists() else None,mcmc={k:dict(draws=d['draws'],converged=d['converged'],input_signature=d['input_signature']) for k,d in mcs.items()}),indent=2)+'\n')


if __name__=='__main__':main()
