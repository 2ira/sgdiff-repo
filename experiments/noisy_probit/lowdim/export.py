"""Export evidence for the English Beamer deck; no manual transcription."""
import json
from pathlib import Path
import numpy as np
from ..run import write_json

ROOT=Path('results/noisy_probit/lowdim')
OUT=Path('deliverables/noisy_probit_lowdim_english/data')
LABELS={'independent':'Independent','copula':'Copula','full':'Full','copula_shrink':'Copula-S','full_shrink':'Full-S'}
CASES={'positive':'Positive','independent':'Independent','negative':'Negative','state_dependent':'State-dependent'}


def texrows(name,rows):
    (OUT/name).write_text('\n'.join(' & '.join(map(str,r))+r' \\' for r in rows)+'\n')


def pct(value):return f'{100*value:.1f}\\%'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    val=json.loads((ROOT/'validation/summary.json').read_text())
    dev=json.loads((ROOT/'development/summary.json').read_text())
    def row(case,q,method,param='rho'):
        return next(r for r in val['rows'] if r['case']==case and r['q']==q and r['method']==method and r['parameter']==param)
    texrows('development.tex',[[('Independent' if r['case']=='legacy_independent' else 'Shared errors'),LABELS[r['method']],
             f"{r['rmse']:.3f}",pct(r['coverage']),f"{r['failures']}/{r['n']}"] for r in dev['rows'] if r['parameter']=='rho'])
    lines=['q copula copula_err full full_err copula_cov full_cov']
    for q in [.01,.025,.05]:
        a=row('positive',q,'copula');b=row('positive',q,'full')
        lines.append(' '.join(map(str,[100*q,a['rmse'],1.95996*a['rmse_mcse'],b['rmse'],1.95996*b['rmse_mcse'],100*a['coverage'],100*b['coverage']])))
    (OUT/'primary.dat').write_text('\n'.join(lines)+'\n')
    rows=[]
    for key in ['same_budget','half_budget']:
        c=val['primary_contrasts'][key];lo,hi=c['rmse_ratio_ci95']
        rows.append([f"{100*c['q_a']:g}\\% / 5\\%",f"{c['rmse_ratio']:.3f}",f'[{lo:.3f}, {hi:.3f}]',
                     f"{c['upper_one_sided_975']:.3f}",pct(c['a_coverage_wilson'][0]),'Meets' if c['all_gates'] else 'Not met'])
    texrows('primary_contrasts.tex',rows)
    texrows('prior_sensitivity.tex',[[f'{100*q:g}\\%',LABELS[m],f"{row('positive',q,m)['rmse']:.3f}",
             pct(row('positive',q,m)['coverage']),f"{row('positive',q,m)['width']:.3f}"]
             for q in [.025,.05] for m in ['copula','copula_shrink','full','full_shrink']])
    for case in ['independent','negative','state_dependent']:
        texrows(case+'.tex',[[f'{100*q:g}\\%',LABELS[m],f"{row(case,q,m)['rmse']:.3f}",f"{row(case,q,m)['bias']:+.3f}",
                    pct(row(case,q,m)['coverage']),f"{row(case,q,m)['width']:.3f}",f"{row(case,q,m)['failures']}/200"]
                    for q in [.025,.05] for m in ['independent','copula','full']])
    texrows('dependence.tex',[[CASES[a['case']],f"{a['kappa_mean']:.3f}",f"{a['rho_kappa_correlation_median']:+.3f}",
                             f"{a['rho_kappa_abs_correlation_p90']:.3f}"] for a in val['auxiliary'] if a['q']==.025])
    texrows('beta.tex',[[CASES[c],LABELS[m],f"{row(c,.025,m,'beta0')['rmse']:.3f}"]+
                      [pct(row(c,.025,m,p)['coverage']) for p in ['beta0','beta1','beta2']]
                      for c in CASES for m in ['copula','full']])
    texrows('budget_secondary.tex',[[f'{100*q:g}\\% / 5\\%',f"{c['rmse_ratio']:.3f}",
                   f"[{c['rmse_ratio_ci95'][0]:.3f}, {c['rmse_ratio_ci95'][1]:.3f}]",pct(c['a_coverage'])]
                   for q in [.01,.025,.05] for c in [val['secondary_contrasts'][f'positive_q{q:g}_vs_full_5']]])
    intervalrows=[];diagrows=[];mcmc_summary=[]
    for case in ['positive','state_dependent']:
        for model in ['copula','full']:
            base=ROOT/'mcmc'/f'{case}_{model}'
            # Only explicit longer-run suffixes are eligible, never another seed.
            choices=[base]+sorted(base.parent.glob(base.name+'_extended*'))
            runs=[(p,json.loads((p/'result.json').read_text())) for p in choices if (p/'result.json').exists()]
            if not runs:raise FileNotFoundError(f'Missing fixed MCMC diagnostic {base}')
            path,d=max(runs,key=lambda v:v[1]['draws'])
            lap=d['laplace'];interval=lambda a,b:f'[{a:.3f}, {b:.3f}]'
            intervalrows.append([CASES[case],LABELS[model],f"{lap['estimate'][3]:.3f}",interval(lap['lower'][3],lap['upper'][3]),
                                 f"{d['mean'][3]:.3f}",interval(d['lower'][3],d['upper'][3])])
            rh=max(d['diagnostics']['rank_split_rhat']);ess=min(d['diagnostics']['bulk_ess'])
            diagrows.append([CASES[case],LABELS[model],f"{d['draws']:,}",f'{rh:.4f}',f'{ess:.0f}','Met' if d['converged'] else 'Not met'])
            mcmc_summary.append(dict(case=case,method=model,path=str(path),max_rhat=rh,min_ess=ess,converged=d['converged'],
                                     draws=d['draws'],attempts=[dict(path=str(p),draws=r['draws'],converged=r['converged']) for p,r in runs]))
    texrows('mcmc_intervals.tex',intervalrows);texrows('mcmc_diagnostics.tex',diagrows)
    failures=sum(r['failures'] for r in val['rows'] if r['parameter']=='rho')
    fits=sum(r['n'] for r in val['rows'] if r['parameter']=='rho')
    stress=row('state_dependent',.05,'copula');full=row('state_dependent',.05,'full')
    stress_sentence=f"With state-dependent errors at 5\\% audit, Copula coverage is {pct(stress['coverage'])}, versus {pct(full['coverage'])} for Full."
    macros=r'\newcommand{\ValidationFailureText}{'+f'Validation fits: {fits:,}; numerical failures: {failures}. All are included in the reported summaries.'+'}\n'
    macros+=r'\newcommand{\StressConclusion}{'+stress_sentence+'}\n'
    (OUT/'macros.tex').write_text(macros)
    write_json(ROOT/'delivery_summary.json',dict(primary=val['primary_contrasts'],mcmc=mcmc_summary,
                                               validation_fits=fits,numerical_failures=failures,stress_copula=stress,stress_full=full))
    print(json.dumps(dict(fits=fits,failures=failures,mcmc=mcmc_summary),indent=2))


if __name__=='__main__':main()
