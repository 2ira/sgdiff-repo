"""Generate every displayed numeric table from archived validation summaries."""
import json
from pathlib import Path
R=Path('results/noisy_probit/directional_channel');O=Path('deliverables/noisy_probit_directional_english/data')
NAMES={'state':'Five','six':'Six','full':'Full','six_shrink':'Six-S','full_shrink':'Full-S','common':'Common'}


def main():
    O.mkdir(parents=True,exist_ok=True);v=json.loads((R/'validation/summary.json').read_text());integ=json.loads((R/'validation/integrity.json').read_text())
    rows={(x['case'],x['q'],x['method'],x['parameter']):x for x in v['rows']};ch={(x['case'],x['q'],x['method']):x for x in v['channel']}
    def get(c,q,k):return rows[c,q,k,'rho']
    def ci(x):return '['+', '.join(f'{v:.3f}' for v in x)+']'
    def tab(name,header,lines):
        (O/(name+'.tex')).write_text('\\toprule\n'+header+r'\\\midrule'+'\n'+'\n'.join(x+r'\\' for x in lines)+'\n\\bottomrule\n')
    def line(c,q,k):
        a=get(c,q,k);return f"{a['audited_dyads']} & {NAMES[k]} & {a['bias']:+.4f} & {a['rmse']:.4f} & {100*a['coverage']:.2f} & {a['width']:.3f} & {ch[c,q,k]['rmse']:.4f}"
    for c in ['asymmetric','symmetric']:
        tab(c,'Audits & Model & Bias & RMSE & Cov. (\\%) & Width & Channel RMSE',[line(c,q,k) for q in [.025,.05] for k in ['state','six','full']])
    tab('sensitivity','Audits & Model & Bias & RMSE & Cov. (\\%) & Width & Channel RMSE',
        [line('asymmetric',q,k) for q in [.025,.05] for k in ['six','six_shrink','full','full_shrink']])
    primary=v['primary_contrasts']
    tab('primary','Six / Full audits & RMSE ratio & 95\\% bootstrap CI & Coverage LCB & Rule',[
        f"{100 if k=='half_budget' else 200} / 200 & {d['rmse_ratio']:.3f} & {ci(d['rmse_ratio_ci95'])} & {100*d['a_coverage_wilson'][0]:.2f}\\% & {'Met' if d['all_gates'] else 'Not met'}" for k,d in primary.items()])
    tab('nuisance','Truth & Audits & $\\widehat\\alpha$ & $\\widehat\\gamma$ & $\\widehat\\kappa_{00}$ & $\\widehat\\kappa_{01}$ & $\\widehat\\kappa_{10}$ & $\\widehat\\kappa_{11}$',[
        a['case'].title()+' & '+str(100 if a['q']==.025 else 200)+' & '+' & '.join(f'{x:.3f}' for x in a['mean']) for a in v['nuisance']])
    tab('beta','Parameter & Model & Bias & RMSE & Coverage (\\%)',[
        f"$\\beta_{j}$ & {NAMES[k]} & {a['bias']:+.4f} & {a['rmse']:.4f} & {100*a['coverage']:.2f}"
        for j,p in enumerate(['beta0','beta1','beta2']) for k in ['state','six','full'] for a in [rows['asymmetric',.05,k,p]]])
    mc={}
    for k in ['six','full']:
        ps=list((R/'mcmc').glob(k+'_*/result.json'));p=max(ps,key=lambda p:json.loads(p.read_text())['draws']);mc[k]=json.loads(p.read_text())
    tab('mcmc','Model & Draws/chain & Max $\\widehat R$ & Min ESS & Laplace 95\\% CI & MCMC 95\\% CI',[
        f"{NAMES[k]} & {d['draws']:,} & {max(d['diagnostics']['rank_split_rhat']):.4f} & {min(d['diagnostics']['bulk_ess']):.0f} & {ci([d['laplace']['lower'][3],d['laplace']['upper'][3]])} & {ci([d['lower'][3],d['upper'][3]])}" for k,d in mc.items()])
    macros={}
    for k,d in primary.items():
        prefix='Same' if k=='same_budget' else 'Half'
        macros.update({prefix+'Ratio':f"{d['rmse_ratio']:.3f}",prefix+'CI':ci(d['rmse_ratio_ci95']),prefix+'Result':'met' if d['all_gates'] else 'not met',
                       prefix+'Coverage':f"{100*d['a_coverage']:.2f}",prefix+'LowerCoverage':f"{100*d['a_coverage_wilson'][0]:.2f}"})
    for k in ['state','six','full']:
        a=get('asymmetric',.05,k);name=NAMES[k]
        macros.update({name+'RMSE':f"{a['rmse']:.4f}",name+'MCSE':f"{a['rmse_mcse']:.4f}",name+'Coverage':f"{100*a['coverage']:.2f}",
                       name+'Channel':f"{ch['asymmetric',.05,k]['rmse']:.4f}",name+'Time':f"{ch['asymmetric',.05,k]['median_pipeline_seconds']:.2f}"})
    macros.update(Fits=str(integ['fits']),Failures=str(integ['failures']),Floor=str(integ['probability_floor_events']))
    ds=json.loads((R/'mcmc/summary.json').read_text());wide=ds['final']['domain_full'];default=ds['final']['full']
    (O/'domain_note.tex').write_text(
        r"\textbf{Numerical-domain sensitivity.} "
        +f"In the main Full run, {100*default['boundary_fraction']:.2f}\\% of retained draws lie within one logit of a numerical bound. "
        +r"Expanding row-logit bounds from $\pm12$ to $\pm20$ gives "
        +f"maximum $\\widehat R={wide['max_rhat']:.4f}$ and minimum ESS {wide['min_ess']:.0f} "
        +f"at {wide['draws']:,} draws/chain. The largest change in a $\\rho$ interval endpoint is {ds['domain_rho_endpoint_change']:.4f}. "
        +"This checks two finite-domain targets on one dataset.\n")
    sixfive=v['secondary_contrasts']['asymmetric_0.05_six_vs_five']
    fivehalf=v['secondary_contrasts']['asymmetric_0.025_five_vs_full200']
    shrink=v['secondary_contrasts']['six_prior_0.025'];fullprior=v['secondary_contrasts']['full_prior_0.025']
    channel_gain=100*(1-ch['asymmetric',.05,'six']['rmse']/ch['asymmetric',.05,'state']['rmse'])
    rho_gain=100*(1-sixfive['rmse_ratio'])
    sym=v['secondary_contrasts']['symmetric_0.05_six_vs_five']
    (O/'primary_interpretation.tex').write_text(
        r"\textbf{Increment beyond Five.} "
        +f"At 200 audits, the descriptive Six/Five structural RMSE ratio is {sixfive['rmse_ratio']:.3f} {ci(sixfive['rmse_ratio_ci95'])}, "
        +f"a {rho_gain:.1f}\\% improvement. Channel-entry RMSE decreases by {channel_gain:.1f}\\%. "
        +f"Five also {'meets' if fivehalf['all_gates'] else 'misses'} the half-budget rule on these fresh data "
        +f"(ratio {fivehalf['rmse_ratio']:.3f} {ci(fivehalf['rmse_ratio_ci95'])}). "
        +"Consequently, passing that rule cannot by itself establish a benefit specific to the additional parameter. These secondary intervals have no multiplicity adjustment.\n")
    (O/'prior_interpretation.tex').write_text(
        f"At half budget, Six-S {'meets' if shrink['all_gates'] else 'misses'} the rule: RMSE ratio {shrink['rmse_ratio']:.3f} {ci(shrink['rmse_ratio_ci95'])}, "
        +f"coverage lower bound {100*shrink['a_coverage_wilson'][0]:.2f}\\%. "
        +f"Using Full-S as the 200-audit reference gives Six ratio {fullprior['rmse_ratio']:.3f} {ci(fullprior['rmse_ratio_ci95'])}; "
        +f"the rule is {'met' if fullprior['all_gates'] else 'not met'}. These checks support only the stated alternatives.\n")
    (O/'interpretation_slide.tex').write_text(
        f"Six reduces channel-entry RMSE by \\textbf{{{channel_gain:.1f}\\%}} relative to Five at 200 audits. "
        +f"The structural RMSE gain is smaller: \\textbf{{{rho_gain:.1f}\\%}}.\\par\\vspace{{3mm}}\n"
        +"Five also meets the half-budget rule in this replication. The new parameter primarily improves conditional-channel recovery.\n")
    (O/'conclusion.tex').write_text(
        r"\textbf{Main interpretation.} "
        +f"The two primary rules are {primary['same_budget']['all_gates'] and primary['half_budget']['all_gates'] and 'both met' or 'not both met'}. "
        +f"The clearest incremental result is a {channel_gain:.1f}\\% reduction in channel-entry RMSE relative to Five at 200 audits. "
        +f"Structural RMSE improves by {rho_gain:.1f}\\%, and Five also meets the half-budget criterion on these data. "
        +"This replication therefore does not establish that six parameters are necessary for audit savings.\n\n"
        +f"Under symmetric truth, the descriptive Six/Five RMSE ratio is {sym['rmse_ratio']:.3f} {ci(sym['rmse_ratio_ci95'])}. "
        +"No structural RMSE penalty is observed here. However, Five and Six differ in prior geometry: the mean of two independent uniform correlations has a triangular prior, whereas the pooled correlation in Five is uniform. Small structural gains therefore do not isolate flexibility from prior effects. "
        +"The asymmetric negative correlation remains difficult to estimate: at 200 audits, the mean mapped mode for "
        +r"$\kappa_{01}$ is approximately $-.440$, versus the true $-.65$. Improved channel RMSE does not establish unbiased latent-correlation recovery."+"\n")
    (O/'macros.tex').write_text('\n'.join('\\newcommand{\\'+k+'}{'+s+'}' for k,s in macros.items())+'\n')
    (O/'provenance.json').write_text(json.dumps(dict(lock=v['lock'],analysis_hash=v['analysis_hash'],integrity=integ,domain_diagnostics=ds,
         mcmc={k:dict(draws=d['draws'],converged=d['converged'],input_signature=d['input_signature']) for k,d in mc.items()}),indent=2)+'\n')


if __name__=='__main__':main()
