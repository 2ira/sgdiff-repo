"""Summarize paired seed curves; generate static, exportable plots."""
import argparse,csv,json
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate import write_csv


def main(root):
    root=Path(root); groups=defaultdict(list);dgroups=defaultdict(list)
    audit_count=0;max_error=0.0
    for path in sorted(root.glob('*_seed*/per_seed_curves.csv')):
        for r in csv.DictReader(path.open()):
            key=tuple(r[k] for k in ['dataset','model','variant','condition','rank','k'])
            groups[key].append(r)
        a=json.loads(path.with_name('audit.json').read_text());audit_count+=len(a['records'])
        max_error=max(max_error,max((r['published_auc_error'] or 0 for r in a['records']),default=0))
        delta=path.with_name('paired_deltas.csv')
        if delta.exists():
            for r in csv.DictReader(delta.open()):
                key=tuple(r[k] for k in ['dataset','model','variant','condition','rank','k'])
                dgroups[key].append(r)
    def aggregate(source,metric):
        result=[]
        for key,rows in sorted(source.items()):
            if len({r['seed'] for r in rows})!=len(rows):raise ValueError('Duplicate seed results')
            vals=[float(r[metric]) for r in rows if r[metric]!='']
            n=len(vals)
            result.append(dict(zip(['dataset','model','variant','condition','rank','k'],key),
                n_seeds_present=len(rows),n_seeds_valid=n,
                mean=float(np.mean(vals)) if n else None,sd=float(np.std(vals,ddof=1)) if n>1 else None,
                min_n_pos=min(int(r['n_pos']) for r in rows),min_n_neg=min(int(r['n_neg']) for r in rows),
                n_low_support=sum(int(r['low_support']) for r in rows),
                mean_pos_coverage=float(np.mean([float(r['pos_coverage']) for r in rows])),
                mean_neg_coverage=float(np.mean([float(r['neg_coverage']) for r in rows]))))
        return result
    summary=aggregate(groups,'auc');delta=aggregate(dgroups,'delta_auc')
    write_csv(root/'summary.csv',summary);write_csv(root/'delta_summary.csv',delta)
    write_csv(root/'key_points.csv',[r for r in summary if int(r['k']) in [1,5,10,20,50,100]])
    ds_order=[d for d in ['cora','citeseer','pubmed'] if any(r['dataset']==d for r in summary)]
    colors={'clean':'#126e82','ea_0.5':'#d98c10','delete_hub_0.5':'#c04747','feature_shuffle':'#6b4c9a'}
    conditions=list(colors)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(len(ds_order),2,figsize=(11,3.1*len(ds_order)),squeeze=False)
    for i,ds in enumerate(ds_order):
        for j,rank in enumerate(['degree','pagerank']):
            ax=axes[i,j]
            for cond in conditions:
                rows=sorted([r for r in summary if r['dataset']==ds and r['model']=='katz' and r['variant']=='full_h5' and r['rank']==rank and r['condition']==cond],key=lambda r:int(r['k']))
                # Missing seeds are not silently shown as a complete multi-seed estimate.
                x=np.array([int(r['k']) for r in rows]);y=np.array([r['mean'] if r['mean'] is not None and r['n_seeds_valid']==r['n_seeds_present'] else np.nan for r in rows])
                sd=np.array([r['sd'] if r['sd'] is not None else np.nan for r in rows])
                ax.plot(x,y,label=cond,color=colors[cond],lw=1.7,ls='--' if cond=='feature_shuffle' else '-')
                ax.fill_between(x,np.maximum(0,y-sd),np.minimum(1,y+sd),color=colors[cond],alpha=.09)
                low=np.array([r['n_low_support']>0 for r in rows]);ax.scatter(x[low],y[low],s=8,color=colors[cond],alpha=.6)
            ax.set(xlim=(0,100),ylim=(0,1),title=f'{ds.capitalize()} | {rank}',xlabel='Top k% nodes by clean training-graph rank',ylabel='Incident-edge AUROC')
            ax.axhline(.5,color='gray',lw=.6,ls=':');ax.grid(alpha=.15)
    axes[0,0].legend(fontsize=8,loc='lower right')
    fig.suptitle('Katz H=5: importance-conditioned robustness\nMean ± sample SD across existing split seeds; k=0 undefined; dots = low support',fontsize=12)
    fig.tight_layout(rect=[0,0,1,.95]);fig.savefig(root/'katz_rank_auc.png',dpi=180);fig.savefig(root/'katz_rank_auc.pdf');plt.close(fig)
    # Coverage diagnostics are common to all models/conditions for a dataset + split.
    fig,axes=plt.subplots(1,len(ds_order),figsize=(5*len(ds_order),3.7),squeeze=False)
    for i,ds in enumerate(ds_order):
        ax=axes[0,i]
        for rank,style in [('degree','-'),('pagerank','--')]:
            rows=sorted([r for r in summary if r['dataset']==ds and r['model']=='katz' and r['variant']=='full_h5' and r['rank']==rank and r['condition']=='clean'],key=lambda r:int(r['k']))
            for field,color in [('mean_pos_coverage','#126e82'),('mean_neg_coverage','#c04747')]:
                ax.plot([int(r['k']) for r in rows],[r[field] for r in rows],style,label=f'{rank}: {"positive" if "pos" in field else "negative"}',color=color)
        ax.set(xlim=(0,100),ylim=(0,1),title=ds.capitalize(),xlabel='Top k% nodes',ylabel='Fraction of original test edges retained');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8);fig.tight_layout();fig.savefig(root/'test_edge_coverage.png',dpi=180);fig.savefig(root/'test_edge_coverage.pdf');plt.close(fig)
    (root/'aggregation_audit.json').write_text(json.dumps(dict(records_recomputed=audit_count,summary_rows=len(summary),max_published_global_auc_error=max_error,seed_dispersion='sample standard deviation ddof=1; not a confidence interval'),indent=2)+'\n')
    print('Summarized',audit_count,'original evaluations; max historical AUC discrepancy',max_error)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=Path(__file__).parent/'results');a=p.parse_args();main(a.results)
