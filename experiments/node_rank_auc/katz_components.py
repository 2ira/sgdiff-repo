"""Summarize the archived Katz truncation ablations under the new conditional metric."""
import argparse,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from evaluate import write_csv

def main(a):
 a.output.mkdir(parents=True,exist_ok=True)
 rows={}
 for path in a.curves.glob('*_seed*/per_seed_curves.csv'):
  with path.open() as f:
   for r in csv.DictReader(f):
    if r['model']=='katz':rows[(r['dataset'],int(r['seed']),r['variant'],r['condition'],r['rank'],int(r['k']))]=r
 datasets=['cora','citeseer','pubmed'];conditions=sorted({k[3] for k in rows});delta=[]
 for d in datasets:
  for v in ['h3','h2']:
   for c in conditions:
    for rank in ['degree','pagerank']:
     for k in range(101):
      vals=[]
      for seed in range(3):
       full=rows[(d,seed,'full_h5',c,rank,k)];other=rows[(d,seed,v,c,rank,k)]
       assert (full['n_pos'],full['n_neg'])==(other['n_pos'],other['n_neg'])
       if full['auc'] and other['auc']:vals.append(100*(float(full['auc'])-float(other['auc'])))
      delta.append(dict(dataset=d,contrast='h5_minus_'+v,condition=c,rank=rank,k=k,n_seeds_valid=len(vals),mean_pp=float(np.mean(vals)) if vals else None,sd_pp=float(np.std(vals,ddof=1)) if len(vals)>1 else None,min_n_pos=min(int(rows[(d,s,'full_h5',c,rank,k)]['n_pos']) for s in range(3)),min_n_neg=min(int(rows[(d,s,'full_h5',c,rank,k)]['n_neg']) for s in range(3)),n_low_support=sum(int(rows[(d,s,'full_h5',c,rank,k)]['low_support']) for s in range(3))))
 write_csv(a.output/'katz_component_deltas.csv',delta)
 fig,axes=plt.subplots(3,3,figsize=(13,9),sharex=True,sharey=True)
 colors=['#176e86','#cf8610','#8e54a2']
 for i,d in enumerate(datasets):
  for j,c in enumerate(['clean','ea_0.5','delete_hub_0.5']):
   ax=axes[i,j]
   for v,color in zip(['full_h5','h3','h2'],colors):
    values=np.array([[float(rows[(d,s,v,c,'degree',k)]['auc']) if rows[(d,s,v,c,'degree',k)]['auc'] else np.nan for k in range(101)] for s in range(3)])
    mean=np.full(101,np.nan);sd=np.full(101,np.nan);ok=np.isfinite(values).all(0)
    mean[ok]=np.mean(values[:,ok],axis=0);sd[ok]=np.std(values[:,ok],axis=0,ddof=1)
    ax.plot(range(101),mean,label=v,color=color);ax.fill_between(range(101),np.maximum(0,mean-sd),np.minimum(1,mean+sd),color=color,alpha=.1)
    low=np.array([any(int(rows[(d,s,v,c,'degree',k)]['low_support']) for s in range(3)) for k in range(101)])
    ax.scatter(np.arange(101)[low],mean[low],s=8,color=color)
   ax.set(title=f'{d} | {c}',xlim=(0,100),ylim=(0,1),xlabel='Degree Top k% nodes',ylabel='Incident-edge AUROC');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8);fig.suptitle('Katz truncation ablations | archived predictions | mean ± sample SD; dots = low support');fig.tight_layout(rect=[0,0,1,.97]);fig.savefig(a.output/'katz_components.png',dpi=180);fig.savefig(a.output/'katz_components.pdf');plt.close(fig)
 print('Katz paired conditional ablations:',len(delta))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--curves',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
