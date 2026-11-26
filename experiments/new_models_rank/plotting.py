"""Static scientific figures from audited summaries; primary AUROC axes are 0..1."""
import argparse,csv,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.new_models_rank.summarize import DATASETS,VARIANTS,CONDITIONS,SELECTED,RANKS,read
NAMES={'ncnc':'NCNC','lpformer':'LPFormer','refined_gae':'Refined-GAE','katz':'Katz H5','hlgnn':'HL-GNN'}
COLORS={'ncnc':'#176e86','lpformer':'#cf8610','refined_gae':'#8753a5','katz':'#777777','hlgnn':'#3c9c63'}
LABELS={'full':'Full','no_completion':'noCNC','no_cn':'noCN','uniform_attention':'Uniform','no_ppr_encoding':'noPE','no_norm':'noLN','dot_decoder':'Dot'}
VC=['#176e86','#cf8610','#8753a5']
def save(fig,out,name):
 fig.savefig(out/(name+'.png'),dpi=180,bbox_inches='tight');fig.savefig(out/(name+'.pdf'),bbox_inches='tight');plt.close(fig)
def line(ax,rows,color,label,style='-'):
 rr=sorted(rows,key=lambda r:int(r['k']));x=np.array([int(r['k']) for r in rr]);y=np.array([float(r['mean']) if r['mean'] not in ['',None] else np.nan for r in rr]);sd=np.array([float(r['sd']) if r['sd'] not in ['',None] else np.nan for r in rr]);valid=np.array([int(r['n_seeds_valid'])==3 for r in rr]);low=np.array([int(r['n_low_support'])>0 for r in rr]);strong=valid&~low
 yy=np.where(strong,y,np.nan);ax.plot(x,yy,color=color,label=label,lw=1.6,ls=style);ax.fill_between(x,np.maximum(0,np.where(strong,y-sd,np.nan)),np.minimum(1,np.where(strong,y+sd,np.nan)),color=color,alpha=.08)
 ax.scatter(x[valid&low],y[valid&low],facecolors='none',edgecolors=color,s=11,lw=.8)
 ax.set(xlim=(0,100),ylim=(0,1));ax.axvline(10,color='#aaaaaa',lw=.6,ls=':');ax.axhline(.5,color='gray',lw=.5,ls=':');ax.grid(alpha=.16)
def main(a):
 a.output.mkdir(parents=True,exist_ok=True);summary=read(a.results/'summary.csv');joint=read(a.results/'five_model_full_summary.csv')
 ds=[d for d in DATASETS if any(r['dataset']==d for r in summary)]
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
 for cond in SELECTED:
  fig,axes=plt.subplots(len(ds),2,figsize=(8.4,3.3*len(ds)),squeeze=False)
  for i,d in enumerate(ds):
   for j,rank in enumerate(RANKS):
    ax=axes[i,j]
    for m in ['katz','hlgnn','ncnc','lpformer','refined_gae']:
     rr=[r for r in joint if r['dataset']==d and r['model']==m and r['rank']==rank and r['condition']==cond]
     line(ax,rr,COLORS[m],NAMES[m],':' if m in ['katz','hlgnn'] else '-')
    ax.set(title=f'{d.capitalize()} | {rank}',xlabel='Top k% nodes (frozen clean train rank)',ylabel='Incident-edge AUROC')
  axes[0,0].legend(fontsize=8,loc='lower right');fig.suptitle(f'{cond}: identical original queries, cached graphs and published node ranks\nMean ± sample SD, 3 seeds; hollow dots = low support; k=0 undefined',fontsize=12);fig.tight_layout(rect=[0,0,1,.94]);save(fig,a.output,'rank_auc_'+cond)
 # All-condition independently retrained components.
 for cond in ['clean','ea_0.5','feature_shuffle']:
  for rank in RANKS:
   fig,axes=plt.subplots(len(ds),3,figsize=(11.7,2.25*len(ds)),squeeze=False)
   for i,d in enumerate(ds):
    for j,m in enumerate(VARIANTS):
     ax=axes[i,j]
     for v,col in zip(VARIANTS[m],VC):
      rr=[r for r in summary if r['dataset']==d and r['model']==m and r['variant']==v and r['condition']==cond and r['rank']==rank]
      line(ax,rr,col,LABELS[v])
     ax.set(title=f'{d.capitalize()} | {NAMES[m]}',xlabel='Top k% nodes',ylabel='Incident-edge AUROC')
     if i==0:ax.legend(fontsize=8,loc='lower right')
   fig.suptitle(f'{rank} | {cond}: full model and independently trained component variants\nMean ± sample SD; hollow dots = fewer than 20 positives or negatives',fontsize=12);fig.tight_layout(rect=[0,0,1,.94]);save(fig,a.output,f'components_{cond}_{rank}')
 robust=read(a.results/'robustness_summary.csv');lookup={(r['dataset'],r['model'],r['variant'],r['condition']):r for r in robust}
 fig,axes=plt.subplots(len(ds),3,figsize=(11.7,2.45*len(ds)),squeeze=False,sharey=True)
 for i,d in enumerate(ds):
  for j,m in enumerate(VARIANTS):
   ax=axes[i,j]
   for v,col in zip(VARIANTS[m],VC):
    rr=[lookup[d,m,v,c] for c in CONDITIONS];ax.errorbar(range(14),[float(r['auc_mean']) for r in rr],yerr=[float(r['auc_sd']) for r in rr],label=LABELS[v],color=col,lw=1.2,marker='o',ms=2,capsize=2)
   ax.set(title=f'{d.capitalize()} | {NAMES[m]}',ylim=(0,1),ylabel='Global AUROC');ax.grid(alpha=.2);ax.set_xticks(range(14),['Clean','RF10','RF25','RF50','DICE10','DICE25','DICE50','EA10','EA25','EA50','Del50','Hub50','Ins50','Shuffle'],rotation=65,ha='right',fontsize=8)
   if i==0:ax.legend(fontsize=8,loc='lower left')
 fig.suptitle('Complete shared protected-target protocol | all 14 conditions\nCora/Citeseer: reused 54 fits; PubMed: 27 fresh fits; same graph snapshots for all variants');fig.tight_layout(rect=[0,0,1,.94]);save(fig,a.output,'robustness_all')
 # Exact common test-subset coverage, independent of scores and model.
 fig,axes=plt.subplots(1,len(ds),figsize=(5*len(ds),3.7),squeeze=False)
 for i,d in enumerate(ds):
  ax=axes[0,i]
  for rank,style in [('degree','-'),('pagerank','--')]:
   rr=sorted([r for r in summary if r['dataset']==d and r['model']=='ncnc' and r['variant']=='full' and r['condition']=='clean' and r['rank']==rank],key=lambda r:int(r['k']))
   for field,col in [('mean_pos_coverage','#176e86'),('mean_neg_coverage','#c04747')]:ax.plot([int(r['k']) for r in rr],[float(r[field]) for r in rr],style,color=col,label=rank+': '+('positive' if 'pos' in field else 'negative'))
  ax.set(title=d.capitalize(),xlim=(0,100),ylim=(0,1),xlabel='Top k% nodes',ylabel='Test-edge coverage');ax.grid(alpha=.2)
 axes[0,0].legend(fontsize=8);fig.tight_layout();save(fig,a.output,'test_edge_coverage')
 # Show Top10 vs100 paired component effects for a fixed feature-mismatch condition.
 paired=read(a.results/'conditional_component_deltas.csv');fig,axes=plt.subplots(len(ds),3,figsize=(11.7,2.35*len(ds)),squeeze=False)
 for i,d in enumerate(ds):
  for j,m in enumerate(VARIANTS):
   ax=axes[i,j]
   for offset,(v,col) in enumerate(zip(VARIANTS[m][1:],VC[1:])):
    rr=[next(r for r in paired if (r['dataset'],r['model'],r['variant'],r['condition'],r['rank'],int(r['k']))==(d,m,v,'feature_shuffle','degree',k)) for k in [10,100]]
    ax.errorbar(np.array([0,1])+(offset-.5)*.1,[float(r['mean_pp']) for r in rr],yerr=[float(r['sd_pp']) for r in rr],fmt='o-',capsize=3,color=col,label='Full - '+LABELS[v])
   ax.axhline(0,color='gray',lw=.6);ax.set(title=f'{d.capitalize()} | {NAMES[m]}',xticks=[0,1],xticklabels=['Degree top10%','Global top100%'],ylabel='Full - variant (pp)');ax.grid(alpha=.2)
   if i==0:ax.legend(fontsize=8)
 fig.suptitle('Feature shuffle: component conclusions depend on the fixed node subset\nSame-seed differences, mean ± sample SD; not a significance test');fig.tight_layout(rect=[0,0,1,.94]);save(fig,a.output,'conditional_component_effects')
 print('Scientific plots complete',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--results',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
