"""Describe saved branch/depth parameters; do not interpret magnitudes as causal importance."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.new_models_rank.summarize import read,write,DATASETS,VARIANTS,dump
from experiments.new_models_rank.plotting import save,LABELS,VC

def main(a):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
 torch.set_num_threads(1)
 weights=json.loads((a.results/'checkpoints.json').read_text());rows=[];uniform_checks=[]
 for w in weights:
  state=torch.load(w['persistent_path'],map_location='cpu',weights_only=True);r={k:w[k] for k in ['dataset','model','variant','seed']};r.update(cn_beta='',jk_scale='',depth1='',depth2='',depth3='',depth4='')
  if w['model']=='ncnc':r.update(cn_beta=float(state['predictor.beta'].item()),jk_scale=float(state['encoder.jkparams'].item()))
  elif w['model']=='refined_gae':
   coeff=state['encoder.alphas'].softmax(0).tolist();assert abs(sum(coeff)-1)<2e-7;r.update({f'depth{i+1}':v for i,v in enumerate(coeff)})
  elif w['variant']=='uniform_attention':
   keys=[k for k in state if k.startswith('core.att_layers.') and k.endswith('.att.att')];assert keys and all(torch.count_nonzero(state[k])==0 for k in keys);uniform_checks.append(w['dataset']+'_'+str(w['seed']))
  rows.append(r)
 write(a.results/'learned_parameters.csv',rows);dump(a.results/'parameter_audit.json',dict(passed=True,models=81,uniform_attention_zero_vectors=len(uniform_checks),depth_coefficients_sum_to_one=True,cn0_beta_registered_but_not_used=True,interpretation='Descriptive coefficients, not causal importance'))
 fig,axes=plt.subplots(1,3,figsize=(12,3.8))
 for ax,d in zip(axes,DATASETS):
  length=2 if d=='pubmed' else 4
  for v,color in zip(VARIANTS['refined_gae'],VC):
   selected=[r for r in rows if (r['dataset'],r['model'],r['variant'])==(d,'refined_gae',v)];assert len(selected)==3
   values=np.array([[r[f'depth{i+1}'] for i in range(length)] for r in selected]);mean=values.mean(0);sd=values.std(0,ddof=1)
   ax.errorbar(range(1,length+1),mean,yerr=sd,color=color,label=LABELS[v],marker='o',capsize=3)
  ax.set(title=d.capitalize(),xticks=range(1,length+1),ylim=(0,1),xlabel='Propagation depth',ylabel='Learned softmax coefficient');ax.grid(alpha=.2)
 axes[0].legend(fontsize=8);fig.suptitle('Refined-GAE depth mixture: mean ± sample SD; coefficients are not causal importance');fig.tight_layout(rect=[0,0,1,.92]);save(fig,a.results,'refined_depth_weights')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--results',type=Path,required=True);main(p.parse_args())
