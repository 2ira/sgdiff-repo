"""Publication-size figures from the report's frozen CSVs; no experimental reruns."""
from pathlib import Path
import csv, json, hashlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
R=Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':12,'axes.labelsize':11,'xtick.labelsize':9,'ytick.labelsize':10,'legend.fontsize':9,'pdf.fonttype':42})
DS=['cora','citeseer','pubmed'];MS=['ncnc','lpformer','refined_gae'];MN=['NCNC','LPFormer','Refined-GAE'];DN=['Cora','Citeseer','PubMed'];COL=['#2180af','#d77a1b','#32976b']
C=[('clean',0.)]+[(a,r) for a in ['rf','dice','ea'] for r in [.1,.25,.5]]
P=[('delete',.5),('delete_hub',.5),('insert',.5),('feature_shuffle',0.)]
VL=[('ncnc','no_completion','NCNC: noCNC'),('ncnc','no_cn','NCNC: noCN'),('lpformer','uniform_attention','LPF: Uniform'),('lpformer','no_ppr_encoding','LPF: noPE'),('refined_gae','no_norm','R-GAE: noLN'),('refined_gae','dot_decoder','R-GAE: Dot')]
def rows(tag,f):return list(csv.DictReader((R/'data'/tag/f).open()))
def get(rs,d,m,c,a,r,k):
 z=next((z for z in rs if z['dataset']==d and z['model']==m and z['component']==c and z['attack']==a and float(z['ratio'])==r),None)
 return 100*float(z[k]) if z else np.nan
def lab(a,r):return 'Clean' if a=='clean' else a.upper()+'\n'+str(int(r*100))+'%'
def finish(fig,n):fig.savefig(R/'figures'/(n+'.pdf'),bbox_inches='tight');plt.close(fig)
for tag in ['main','protected']:
 rs=rows(tag,'summary.csv');ds=DS if tag=='main' else DS[:2];con=C if tag=='main' else C+P
 labels=[lab(a,r) if a not in ['delete','delete_hub','insert','feature_shuffle'] else {'delete':'Del\n50%','delete_hub':'HubDel\n50%','insert':'Ins\n50%','feature_shuffle':'Shuffle'}[a] for a,r in con]
 # Keep two large metric panels per page, with larger labels than the original archive plots.
 fig,axs=plt.subplots(len(ds),2,figsize=(10.5,2.35*len(ds)),layout='constrained',squeeze=False)
 for i,d in enumerate(ds):
  for j,k in enumerate(['auc','ap']):
   ax=axs[i,j];z=np.array([[get(rs,d,m,'full',a,r,k+'_mean') for a,r in con] for m in MS]);cmap=plt.get_cmap('YlGnBu').copy();cmap.set_bad('#e2e4e6')
   assert np.isnan(z).sum()==(9 if d=='pubmed' else 0), (tag,d,k,z)
   ax.imshow(z,vmin=60 if tag=='protected' else 85,vmax=100,cmap=cmap,aspect='auto');ax.set(xticks=range(len(con)),xticklabels=labels,yticks=range(3),yticklabels=MN if j==0 else ['']*3,title=DN[DS.index(d)]+' | '+k.upper()+' (%)')
   ax.tick_params(axis='x',length=0,labelsize=7.5 if tag=='protected' else 9)
   for y in range(3):
    for x in range(len(con)):
     v=z[y,x];ax.text(x,y,'N/A' if np.isnan(v) else f'{v:.1f}',ha='center',va='center',fontsize=7 if tag=='protected' else 8.5,color='white' if v>(84 if tag=='protected' else 94) else '#17324d')
 finish(fig,'main_full_models' if tag=='main' else 'protected_performance_overview')
 fig,axs=plt.subplots(len(ds),3,figsize=(10.5,2.35*len(ds)),layout='constrained',squeeze=False)
 for i,d in enumerate(ds):
  vals=[get(rs,d,m,'full',a,r,'auc_mean') for m in MS for a,r in C];y0=np.nanmin(vals)-1.2;y1=np.nanmax(vals)+1.2
  for j,a in enumerate(['rf','dice','ea']):
   ax=axs[i,j]
   if d=='pubmed' and a=='ea':ax.axis('off');ax.text(.5,.5,'Skipped by main\n19,717 nodes > 4,000',ha='center',va='center',transform=ax.transAxes);continue
   for m,name,color in zip(MS,MN,COL):
    conds=[('clean',0.)]+[(a,r) for r in [.1,.25,.5]];y=[get(rs,d,m,'full',aa,r,'auc_mean') for aa,r in conds];sd=[get(rs,d,m,'full',aa,r,'auc_std') for aa,r in conds]
    ax.errorbar([0,10,25,50],y,yerr=sd,label=name,color=color,marker='o',ms=3.5,lw=1.5,capsize=2)
   ax.set(title=DN[DS.index(d)]+' | '+a.upper(),ylim=(y0,y1),xticks=[0,10,25,50]);ax.grid(alpha=.22)
   if j==0:ax.set_ylabel('AUC (%)')
   if i==len(ds)-1:ax.set_xlabel('Edit budget (%)')
   if i==0 and j==0:ax.legend(loc='lower left',framealpha=.85)
 finish(fig,tag+'_attack_curves')
 ps=rows(tag,'paired_components_summary.csv')
 fig,axs=plt.subplots(len(ds),1,figsize=(9.8,3.25*len(ds)),layout='constrained',squeeze=False)
 matrices=[]
 for d in ds:matrices.append(np.array([[get(ps,d,m,c,a,r,'full_minus_variant_auc_mean') for a,r in con] for m,c,label in VL]))
 cap=max(1.,max(np.nanmax(abs(z)) for z in matrices));norm=TwoSlopeNorm(vmin=-cap,vcenter=0,vmax=cap)
 for i,(d,z) in enumerate(zip(ds,matrices)):
  assert np.isnan(z).sum()==(18 if d=='pubmed' else 0), (tag,d,z)
  ax=axs[i,0];cmap=plt.get_cmap('RdBu').copy();cmap.set_bad('#e2e4e6');im=ax.imshow(z,cmap=cmap,norm=norm,aspect='auto');ax.set(yticks=range(6),yticklabels=[v[2] for v in VL],xticks=range(len(con)),xticklabels=labels,title=DN[DS.index(d)]+' | paired full - variant AUC (pp)');ax.tick_params(axis='x',length=0,labelsize=8.5 if tag=='protected' else 10)
  for y in range(6):
   for x in range(len(con)):
    v=z[y,x];t='N/A' if np.isnan(v) else ('0.0' if abs(v)<.05 else f'{v:.1f}');ax.text(x,y,t,ha='center',va='center',fontsize=8.5 if tag=='protected' else 10,color='white' if abs(v)>.7*cap else '#193348')
 fig.colorbar(im,ax=axs[:,0],shrink=.65,label='Full - variant AUC (pp)')
 finish(fig,tag+'_component_effects')
 # Probe figure is only from the original protected-target experiment.
 if tag=='protected':
  fig,axs=plt.subplots(1,2,figsize=(10.5,4.6),layout='constrained')
  for i,d in enumerate(ds):
   ax=axs[i];x=np.arange(4)
   for j,(m,name,col) in enumerate(zip(MS,MN,COL)):
    probes=[('delete',.5),('delete_hub',.5),('insert',.5),('feature_shuffle',0.)]
    # Shuffle is recorded with ratio 0 in some archives; select by attack explicitly.
    y=[];sd=[]
    for a,r in probes:
     z=next(z for z in rs if z['dataset']==d and z['model']==m and z['component']=='full' and z['attack']==a)
     y.append(100*float(z['drop_auc_mean']));sd.append(100*float(z['drop_auc_std']))
    ax.bar(x+(j-1)*.25,y,width=.24,label=name,color=col,yerr=sd,capsize=2)
   ax.set(xticks=x,xticklabels=['Random\ndelete 50%','Degree\ndelete 50%','Insert\n50%','Feature\nshuffle'],title=DN[i],ylabel='Clean - perturbation AUC (pp)');ax.grid(axis='y',alpha=.2)
   if i==0:ax.legend()
  finish(fig,'protected_probe_damage')
sources={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'data').rglob('*.csv')}
outputs={str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (R/'figures').glob('*.pdf')}
(R/'figure_audit.json').write_text(json.dumps(dict(passed=True,source_csv_sha256=sources,output_pdf_sha256=outputs,main_missing_cells='PubMed EA only; checked explicitly',scores_multiplied_by=100,uncertainty='sample SD ddof=1',source='included frozen result CSVs'),indent=2)+'\n')
print('7 report-sized figures rebuilt')
