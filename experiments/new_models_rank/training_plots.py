"""Plot all81 archived/new validation histories and their selected checkpoints."""
import argparse,io,json
from pathlib import Path
from zipfile import ZipFile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from experiments.new_models_rank.summarize import DATASETS,VARIANTS
from experiments.new_models_rank.plotting import NAMES,LABELS,save

def main(a):
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
 with ZipFile(a.records) as z:
  for d in DATASETS:
   fig,axes=plt.subplots(3,3,figsize=(11.7,7.5),squeeze=False)
   for i,m in enumerate(VARIANTS):
    for j,v in enumerate(VARIANTS[m]):
     ax=axes[i,j]
     for s,color in zip(range(3),['#176e86','#cf8610','#8753a5']):
      x=json.loads(z.read(f'runs/{d}_{m}_{v}_s{s}/training.json'));history=x['history'];selected=next(r for r in history if r['epoch']==x['best_epoch']);best=max(r['auc'] for r in history)
      assert abs(selected['auc']-best)<1e-12 and abs(best-x['best_val_auc'])<1e-12
      ax.plot([r['epoch'] for r in history],[r['auc'] for r in history],color=color,label=f'seed {s}',lw=1.1);ax.scatter([selected['epoch']],[selected['auc']],color=color,marker='*',s=50)
     ax.set(title=NAMES[m]+' | '+LABELS[v],xlabel='Epoch',ylabel='Val. AUROC (zoomed)',xlim=(0,100),ylim=(.75,1));ax.grid(alpha=.2)
   axes[0,0].legend(fontsize=8);fig.suptitle(f'{d}: all27 independent histories; stars = clean validation selections\nRaw attributes, train-only validation; fixed max100 epoch budget, not proof of convergence');fig.tight_layout(rect=[0,0,1,.94]);save(fig,a.output,'training_'+d)
 print('All81 validation selections checked and plotted',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--records',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
