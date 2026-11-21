"""Original vector schematics of the evaluated adapters; paper crops have separate provenance."""
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
ROOT=Path(__file__).resolve().parent/'figures'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'pdf.fonttype':42})
C={'base':'#e8f1fa','structure':'#e7f3e8','decode':'#fff0db','text':'#213448'}
def box(ax,x,y,w,h,s,color='base'):
 p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.02,rounding_size=0.06',facecolor=C[color],edgecolor='#45647c',linewidth=1.2);ax.add_patch(p);ax.text(x+w/2,y+h/2,s,ha='center',va='center',color=C['text'],fontsize=10)
def arrow(ax,a,b):ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->',lw=1.3,color='#45647c'))
def canvas():
 f,a=plt.subplots(figsize=(11,4.4));a.set(xlim=(0,11),ylim=(0,4.4));a.axis('off');return f,a
def finish(f,n):f.savefig(ROOT/(n+'.pdf'),bbox_inches='tight');plt.close(f)
f,a=canvas();box(a,.1,1.8,1.2,.8,'Graph A\nFeatures X');box(a,1.85,1.8,1.5,.8,'GCN\nnode states');arrow(a,(1.3,2.2),(1.85,2.2))
box(a,4,3.05,2.0,.7,'Endpoint product');box(a,4,1.85,2,.7,'Common-neighbor\nstate sum','structure');box(a,4,.35,2,.9,'Missing-link scores\n+ calibration\n(one-sided neighbors)','structure')
for y in [3.4,2.2,.8]:arrow(a,(3.35,2.2),(4,y))
box(a,6.8,.5,1.6,.65,'Weighted\ncompletion','structure');arrow(a,(6,.8),(6.8,.8));box(a,8.8,1.8,1.65,.8,'Author predictor\nlink logit','decode');arrow(a,(6,3.4),(8.8,2.4));arrow(a,(6,2.2),(8.8,2.2));arrow(a,(8.4,.8),(9.3,1.8))
a.text(.2,4.05,'NCNC: explicit pair-conditioned aggregation and probabilistic completion',fontsize=12,weight='bold');a.text(.2,.03,'noCNC: completion depth = 0.   noCN: endpoint predictor; GCN(A, X) remains.',fontsize=10);finish(f,'ncnc_evaluated_architecture')
f,a=canvas();box(a,.1,1.8,1.25,.8,'Graph A\nFeatures X');box(a,1.9,1.8,1.5,.8,'GCN\nnode states');arrow(a,(1.35,2.2),(1.9,2.2));box(a,3.9,3.1,2,.7,'Endpoint product');arrow(a,(3.4,2.2),(3.9,3.45));box(a,3.9,1.5,2,1,'PPR selection\nCN / 1-hop / far\ncandidate counts','structure');arrow(a,(1.35,1.8),(3.9,1.7));box(a,6.5,1.5,1.8,1,'Pair-to-node\nattention + PPR\nposition encoding','structure');arrow(a,(5.9,2),(6.5,2));arrow(a,(3.4,2.2),(6.5,2.4));box(a,8.9,1.8,1.5,.8,'Concatenate\n+ MLP logit','decode');arrow(a,(8.3,2),(8.9,2.2));arrow(a,(5.9,3.45),(9.5,2.6))
a.text(.2,4.05,'LPFormer: adaptive context for each target pair',fontsize=12,weight='bold');a.text(.2,.45,'Uniform: learned attention weights replaced by uniform weights on the same candidates.\nnoPE: positional encoding is zero; PPR selection and structural counts remain.',fontsize=10);finish(f,'lpformer_evaluated_architecture')
f,a=canvas();box(a,.1,1.8,1.2,.8,'Graph A\nFeatures X');box(a,1.85,1.8,1.45,.8,'First propagation\n+ projection');arrow(a,(1.3,2.2),(1.85,2.2));box(a,3.85,1.65,2.05,1.1,'LayerNorm + dropout\nthen shared linear\npropagation','structure');arrow(a,(3.3,2.2),(3.85,2.2));box(a,6.45,1.8,1.65,.8,'Learned softmax\ndepth mixture','structure');arrow(a,(5.9,2.2),(6.45,2.2));arrow(a,(3.3,2.5),(6.45,2.5));box(a,8.65,1.6,1.9,1.2,'Endpoint product\n+ normalized MLP\nlink logit','decode');arrow(a,(8.1,2.2),(8.65,2.2))
a.text(.2,4.05,'Refined-GAE: feature-input author LightGCN path + Hadamard MLP decoder',fontsize=12,weight='bold');a.text(.2,.6,'Cora / Citeseer: 4 propagation steps.   PubMed: 2 steps, depth parameters initialized [1, 0.5].\nnoLN: replace LayerNorm only.   Dot: independently retrain with dot-product decoder.',fontsize=10);finish(f,'refined_evaluated_architecture')
print('wrote 3 architecture diagrams')
