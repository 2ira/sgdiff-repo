"""Regenerate every numerical table from the included, fingerprinted experiment data."""
from pathlib import Path
import csv, json, hashlib, statistics, math
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'generated';OUT.mkdir(exist_ok=True)
DS=['cora','citeseer','pubmed'];MS=['ncnc','lpformer','refined_gae']
DN={'cora':'Cora','citeseer':'Citeseer','pubmed':'PubMed'}
MN={'ncnc':'NCNC','lpformer':'LPFormer','refined_gae':'Refined-GAE'}
CM={'full':'Full','no_completion':'noCNC','no_cn':'noCN','uniform_attention':'Uniform','no_ppr_encoding':'noPE','no_norm':'noLN','dot_decoder':'Dot'}
AR=['clean','rf','dice','ea','delete','delete_hub','insert','feature_shuffle']
def read(tag,name):
 with (ROOT/'data'/tag/name).open() as f:return list(csv.DictReader(f))
def esc(s):
 for a,b in [('\\',r'\textbackslash{}'),('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:s=str(s).replace(a,b)
 return s

def cond(r):
 a=r['attack'];v=float(r['ratio'])
 names={'clean':'Clean','rf':'RF','dice':'DICE','ea':'EA','delete':'Del','delete_hub':'HubDel','insert':'Ins','feature_shuffle':'Shuffle'}
 # Read original labels rather than silently omitting archived probes.
 return names.get(a,esc(a))+((' '+str(round(v*100))+r'\%') if v and a!='feature_shuffle' else '')
def fmt(v):return f'{100*float(v):.2f}' if math.isfinite(float(v)) else 'N/A'
def pm(r,k):return '$'+fmt(r[k+'_mean'])+r'\pm'+fmt(r[k+'_std'])+'$'
def order(r):
 a=r['attack'];return (a!='clean',AR.index(a) if a in AR else 99,a,float(r['ratio']))
def table(caption,headers,body,spec=None,size='small'):
 spec=spec or 'l'+'r'*(len(headers)-1)
 n=len(headers)
 return ('{\\'+size+'\n\\setlength{\\tabcolsep}{3.2pt}\n\\renewcommand{\\arraystretch}{1.10}\n'
   +'\\begin{longtable}{'+spec+'}\n\\caption{'+caption+'}\\\\\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n\\endfirsthead\n'
   +'\\multicolumn{'+str(n)+'}{l}{'+caption+'（续）}\\\\\n\\toprule\n'+' & '.join(headers)+r' \\'+'\n\\midrule\n\\endhead\n'
   +'\\midrule\\multicolumn{'+str(n)+'}{r}{续下页}\\\\\n\\endfoot\n\\bottomrule\n\\endlastfoot\n'
   +'\n'.join(' & '.join(map(str,row))+r' \\' for row in body)+'\n\\end{longtable}\n}\n')
def save(n,s):(OUT/(n+'.tex')).write_text(s)
counts={}
for tag in ['main','protected']:
 rows=read(tag,'summary.csv');seed=read(tag,'per_seed.csv');paired=read(tag,'paired_components_summary.csv')
 expected=(243,729,162) if tag=='main' else (252,756,168)
 assert (len(rows),len(seed),len(paired))==expected
 groups={}
 for r in seed:groups.setdefault(tuple(r[k] for k in ['dataset','model','component','attack','ratio']),[]).append(r)
 metrics=['auc','ap','drop_auc','drop_ap']+(['hit@100','strict_hits100','precision100_tie_average'] if tag=='main' else ['hits100','precision100'])
 for r in rows:
  g=groups[tuple(r[k] for k in ['dataset','model','component','attack','ratio'])]
  assert len(g)==3 and {x['seed'] for x in g}=={'0','1','2'}
  for k in metrics:
   vals=[float(x[k]) for x in g]
   for suf,val in [('mean',statistics.mean(vals)),('std',statistics.stdev(vals))]:assert math.isclose(float(r[k+'_'+suf]),val,abs_tol=1e-11)
 lookup={tuple(r[k] for k in ['dataset','model','component','attack','ratio','seed']):r for r in seed}
 for r in paired:
  key=tuple(r[k] for k in ['dataset','model','component','attack','ratio'])
  for metric in ['auc','ap']:
   diffs=[];drops=[]
   for v in groups[key]:
    f=lookup[(r['dataset'],r['model'],'full',r['attack'],r['ratio'],v['seed'])]
    diffs.append(float(f[metric])-float(v[metric]))
    drops.append(float(v['drop_'+metric])-float(f['drop_'+metric]))
   for pre,vals in [('full_minus_variant_',diffs),('variant_minus_full_drop_',drops)]:
    for suf,val in [('mean',statistics.mean(vals)),('std',statistics.stdev(vals))]:assert math.isclose(float(r[pre+metric+'_'+suf]),val,abs_tol=1e-11)
 tables=[];paired_tables=[];seedtables=[]
 for d in DS:
  for m in MS:
   subset=sorted([r for r in rows if r['dataset']==d and r['model']==m],key=lambda r:(r['component']!='full',r['component'],order(r)))
   if not subset:continue
   headers=['版本','条件','AUC','AP',r'$P_{100}$',r'$H_{100}$',r'$\Delta$AUC',r'$\Delta$AP']
   mk=['auc','ap','hit@100' if tag=='main' else 'precision100','strict_hits100' if tag=='main' else 'hits100','drop_auc','drop_ap']
   tables.append(table(f'{DN[d]} / {MN[m]}：全部条件，{tag} 协议',[*headers],[[CM[r['component']],cond(r),*[pm(r,k) for k in mk]] for r in subset],spec='llrrrrrr',size='footnotesize'))
   ps=sorted([r for r in paired if r['dataset']==d and r['model']==m],key=lambda r:(r['component'],order(r)))
   pk=['full_minus_variant_auc','full_minus_variant_ap','variant_minus_full_drop_auc','variant_minus_full_drop_ap']
   paired_tables.append(table(f'{DN[d]} / {MN[m]}：配对组件效应，{tag} 协议',['变体','条件',r'$F-V$ AUC',r'$F-V$ AP',r'$D_V-D_F$ AUC',r'$D_V-D_F$ AP'],[[CM[r['component']],cond(r),*[pm(r,k) for k in pk]] for r in ps],spec='llrrrr',size='footnotesize'))
   ss=sorted([r for r in seed if r['dataset']==d and r['model']==m],key=lambda r:(r['component']!='full',r['component'],int(r['seed']),order(r)))
   sm=['auc','ap','hit@100' if tag=='main' else 'precision100','strict_hits100' if tag=='main' else 'hits100']
   if tag=='main':sm+=['precision100_tie_average']
   sm+=['drop_auc','drop_ap']
   sh=['版本','种子','条件','AUC','AP',r'$P_{100}$',r'$H_{100}$']+([r'$\bar P_{100}$'] if tag=='main' else [])+[r'$\Delta$AUC',r'$\Delta$AP']
   seedtables.append(table(f'{DN[d]} / {MN[m]}：逐种子结果，{tag} 协议',sh,[[CM[r['component']],r['seed'],cond(r),*[fmt(r[k]) for k in sm]] for r in ss],spec='rll'+'r'*len(sm),size='footnotesize'))
 save(tag+'_all', '\n'.join(tables));save(tag+'_paired','\n'.join(paired_tables));save(tag+'_seeds','\n'.join(seedtables))
 trains=read(tag,'training_summary.csv');body=[]
 for r in sorted(trains,key=lambda r:(DS.index(r['dataset']),MS.index(r['model']),r['component']!='full',r['component'],int(r['seed']))):
  mem=float(r['peak_cuda_bytes' if tag=='main' else 'peak_torch_allocated_bytes'])/2**30
  sec=float(r['seconds' if tag=='main' else 'training_seconds_excluding_ppr'])
  body.append([DN[r['dataset']],MN[r['model']],CM[r['component']],r['seed'],r['best_epoch'],r['stopped_epoch'],fmt(r['best_val_auc']),f'{sec:.1f}',f'{mem:.3f}'])
 save(tag+'_training',table(tag+' 协议：每次训练的选点与资源记录',['数据','模型','版本','种子','最优轮','停止轮','验证AUC','秒','GiB'],body,spec='lllrrrrrr',size='footnotesize'))
 counts[tag]=dict(training=len(trains),summary=len(rows),evaluations=len(seed),paired_summary=len(paired))
# Main overview with absolute performance; missing EA is explicit.
rows=read('main','summary.csv');body=[]
for d in DS:
 for m in MS:
  cells=[]
  for a,v in [('clean',0),('rf',.5),('dice',.5),('ea',.5)]:
   r=next((r for r in rows if r['dataset']==d and r['model']==m and r['component']=='full' and r['attack']==a and float(r['ratio'])==v),None)
   cells.append(pm(r,'auc') if r else 'N/A')
  body.append([DN[d],MN[m],*cells])
save('main_overview',table('main 协议：完整模型 AUC（百分数）',['数据','模型','Clean','RF50','DICE50','EA50'],body,spec='llrrrr',size='small'))
# Attack budgets retain raw and inference edit counts; clean rows included.
manifest=json.loads((ROOT/'data/main/input_manifest.json').read_text());body=[]
for r in sorted(manifest['attacks'],key=lambda r:(DS.index(r['dataset']),r['seed'],order(r))):
 body.append([DN[r['dataset']],r['seed'],cond(r),r['requested_budget'],r['raw_deleted'],r['raw_inserted'],r['visible_deleted'],r['visible_inserted'],r['inserted_scored_negatives']])
save('main_budgets',table('main 协议：完整图及推理图实际编辑数',['数据','种子','条件','预算','原删','原插','可见删','可见插','负例插入'],body,spec='lr l rrrrrr',size='footnotesize'))
# Main input/split counts from released snapshots are frozen alongside the report.
stats=[]
split_counts=json.loads((ROOT/'data/main/split_counts.json').read_text())
for d in DS:
 ns=[split_counts[d][k] for k in ['train_pos','val_pos','test_pos']]
 info=manifest['datasets'][d];stats.append([DN[d],info['nodes'],info['edges'],info['feature_dim'],*ns])
save('datasets',table('规范化数据统计与每个种子的正例划分（各划分负例数相同）',['数据','节点','无向边','属性维数','训练','验证','测试'],stats,spec='lrrrrrr'))
# Hyperparameter table directly from run manifests.
runs=json.loads((ROOT/'data/main/run_manifest.json').read_text());body=[]
for d in DS:
 for m in MS:
  c=next(r for r in runs if r['dataset']==d and r['model']==m and r['component']=='full')['settings']
  body.append([DN[d],MN[m],c['hidden'],str(c['dropout']),str(c['lr'])+('/'+str(c['predictor_lr']) if 'predictor_lr' in c else ''),str(c['weight_decay']),str(c['decay']),c['author_batch_size']])
save('hyperparameters',table('主训练配置；统一采用 batch512，dropout 指预测器/共享配置（详见正文）',['数据','模型','维数','dropout','学习率','衰减项','LR因子','作者batch'],body,spec='llrrrrrr',size='footnotesize'))
# Selected old-protocol strata, with per-class counts, not pooled correlations.
rows=read('protected','strata_summary.csv');body=[]
for r in rows:
 if r['attack']=='clean':body.append([DN[r['dataset']],MN[r['model']],esc(r['stratum']),f"{float(r['n_positive_mean']):.1f}",f"{float(r['n_negative_mean']):.1f}",pm(r,'auc'),pm(r,'ap')])
save('protected_strata',table('候选保护协议：干净图分层结果，计数为三种子均值',['数据','模型','分层','正例数','负例数','AUC','AP'],body,spec='lllrrrr',size='footnotesize'))
# A report is built from all numeric experiments, not new measurements.
files={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').rglob('*') if p.is_file()}
(ROOT/'table_audit.json').write_text(json.dumps(dict(passed=True,counts=counts,input_sha256=files,mean_std_recomputed=True,unit='all scores and differences times 100; sample SD ddof=1'),indent=2)+'\n')
print(json.dumps(counts))
# Readable main-body component and old-probe summaries.
pr=read('main','paired_components_summary.csv');body=[]
for d in DS:
 for m in MS:
  variants=sorted({r['component'] for r in pr if r['dataset']==d and r['model']==m})
  for v in variants:
   cells=[]
   for a,rat in [('clean',0),('rf',.5),('dice',.5),('ea',.5)]:
    r=next((r for r in pr if r['dataset']==d and r['model']==m and r['component']==v and r['attack']==a and float(r['ratio'])==rat),None)
    cells.append(pm(r,'full_minus_variant_auc') if r else 'N/A')
   body.append([DN[d],MN[m],CM[v],*cells])
save('main_paired_overview',table('M 协议组件效应：完整模型减变体 AUC（百分点）',['数据','模型','变体','Clean','RF50','DICE50','EA50'],body,spec='lllrrrr',size='footnotesize'))
r=read('protected','summary.csv');body=[]
for d in DS[:2]:
 for m in MS:
  cells=[]
  for a in ['clean','delete','delete_hub','insert','feature_shuffle']:
   x=next(x for x in r if x['dataset']==d and x['model']==m and x['component']=='full' and x['attack']==a)
   cells.append(pm(x,'auc'))
  body.append([DN[d],MN[m],*cells])
save('protected_probes',table('P 协议完整模型的干净与机制探针 AUC（百分数）',['数据','模型','Clean','Del50','HubDel50','Ins50','Shuffle'],body,spec='llrrrrr',size='footnotesize'))
a=json.loads((ROOT/'data/protected/protocol_audit.json').read_text())
body=[]
for r in a['attacks']:
 ds=r['file'].split('_s')[0];seed=r['file'].split('_s')[1].split('_')[0]
 body.append([DN[ds],seed,cond(r),r['base_edges'],r['requested_budget'],r['deleted'],r['inserted'],r['effective_edits']])
save('protected_budgets',table('P 协议：78张共享结构图（含6张clean）的实际编辑',['数据','种子','条件','基图边数','请求','删除','插入','差集数'],body,spec='lr l rrrrr',size='footnotesize'))
rows=read('protected','strata_summary.csv');tables=[]
for d in DS[:2]:
 for m in MS:
  body=[]
  for r in rows:
   if r['dataset']!=d or r['model']!=m:continue
   def maybe(k):return pm(r,k) if r[k+'_mean'] and math.isfinite(float(r[k+'_mean'])) else 'N/A'
   body.append([esc(r['stratum']),cond(r),r['n_seeds'],f"{float(r['n_positive_mean']):.1f}",f"{float(r['n_negative_mean']):.1f}",maybe('auc'),maybe('ap'),maybe('drop_auc')])
  tables.append(table(DN[d]+' / '+MN[m]+'：P 协议全部候选分层',['分层','条件','n','正例数','负例数','AUC','AP',r'$\Delta$AUC'],body,spec='llrrrrrr',size='footnotesize'))
save('protected_strata_all','\n'.join(tables))
