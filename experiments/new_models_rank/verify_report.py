"""Audit the completed scientific report and portable source package."""
import argparse,csv,hashlib,json,re,sys
from pathlib import Path
from zipfile import ZipFile
import pymupdf
root=Path(__file__).resolve().parents[2]
p=argparse.ArgumentParser(description='Verify complete report tables, figures, PDF and archives; needs PyMuPDF.')
p.add_argument('--results',type=Path,default=root/'results/new-models-node-rank')
out=p.parse_args().results.resolve()
sys.path.insert(0,str(root))
from experiments.new_models_rank.plotting import LABELS
from experiments.new_models_rank.tex_report import esc

def read(name):
 with (out/name).open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
expected={'per_seed.csv':1134,'robustness_summary.csv':378,'paired_component_deltas.csv':252,'summary.csv':76356,'conditional_component_deltas.csv':50904,'training_summary.csv':81,'learned_parameters.csv':81,'five_model_full_summary.csv':42420}
counts={n:len(read(n)) for n in expected};assert counts==expected,counts
assert {r['model'] for r in read('five_model_full_summary.csv')}=={'katz','hlgnn','ncnc','lpformer','refined_gae'}
for name in ['delivery_audit.json','package_audit.json','replay_audit.json','new_checkpoint_replay_audit.json','parameter_audit.json','feature_permutation_audit.json']:
 assert json.loads((out/name).read_text())['passed'],name
old=json.loads((out/'reused_checkpoint_replay_audit.json').read_text());assert old['checkpoints']==54 and old['evaluations']==162
new=json.loads((out/'new_checkpoint_replay_audit.json').read_text());assert new['checkpoints']==27 and new['evaluations']==81
tex=(out/'report.tex').read_text();md=(out/'report.md').read_text()
for r in read('per_seed.csv'):
 scores=[f'{float(r[k]):.6f}' for k in ['auc','ap','hits100','precision100']]
 assert ' & '.join([LABELS[r['variant']],r['seed'],esc(r['condition']),*scores])+r' \\' in tex
 assert '|'+'|'.join([LABELS[r['variant']],r['seed'],r['condition'],*scores])+'|' in md
figures=re.findall(r'\\includegraphics\[[^\]]*\]\{([^}]+)\}',tex)
assert len(figures)==len(set(figures))==17
for n in figures:assert (out/n).exists() and (out/Path(n).with_suffix('.png')).exists()
for n in ['predictions.zip','training_records.zip','per_seed_curves.zip','report_sources.zip']:
 with ZipFile(out/n) as z:assert z.testzip() is None
with ZipFile(out/'predictions.zip') as z:
 assert len([n for n in z.namelist() if n.startswith('runs/') and n.endswith('.npz')])==1134
 assert len([n for n in z.namelist() if n.startswith('splits/')])==9
with ZipFile(out/'report_sources.zip') as z:
 for n in ['report.tex','references.bib',*figures]:assert z.read(n)==(out/n).read_bytes()
log=(out/'report.log').read_text(errors='replace')
for bad in ['Overfull \\hbox','Overfull \\vbox','Missing character:','undefined references','Citation `']:assert bad not in log,bad
pdf=pymupdf.open(out/'report.pdf');bounds=[];fonts={}
for i,page in enumerate(pdf):
 box=page.cropbox
 for block in page.get_text('dict')['blocks']:
  for line in block.get('lines',[]):
   for span in line['spans']:
    b=pymupdf.Rect(span['bbox'])
    if b.x0<box.x0-1 or b.y0<box.y0-1 or b.x1>box.x1+1 or b.y1>box.y1+1:bounds.append([i+1,span['text'],list(b),list(box)])
 for font in page.get_fonts(full=True):fonts[font[0]]=font
assert not bounds,bounds[:10]
for xref in fonts:
 name,ext,kind,content=pdf.extract_font(xref)
 assert content,(xref,name,ext,kind)
verification={'passed':True,'counts':counts,'all1134_raw_rows_present_in_md_and_tex':True,'figures':17,'pages':len(pdf),'embedded_fonts':len(fonts),'text_outside_cropbox':len(bounds),'tex_missing_glyph_or_overfull_or_reference_errors':False,'all4_zip_integrity_checks':True,'source_bundle_assets_exact':True}
(out/'verification.json').write_text(json.dumps(verification,indent=2)+'\n')
sourcefiles=sorted((root/'experiments/new_models_rank').glob('*.py'))+[root/'experiments/new_models_rank/README.md',root/'README.md']
manifest={'base_commit':'bef9674','publication_base_commit':'2f4a48231882a377c603f233dc839b17c83793eb','publication_branch':'feat/ncnc-lpformer-refined-gae','native_commit':'95083da808e500c02da458e205f4b861ceb80871','files':{str(p.relative_to(root)):sha(p) for p in sourcefiles},'published_rank_scoring_sha256':sha(root/'experiments/node_rank_auc/evaluate.py')}
(out/'code_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(verification,indent=2))
