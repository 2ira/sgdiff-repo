"""CPU-only score replay: no torch/model dependencies, no new graph or model."""
import argparse,csv,hashlib,json
from pathlib import Path
from zipfile import ZipFile
from experiments.new_models_rank.frozen_rank import run
from experiments.new_models_rank.summarize import read,dump

def main(a):
 a.output.mkdir(parents=True,exist_ok=True)
 if any(a.output.iterdir()):raise FileExistsError('Use an empty replay directory')
 with ZipFile(a.predictions) as z:
  for name in z.namelist():
   path=(a.output/'inputs'/name).resolve()
   if not path.is_relative_to((a.output/'inputs').resolve()):raise ValueError('Invalid archive path')
   path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(z.read(name))
 origin=json.loads((a.output/'inputs/origin.json').read_text());assert origin['evaluations']==1134
 assert origin['rank_archive_sha256']==hashlib.sha256(a.rank_archive.read_bytes()).hexdigest()
 count=0
 for m in sorted((a.output/'inputs/manifests').glob('*_seed*.json')):
  manifest=json.loads(m.read_text())
  for record in manifest['records']:
   assert record['scores_file_sha256']==hashlib.sha256((m.parent/record['scores_path']).read_bytes()).hexdigest()
  run(m,a.output/'curves'/m.stem,a.rank_archive);count+=1
 assert count==9
 # Compare all curve values, masks and undefined points with the released tables.
 checked=0;max_error=0.
 with ZipFile(a.expected) as z:
  for m in sorted((a.output/'curves').glob('*_seed*')):
   import io
   expected=list(csv.DictReader(io.StringIO(z.read(f'curves/{m.name}/per_seed_curves.csv').decode())))
   actual=read(m/'per_seed_curves.csv');assert len(actual)==len(expected)==25452
   for x,y in zip(actual,expected):
    for k in ['dataset','seed','model','variant','condition','rank','k','n_selected_nodes','n_pos','n_neg','low_support','undefined_reason','test_pos_sha256','test_neg_sha256']:assert x[k]==y[k],(m.name,k,x[k],y[k])
    if x['auc'] or y['auc']:
     assert x['auc'] and y['auc'];error=abs(float(x['auc'])-float(y['auc']));assert error<1e-12;max_error=max(max_error,error)
    checked+=1
 audit=dict(passed=True,original_predictions=1134,curve_rows_checked=checked,rankings_byte_identical=True,k0_undefined=True,k100_original_auc=True,max_curve_auc_error=max_error,model_inference=False,gpu_required=False)
 dump(a.output/'replay_audit.json',audit);print(json.dumps(audit,indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 for x in ['predictions','rank-archive','expected','output']:p.add_argument('--'+x,type=Path,required=True)
 main(p.parse_args())
