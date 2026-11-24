"""After native evaluation completes, verify NAS weights and export only portable small artifacts."""
import argparse,hashlib,json,shutil,subprocess,sys,time
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def main(a):
 b=a.workspace.resolve();deadline=time.monotonic()+7200
 while True:
  try:
   audit=json.loads((b/'evaluation/audit.json').read_text())
   if audit['native_evaluator_cells']==378:break
  except (FileNotFoundError,json.JSONDecodeError):pass
  if time.monotonic()>deadline:raise TimeoutError('Native evaluation did not finish')
  time.sleep(5)
 subprocess.run([sys.executable,str(b/'code/scripts/audit_hlgnn_artifacts.py'),'--training',str(b/'training'),'--evaluation',str(b/'evaluation'),'--historical',str(b/'code/results/hlgnn-robustness'),'--output',str(b/'artifact_audit.json')],check=True)
 if (b/'pilot.log').exists():shutil.copy2(b/'pilot.log',b/'training/logs/cora_readme_exact_full_s0.initial.log')
 code={str(p.relative_to(b/'code')):sha(p) for p in (b/'code').rglob('*.py')}
 (b/'code_manifest.json').write_text(json.dumps(code,indent=2)+'\n')
 import torch
 hardware=dict(torch_cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),interpreter=sys.executable)
 (b/'hardware.json').write_text(json.dumps(hardware,indent=2)+'\n')
 files=[]
 for directory in ['training','evaluation']:
  for p in (b/directory).rglob('*'):
   if not p.is_file():continue
   if directory=='evaluation' and 'attacks' in p.relative_to(b/directory).parts:continue
   if p.suffix in ['.json','.log','.csv','.npz'] or p.name=='TRAINING_COMPLETE':files.append(p)
 files.extend(b/name for name in ['artifact_audit.json','code_manifest.json','hardware.json','data_receipt.json'])
 archive=b/'hlgnn_results_20261124.zip'
 with ZipFile(archive,'w',ZIP_DEFLATED,compresslevel=6) as z:
  for p in sorted(files):z.write(p,str(p.relative_to(b)))
 with ZipFile(archive) as z:assert z.testzip() is None
 receipt=dict(file=archive.name,bytes=archive.stat().st_size,sha256=sha(archive),files=len(files),checkpoints_in_nas=True,weights_in_archive=False)
 (b/'export_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
 print('EXPORT_COMPLETE',json.dumps(receipt),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--workspace',type=Path,required=True);main(p.parse_args())
