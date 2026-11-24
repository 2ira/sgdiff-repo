"""Replay released Katz and HL-GNN scores with the same node-rank AUROC implementation."""
import argparse,json,shutil
from pathlib import Path
from zipfile import ZipFile
from evaluate import run
from prepare_hlgnn import prepare

def extract(archive,directory):
 with ZipFile(archive) as z:
  for name in z.namelist():
   p=Path(name)
   if p.is_absolute() or '..' in p.parts:raise ValueError(name)
  z.extractall(directory)

def main(a):
 if a.output.exists() and any(a.output.iterdir()):raise FileExistsError('Choose a new output directory')
 a.output.mkdir(parents=True,exist_ok=True)
 extract(a.published/'katz-replay-inputs.zip',a.output/'katz')
 extract(a.published.parent/'hlgnn-retrained'/'predictions.zip',a.output/'hlgnn')
 manifests=prepare(a.output/'hlgnn/evaluation',a.output/'hlgnn/manifests',a.historical)
 for m in sorted((a.output/'katz/inputs').glob('*_seed*.json')):
  run(m,a.output/'curves'/f'katz_{m.stem}')
 for m in manifests:
  run(m,a.output/'curves'/f'hlgnn_{Path(m).stem}')
 print('All published predictions replayed. Each k=100 value verified against its native AUROC.')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 repo=Path(__file__).resolve().parents[2]
 p.add_argument('--published',type=Path,default=repo/'results/node-rank-auc')
 p.add_argument('--historical',type=Path,default=repo/'results/hlgnn-historical/per_seed.csv')
 p.add_argument('--output',type=Path,required=True)
 main(p.parse_args())
