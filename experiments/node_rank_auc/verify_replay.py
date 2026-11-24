"""Compare CPU replay against every released per-seed curve, delta and ranking file."""
import argparse,hashlib,json
from pathlib import Path
from zipfile import ZipFile

def main(a):
 checks=[]
 with ZipFile(a.published/'per_seed_curves.zip') as z:
  for name in sorted(z.namelist()):
   if not name.endswith('.csv'):continue
   expected=z.read(name);actual=(a.replay_output/name).read_bytes()
   if actual!=expected:raise ValueError(f'Replay differs: {name}')
   checks.append(dict(file=str(Path(name).relative_to('curves')),sha256=hashlib.sha256(actual).hexdigest()))
 assert len(checks)==54
 a.output.write_text(json.dumps(dict(archived_score_evaluations=756,matched_files=54,byte_identical=True,checks=checks),indent=2)+'\n')
 print('All 54 published curve/delta/rank files match CPU replay byte-for-byte.')
if __name__=='__main__':
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--published',type=Path,default=Path(__file__).resolve().parents[2]/'results/node-rank-auc')
 p.add_argument('--replay-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True);main(p.parse_args())
