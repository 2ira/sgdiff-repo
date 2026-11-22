"""Publish compact records and restore them safely for exact re-analysis."""
import argparse,gzip,json,zipfile
from pathlib import Path
from .analysis import load,ROOT
from ..run import write_json

BASE=ROOT.parent


def main():
    p=argparse.ArgumentParser();p.add_argument('operation',choices=['pack','extract']);a=p.parse_args()
    records=BASE/'fit_records.jsonl.gz';draws=BASE/'posterior_draws.zip'
    if a.operation=='pack':
        data,lock=load();count=0
        with gzip.open(records,'wt',encoding='utf-8',compresslevel=9) as f:
            for key in sorted(data):
                for d in data[key]:f.write(json.dumps(d,sort_keys=True,separators=(',',':'),allow_nan=False)+'\n');count+=1
        with zipfile.ZipFile(draws,'w',compression=zipfile.ZIP_STORED) as z:
            for file in sorted((BASE/'mcmc').glob('*/draws.npz')):z.write(file,file.relative_to(BASE))
        with zipfile.ZipFile(draws) as z:assert z.testzip() is None
        print(f'Packed {count} fit records, {records.stat().st_size} bytes; posterior archive {draws.stat().st_size} bytes.')
    else:
        design=json.loads((ROOT/'LOCK.json').read_text())['design'];seen=set()
        with gzip.open(records,'rt',encoding='utf-8') as f:
            for line in f:
                d=json.loads(line);case=d['case'];q=d['audit_fraction'];seed=d['seed']
                assert case in design['cases'] and q in design['budgets'] and isinstance(seed,int)
                start=design['cases'][case]['seed_start'];assert start<=seed<start+design['replicates']
                key=(case,q,seed);assert key not in seen;seen.add(key)
                out=ROOT/case/f'q{q:g}'/f'seed{seed}.json'
                if out.exists():assert json.loads(out.read_text())==d
                else:write_json(out,d)
        assert len(seen)==len(design['cases'])*len(design['budgets'])*design['replicates']
        with zipfile.ZipFile(draws) as z:
            for member in z.namelist():
                out=BASE/member;assert out.resolve().is_relative_to(BASE.resolve()) and member.startswith('mcmc/') and member.endswith('/draws.npz')
                value=z.read(member)
                if out.exists():assert out.read_bytes()==value
                else:out.parent.mkdir(parents=True,exist_ok=True);out.write_bytes(value)
        load();print(f'Restored and verified {len(seen)} fit records and posterior draws.')


if __name__=='__main__':main()
