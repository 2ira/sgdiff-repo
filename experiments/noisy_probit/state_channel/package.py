"""Package the reproducible study without unrelated workspace data."""
import hashlib,json,zipfile
from pathlib import Path


def main():
    root=Path.cwd();out=Path('deliverables/noisy_probit_state_channel_english')
    files=set()
    for base in [Path('experiments/noisy_probit'),Path('results/noisy_probit/state_channel/validation'),
                 Path('results/noisy_probit/state_channel/development'),Path('results/noisy_probit/state_channel/mcmc'),out]:
        for p in base.rglob('*'):
            if p.is_file() and '__pycache__' not in p.parts and p.suffix not in ['.pyc','.aux','.nav','.snm','.out','.toc'] and p.name!='MANIFEST.json':files.add(p)
    parent=Path('experiments/__init__.py')
    if parent.exists():files.add(parent)
    old=Path('results/noisy_probit/lowdim/validation')
    files.update((old/'state_dependent/q0.05').glob('seed*.json'))
    files.add(old/'LOCK.json')
    manifest={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)}
    mf=out/'MANIFEST.json';mf.write_text(json.dumps(manifest,indent=2)+'\n');files.add(mf)
    dest=out.parent/'noisy_probit_state_channel_reproducible.zip'
    with zipfile.ZipFile(dest,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,p.as_posix())
    with zipfile.ZipFile(dest) as z:
        assert z.testzip() is None
        for name,digest in manifest.items():assert hashlib.sha256(z.read(name)).hexdigest()==digest
    print(json.dumps(dict(bundle=str(dest.resolve()),files=len(files),bytes=dest.stat().st_size,sha256=hashlib.sha256(dest.read_bytes()).hexdigest()),indent=2))


if __name__=='__main__':main()
