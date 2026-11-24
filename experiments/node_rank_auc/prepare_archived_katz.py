"""Extract ONLY published Katz scores/splits and build audited manifests."""
import csv,hashlib,json
from pathlib import Path
from zipfile import ZipFile

ROOT=Path(__file__).resolve().parent

def main():
    archive=ROOT/'inputs/predictions-and-graphs.zip'
    out=ROOT/'inputs/published_katz'
    out.mkdir(exist_ok=True)
    with ZipFile(archive) as z:
        assert z.testzip() is None
        for name in z.namelist():
            if name.startswith('splits/') or (name.startswith('runs/') and '_katz' in name):
                path=Path(name)
                assert not path.is_absolute() and '..' not in path.parts
                target=out/path;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(z.read(name))
    with (ROOT/'inputs/published_per_seed.csv').open() as f: table=list(csv.DictReader(f))
    for dataset,n in [('cora',2708),('citeseer',3327)]:
        for seed in range(3):
            records=[]
            for model,variant in [('katz','full_h5'),('katz_h3','h3'),('katz_h2','h2')]:
                folder=out/'runs'/f'{dataset}_{model}_s{seed}'
                receipt=json.loads((folder/'receipt.json').read_text())
                assert receipt['dataset']==dataset and receipt['seed']==seed and receipt['model']==model
                for score in sorted(folder.glob('predictions_*.npz')):
                    condition=score.stem[len('predictions_'):]
                    matched=[r for r in table if r['dataset']==dataset and int(r['seed'])==seed and r['model']==model and r['condition']==condition]
                    assert len(matched)==1,(dataset,seed,model,condition)
                    row=matched[0]
                    records.append(dict(model='katz',variant=variant,condition=condition,
                        scores_path=str(score.relative_to(ROOT/'inputs')),pos_key='positive',neg_key='negative',
                        positive_sha256=receipt['positive'],negative_sha256=receipt['negative'],
                        query_order_evidence=f'Published archive {folder.name}/receipt.json; export_source.py -> pinned native.evaluate; no resampling',
                        expected_auc=float(row['auc']),graph_sha256=row['graph_sha256']))
            assert len(records)==42
            manifest=dict(dataset=dataset,seed=seed,num_nodes=n,split_path=f'published_katz/splits/{dataset}_seed{seed}.npz',
                protocol_commit='95083da808e500c02da458e205f4b861ceb80871',records=records)
            (ROOT/'inputs'/f'{dataset}_seed{seed}.json').write_text(json.dumps(manifest,indent=2)+'\n')
    provenance=dict(url='https://raw.githubusercontent.com/2ira/sgdiff-repo/2cd4a86a02874df461ce020cdbd17a62a7f33f6e/results/peer-protocol/predictions-and-graphs.zip',
        sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),models_used=['katz','katz_h3','katz_h2'],models_not_used=['historical sp4lp'])
    (ROOT/'inputs/archive_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Prepared six dataset-seed manifests, each with 42 archived Katz conditions.')

if __name__=='__main__':main()
