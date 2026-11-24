"""Adapt published PubMed arrays; verify pair identities before slicing scores."""
import csv,hashlib,io,json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
from evaluate import digest
ROOT=Path(__file__).resolve().parent

def main():
    with (ROOT/'inputs/pubmed_published_per_seed.csv').open() as f:table=list(csv.DictReader(f))
    provenance=[]
    for seed in range(3):
        archive=ROOT/'inputs'/f'pubmed_seed{seed}.zip'
        dest=ROOT/'inputs'/'published_pubmed'/f'seed{seed}';dest.mkdir(parents=True,exist_ok=True)
        records=[]
        with ZipFile(archive) as z:
            assert z.testzip() is None
            payload=z.read(f'splits/pubmed_seed{seed}.npz');(dest/'split.npz').write_bytes(payload)
            with np.load(io.BytesIO(payload),allow_pickle=False) as f:tp,tn=f['test_pos'],f['test_neg']
            with np.load(io.BytesIO(z.read(f'pubmed_s{seed}/queries.npz')),allow_pickle=False) as f:
                assert int(f['positive_count'])==len(tp)
                np.testing.assert_array_equal(f['pairs'],np.vstack([tp,tn]))
            for member in z.namelist():
                if not member.startswith(f'pubmed_s{seed}/scores_'):continue
                condition=Path(member).stem[len('scores_'):]
                with np.load(io.BytesIO(z.read(member)),allow_pickle=False) as f:
                    for model,variant in [('katz','full_h5'),('katz_h3','h3'),('katz_h2','h2')]:
                        scores=f[model];assert scores.shape==(len(tp)+len(tn),)
                        target=dest/f'{model}_{condition}.npz'
                        np.savez_compressed(target,positive=scores[:len(tp)],negative=scores[len(tp):],test_pos=tp,test_neg=tn)
                        rows=[r for r in table if int(r['seed'])==seed and r['model']==model and r['condition']==condition];assert len(rows)==1
                        row=rows[0]
                        records.append(dict(model='katz',variant=variant,condition=condition,scores_path=str(target.relative_to(ROOT/'inputs')),
                            positive_sha256=digest(tp),negative_sha256=digest(tn),query_order_evidence='Explicit published queries.npz pairs == vstack(test_pos,test_neg); embedded IDs retained',
                            expected_auc=float(row['auc']),graph_sha256=row['graph_sha256']))
        assert len(records)==42
        m=dict(dataset='pubmed',seed=seed,num_nodes=19717,split_path=str((dest/'split.npz').relative_to(ROOT/'inputs')),
            protocol_commit='95083da808e500c02da458e205f4b861ceb80871',records=records)
        (ROOT/'inputs'/f'pubmed_seed{seed}.json').write_text(json.dumps(m,indent=2)+'\n')
        provenance.append(dict(seed=seed,archive=str(archive.name),sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
            url=f'https://raw.githubusercontent.com/2ira/sgdiff-repo/2cd4a86a02874df461ce020cdbd17a62a7f33f6e/results/peer-protocol/pubmed/pubmed_seed{seed}.zip'))
    (ROOT/'inputs/pubmed_archive_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print('Prepared three PubMed manifests with verified explicit query IDs.')
if __name__=='__main__':main()
