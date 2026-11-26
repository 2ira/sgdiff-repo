import json
import itertools
import numpy as np
import pytest
from evaluate import auc, digest, edge_array, masks, ranks, run


def test_auc_hand_calculated_and_ties():
    assert auc([3,1],[2,1]) == .625
    assert auc([0,0],[0,0,0]) == .5
    assert auc([3,4],[1,2]) == 1
    assert auc([1],[2]) == 0
    assert auc([], [1]) is None
    assert auc([1], []) is None
    with pytest.raises(ValueError): auc([np.nan],[1])


def test_or_masks_not_and_and_no_double_counting():
    position=np.array([1,2,3,4])
    pairs=np.array([[0,1],[0,3],[2,3]])
    ms=masks(position,pairs,[0,25,50,75,100])
    assert ms[25].tolist()==[True,True,False]
    assert ms[50].sum()==2
    assert not ms[0].any() and ms[100].all()
    for k,j in zip([0,25,50,75],[25,50,75,100]):
        assert np.all(ms[k]<=ms[j])


def test_node_ranking_is_input_order_invariant():
    e=np.array([[0,1],[0,2],[0,3],[3,4]])
    a,t1=ranks(6,e,'tiny',0); b,t2=ranks(6,e[::-1,::-1],'tiny',0)
    assert t1==t2
    for name in a:
        np.testing.assert_allclose(a[name][0],b[name][0],rtol=0,atol=1e-15)
        np.testing.assert_array_equal(a[name][1],b[name][1])
    assert a['degree'][1][0]==0
    assert a['degree'][0][5]==0  # Isolates still in node universe.


def test_duplicate_undirected_edges_rejected():
    with pytest.raises(ValueError): edge_array(np.array([[0,1],[1,0]]),3,'train')


def fixture_manifest(tmp_path):
    train=np.array([[0,1],[0,2],[0,3],[3,4]])
    pos=np.array([[1,2],[1,3]])
    neg=np.array([[2,4],[1,4]])
    np.savez(tmp_path/'split.npz',train_pos=train,test_pos=pos,test_neg=neg)
    records=[]
    for cond,pp,nn in [('clean',[.8,.6],[.7,.1]),('attack',[.7,.3],[.8,.1])]:
        np.savez(tmp_path/f'{cond}.npz',positive=pp,negative=nn)
        records.append(dict(model='toy',variant='full',condition=cond,scores_path=f'{cond}.npz',
            positive_sha256=digest(pos),negative_sha256=digest(neg),query_order_evidence='test fixture ordered IDs',
            expected_auc=.75 if cond=='clean' else .5))
    manifest=dict(dataset='tiny',seed=0,num_nodes=5,split_path='split.npz',records=records)
    p=tmp_path/'manifest.json';p.write_text(json.dumps(manifest))
    return p


def test_full_pipeline_boundaries_and_frozen_masks(tmp_path):
    p=fixture_manifest(tmp_path)
    rows=run(p,tmp_path/'out')
    assert len(rows)==2*2*101
    assert all(r['auc'] is None for r in rows if r['k']==0)
    assert {r['auc'] for r in rows if r['k']==100}=={.75,.5}
    for rank,k in itertools.product(['degree','pagerank'],range(101)):
        selected=[r for r in rows if r['rank']==rank and r['k']==k]
        assert selected[0]['n_pos']==selected[1]['n_pos']
        assert selected[0]['n_neg']==selected[1]['n_neg']
    with pytest.raises(FileExistsError):run(p,tmp_path/'out')


def test_query_reordering_is_rejected(tmp_path):
    p=fixture_manifest(tmp_path)
    with np.load(tmp_path/'split.npz') as f: d={k:f[k] for k in f.files}
    d['test_pos']=d['test_pos'][::-1]
    np.savez(tmp_path/'split.npz',**d)
    with pytest.raises(ValueError,match='fingerprints'):run(p,tmp_path/'out')
