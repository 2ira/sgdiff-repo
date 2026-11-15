import numpy as np
import pytest
import torch

pytest.importorskip('torch_geometric')
pytest.importorskip('sklearn')
from sgdiff.config import load_config
from sgdiff.engine.benchmark import run_benchmark, BUDGET_PROFILES
from sgdiff.engine.course_baselines import CLASSICAL, PATH_MODELS, build_course_model, score_course_model
from sgdiff.data.graph import GraphData


def test_classical_scores_and_attribute_invariance():
    g=GraphData('toy',4,np.array([[0,2],[1,2],[0,3],[1,3]]),np.eye(4,dtype=np.float32))
    pairs=np.array([[0,1]])
    assert score_course_model(build_course_model('cn',4),g,pairs)[0]==2
    assert score_course_model(build_course_model('pa',4),g,pairs)[0]==4
    model=build_course_model('feature_cosine',4)
    np.testing.assert_array_equal(score_course_model(model,g,pairs),
        score_course_model(model,g.drop_edges([[0,2]]),pairs))


def test_registry_models_use_one_actual_query_set(tmp_path):
    torch.set_num_threads(2)
    cfg=load_config(None,BUDGET_PROFILES['smoke'])
    cfg.data.root=str(tmp_path/'data')
    result=run_benchmark(cfg,models=list(CLASSICAL+PATH_MODELS),seeds=[0],
        node_pair_epochs=1,device='cpu',write=False,verbose=False)
    assert set(result['summary'])==set(CLASSICAL+PATH_MODELS)
    assert len({r['split_fingerprint']for r in result['runs']})==1
    for row in result['runs']:
        assert np.isfinite(row['metrics']['auc'])


def test_path_context_is_rebuilt_after_missingness():
    g=GraphData('toy',4,np.array([[0,2],[2,1],[0,3],[3,1]]),np.eye(4,dtype=np.float32))
    model=build_course_model('sp4lp',4).eval()
    p=np.array([[0,1]])
    score_course_model(model,g,p)
    assert model.store.gather(p)[2][0]
    score_course_model(model,g.drop_edges([[0,2],[0,3]]),p)
    assert not model.store.gather(p)[2][0]
