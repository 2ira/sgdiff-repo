import numpy as np
import pytest
from .analysis import compare,summarize


def records():
    ds=[]
    for i,e in enumerate(np.linspace(-.2,.2,40)):
        results={}
        for method,scale in [('a',2),('b',1)]:
            results[method]=dict(success=True,estimate=[0,0,0,scale*e],lower=[-1]*4,upper=[1]*4,seconds=1.)
        ds.append(dict(seed=i,audit_fraction=.05,truth=[0]*4,results=results))
    return ds


def test_bootstrap_preserves_pairing():
    ds=records();r=compare(ds,'a',ds,'b')
    assert np.allclose(r['rmse_ratio_ci95'],[2,2])
    with pytest.raises(AssertionError):compare(ds,'a',list(reversed(ds)),'b')


def test_failed_intervals_are_not_counted_as_covered():
    ds=records();ds[0]['results']['a']['success']=False
    assert all(np.isfinite(r['rmse_mcse']) for r in summarize(ds,'a'))
    r=summarize(ds,'a')[3]
    assert r['failures']==1
    assert r['coverage']==39/40
    assert r['coverage_conditional']==1
    assert np.isclose(r['rmse'],np.sqrt(np.mean([(2*e)**2 for e in np.linspace(-.2,.2,40)])))
