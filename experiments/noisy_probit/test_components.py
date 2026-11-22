"""Checks for the exploratory components, independent of fitted outcomes."""
from dataclasses import replace
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from .audit_design import stratified_mask
from .full_channel import FullChannel
from .experiment import simulate


def test_stratified_selection_budget_and_support():
    observed=np.array([[0,0]]*35+[[0,1]]*2+[[1,0]]*15+[[1,1]]*10)
    mask=stratified_mask(observed,20,91)
    assert mask.sum()==20
    assert np.array_equal(mask,stratified_mask(observed,20,91))
    counts=np.bincount((2*observed[:,0]+observed[:,1])[mask],minlength=4)
    assert counts.tolist()==[6,2,6,6]


def test_full_channel_gradient_and_no_truth_leakage():
    data=simulate(n=15,seed=91)
    p=FullChannel(data);z=np.r_[[1,-.5,2,.5],np.arange(12)*.03]
    analytic=p.value_grad(z)[1]
    numeric=approx_derivative(lambda v:p.value_grad(v)[0],z,method='3-point').ravel()
    assert np.allclose(analytic,numeric,atol=2e-5,rtol=2e-5)
    _,_,channel,_=p.unpack(z)
    assert np.allclose(channel.sum(1),1)
    truth=data.truth.copy();truth[~data.audit]=1-truth[~data.audit]
    other=FullChannel(replace(data,truth=truth))
    assert p.value_grad(z)[0]==other.value_grad(z)[0]
    assert np.array_equal(p.value_grad(z)[1],other.value_grad(z)[1])
