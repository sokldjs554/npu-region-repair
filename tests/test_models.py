import copy
import numpy as np
import pytest
import torch
from nrr.models import RegionCNN, make_candidate, forward_macs
from nrr.training import batch_stream, fit_budgeted, evaluate
from nrr.data import load_digits_split
from nrr.artifacts import state_digest
from nrr.budget import batch_charge

@pytest.fixture(autouse=True)
def threads():
    n=torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)

@pytest.mark.parametrize('kind',['continuation','operatorwise','regionwise','single_region','small_student'])
def test_candidate_shape_and_original_preserved(kind):
    torch.manual_seed(11); teacher=RegionCNN(width=8).eval(); digest=state_digest(teacher)
    c=make_candidate(teacher,kind,12)
    assert c(torch.rand(3,1,8,8)).shape==(3,10)
    assert c.features(torch.rand(3,1,8,8)).shape==teacher.features(torch.rand(3,1,8,8)).shape
    assert state_digest(teacher)==digest

def test_operator_substitution_preserves_convolution_and_removes_both_groups():
    t=RegionCNN(width=8).eval(); c=make_candidate(t,'operatorwise',1)
    assert torch.equal(t.r0.conv.weight,c.r0.conv.weight)
    assert not any(isinstance(m,(torch.nn.GroupNorm,torch.nn.GELU)) for m in c.modules())

def test_region_replacement_is_smaller_and_named_scope_is_explicit():
    t=RegionCNN(width=8).eval(); c=make_candidate(t,'regionwise',1)
    assert sum(p.numel() for p in c.parameters())<sum(p.numel() for p in t.parameters())
    assert forward_macs(c,(1,8,8))<forward_macs(t,(1,8,8))
    with pytest.raises(ValueError): make_candidate(t,'magic',0)

def test_recovery_is_real_has_shared_budget_and_no_test_selection():
    d=load_digits_split(17); t=RegionCNN(width=8).eval(); th=state_digest(t)
    c=make_candidate(t,'operatorwise',2); before=state_digest(c)
    stream=batch_stream(d.train,3,32,8)
    charge=batch_charge(forward_macs(c,(1,8,8)),forward_macs(t,(1,8,8)),8)
    stats=fit_budgeted(c,t,d,stream,charge*3,seed=3,lr=.02)
    assert stats['updates']==3 and stats['images']==24 and stats['teacher_images']==24
    assert stats['budget_spent']==stats['budget_cap']
    assert stats['last_state_sha256']!=before and state_digest(t)==th
    assert all(p['selection_split']=='validation' for p in stats['checkpoints'])
    assert stats['test_accesses_during_training']==0

def test_budget_too_small_fails_before_training():
    d=load_digits_split(); t=RegionCNN(width=8).eval(); c=copy.deepcopy(t)
    with pytest.raises(ValueError,match='one batch'):
        fit_budgeted(c,t,d,batch_stream(d.train,1,4,8),1,seed=1)

def test_batch_stream_uses_train_only_and_reproduces():
    ids=np.arange(11)
    a=batch_stream(ids,3,10,8); b=batch_stream(ids,3,10,8)
    assert a.shape==(10,8) and np.array_equal(a,b) and set(a.flat)==set(ids)

def test_eval_has_exact_denominator():
    d=load_digits_split(); t=RegionCNN(width=8).eval()
    r=evaluate(t,d,d.val[:7]); assert r['n']==7 and len(r['predictions'])==7
