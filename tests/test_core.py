from pathlib import Path
import numpy as np
import pytest
from nrr.artifacts import reserve_output, write_json, digest_file
from nrr.budget import Budget, batch_charge
from nrr.data import load_digits_split, Dataset, load_npz


def test_budget_exact_cap_and_rejection_does_not_mutate():
    b = Budget(100)
    b.charge(60)
    assert not b.can_charge(41)
    with pytest.raises(ValueError): b.charge(41)
    assert b.spent == 60
    b.charge(40)
    assert b.remaining == 0

@pytest.mark.parametrize('value',[0,-1,True,1.5])
def test_budget_invalid(value):
    with pytest.raises(ValueError): Budget(value)

def test_charge_includes_teacher_and_backward_proxy():
    assert batch_charge(student_macs=100, teacher_macs=200, images=3) == 1500

def test_output_refuses_nonempty_and_symlink(tmp_path):
    p = tmp_path/'run'
    reserve_output(p)
    (p/'keep').write_text('do not change')
    with pytest.raises(FileExistsError): reserve_output(p)
    link = tmp_path/'link'; link.symlink_to(p, target_is_directory=True)
    with pytest.raises(FileExistsError): reserve_output(link)

def test_json_is_atomic_finite_and_hashable(tmp_path):
    p=tmp_path/'x.json'; write_json(p,{'a':1})
    before=digest_file(p)
    with pytest.raises(ValueError): write_json(p,{'a':float('nan')})
    assert digest_file(p)==before

def test_split_reproducible_disjoint_and_not_official_benchmark():
    a=load_digits_split(17); b=load_digits_split(17)
    assert len(a.x)==1797 and a.x.shape[1:]==(1,8,8)
    assert np.array_equal(a.train,b.train)
    assert not (set(a.train)&set(a.val) or set(a.train)&set(a.test) or set(a.val)&set(a.test))
    assert len(a.train)+len(a.val)+len(a.test)==1797
    assert 'mechanism' in a.scope
    for ix in (a.train,a.val,a.test): assert len(np.unique(a.y[ix]))==10

def test_duplicate_split_ids_rejected():
    with pytest.raises(ValueError, match='overlap'):
        Dataset(np.zeros((3,1,8,8),np.float32),np.array([0,1,0]),np.array([0]),np.array([0]),np.array([2]),'fixture')

def test_npz_roundtrip_and_object_arrays_rejected(tmp_path):
    d=load_digits_split(2)
    p=tmp_path/'data.npz'
    np.savez(p,x=d.x,y=d.y,train=d.train,val=d.val,test=d.test)
    other=load_npz(p)
    assert other.source.endswith('data.npz')
    assert np.array_equal(other.x,d.x)
    np.savez(tmp_path/'bad.npz',x=np.array([object()],dtype=object))
    with pytest.raises(ValueError): load_npz(tmp_path/'bad.npz')

def test_npz_integer_labels_are_normalized_for_cross_entropy(tmp_path):
    d=load_digits_split(17)
    p=tmp_path/'int32_labels.npz'
    np.savez(p,x=d.x,y=d.y.astype(np.int32),train=d.train,val=d.val,test=d.test)
    loaded=load_npz(p)
    assert loaded.y.dtype==np.int64
