import importlib.util
import inspect
import numpy as np
import pytest
import nrr.tflite_bridge as b
from nrr.compiler import CompilerUnavailable

def test_saved_probe_can_supply_validation_images_separately_from_calibration():
    assert 'parity_images' in inspect.signature(b.convert_tflite).parameters
    if importlib.util.find_spec('tensorflow') is None:
        with pytest.raises(CompilerUnavailable):
            b.convert_tflite(None,np.zeros((1,1,8,8),np.float32),'unused',parity_images=np.zeros((1,1,8,8),np.float32))

@pytest.mark.parametrize('wrong',[np.array([[np.nan]]),np.array([[np.inf]]),np.zeros((2,1))])
def test_parity_check_rejects_nonfinite_or_broadcast_shapes(wrong):
    with pytest.raises(ValueError): b.parity_error(np.zeros((1,1)),wrong)

def test_parity_reports_actual_max_absolute_difference():
    assert b.parity_error(np.array([[0.,1.]]),np.array([[.2,1.1]]))==pytest.approx(.2)

@pytest.mark.skipif(importlib.util.find_spec('tensorflow') is None,reason='TensorFlow absent: float TFLite parity not executed')
def test_real_optional_bridge_checks_float_tflite_before_quantized_export(tmp_path):
    from nrr.models import RegionCNN,make_candidate
    import torch
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        model=make_candidate(RegionCNN(channels=3,width=8).eval(),'single_region',3)
        cal=np.random.default_rng(1).random((16,3,12,12),dtype=np.float32)
        val=np.random.default_rng(2).random((5,3,12,12),dtype=np.float32)
        r=b.convert_tflite(model,cal,tmp_path/'bridge',parity_images=val)
        assert r['float_tflite_max_abs']<1e-4
        assert r['parity_images']==5 and r['calibration_images']==16
        assert r['npu_execution'] is False
    finally:torch.set_num_threads(previous)
