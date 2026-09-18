"""Actual CPU PTQ, deliberately not an NPU support oracle.

Pinned legacy FX adapter for locally available PyTorch 2.10. New quantization
API development has moved to torchao; this adapter is not a future API promise.
"""
from __future__ import annotations
import copy
import warnings
import torch
from torch.ao.quantization import get_default_qconfig_mapping
from torch.ao.quantization.quantize_fx import prepare_fx,convert_fx
from .artifacts import state_digest


def quantize_cpu(model: torch.nn.Module,calibration: torch.Tensor):
    if calibration.ndim!=4 or not len(calibration) or not torch.isfinite(calibration).all():
        raise ValueError('Expected nonempty finite NCHW calibration tensor')
    if 'x86' not in torch.backends.quantized.supported_engines:
        raise RuntimeError('This CPU adapter requires the x86 quantization engine')
    original_hash=state_digest(model);old=torch.backends.quantized.engine
    caught=[]
    try:
        torch.backends.quantized.engine='x86'
        with warnings.catch_warnings(record=True) as ws:
            warnings.simplefilter('always')
            prepared=prepare_fx(copy.deepcopy(model).eval(),get_default_qconfig_mapping('x86'),(calibration[:1],))
            with torch.no_grad():
                for chunk in calibration.split(32):prepared(chunk)
            quantized=convert_fx(prepared).eval()
            with torch.no_grad():out=quantized(calibration[:1])
            caught=sorted({str(w.message) for w in ws})
        modules=[{'name':name,'class':type(m).__module__+'.'+type(m).__name__}
                 for name,m in quantized.named_modules() if 'quantized' in type(m).__module__]
        convs=[m for m in modules if 'Conv' in m['class'] or 'Linear' in m['class'] and 'Packed' not in m['class']]
        if not convs:raise RuntimeError('No actual quantized conv/linear modules produced')
        if not torch.isfinite(out).all():raise RuntimeError('Nonfinite quantized result')
    finally:
        torch.backends.quantized.engine=old
    if state_digest(model)!=original_hash:raise RuntimeError('PTQ mutated the FP32 source')
    return quantized,{'target':'pytorch_x86_cpu','npu_execution':False,'calibration_images':len(calibration),
                     'quantized_conv_or_linear_modules':len(convs),'quantized_modules':modules,
                     'graph':str(quantized.graph),'warnings':caught,
                     'scope':'Real quantized CPU modules; remaining operations may be floating point. Not an all-integer NPU claim.'}
