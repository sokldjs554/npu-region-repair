"""Optional, numerically checked PyTorch→TensorFlow bridge for our fixed CNN.

Not a general PyTorch converter. Tested only when the explicit optional
integration test runs; absent TensorFlow is not a passing test.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from .artifacts import digest_file,write_json
from .compiler import CompilerUnavailable


def parity_error(expected: np.ndarray, actual: np.ndarray) -> float:
    expected, actual = np.asarray(expected), np.asarray(actual)
    if expected.shape != actual.shape or not expected.size:
        raise ValueError('Parity tensors must have identical nonempty shapes')
    if not np.isfinite(expected).all() or not np.isfinite(actual).all():
        raise ValueError('Parity tensors must be finite')
    return float(np.max(np.abs(expected.astype(np.float64)-actual.astype(np.float64))))


def convert_tflite(model,calibration: np.ndarray,out: str | Path,parity_images: np.ndarray | None=None)->dict:
    try:import tensorflow as tf
    except ImportError as exc:raise CompilerUnavailable('TensorFlow export is unavailable') from exc
    from torch import nn
    model.eval();out=Path(out);out.mkdir(parents=True,exist_ok=False)
    if calibration.ndim!=4 or not len(calibration) or not np.isfinite(calibration).all():raise ValueError('Invalid calibration images')
    shape=calibration.shape[1:]
    if calibration.dtype != np.float32 or shape[1]!=shape[2] or shape[1]<8 or shape[1]%4:
        raise ValueError('Bridge requires square float32 inputs divisible by four')
    parity = calibration[:16] if parity_images is None else np.asarray(parity_images)
    if parity.ndim!=4 or not len(parity) or tuple(parity.shape[1:])!=tuple(shape) or not np.isfinite(parity).all():
        raise ValueError('Invalid separate parity inputs')
    def arr(t):return t.detach().cpu().numpy().astype(np.float32)
    def layer(x,m,name):
        with tf.name_scope(name):
            if isinstance(m,nn.Sequential):
                for key,child in m.named_children():x=layer(x,child,key)
                return x
            if isinstance(m,nn.Conv2d):
                if m.groups!=1 or m.stride!=(1,1):raise ValueError('Bridge only supports ungrouped stride-1 convolution')
                w=tf.constant(arr(m.weight).transpose(2,3,1,0))
                x=tf.nn.conv2d(x,w,strides=1,padding='SAME' if m.padding!=(0,0) else 'VALID')
                return tf.nn.bias_add(x,tf.constant(arr(m.bias))) if m.bias is not None else x
            if isinstance(m,nn.GroupNorm):
                if m.num_groups!=1:raise ValueError('Bridge only supports GroupNorm with one group')
                mean,var=tf.nn.moments(x,axes=[1,2,3],keepdims=True)
                z=(x-mean)*tf.math.rsqrt(var+m.eps)
                return z*tf.constant(arr(m.weight))[None,None,None,:]+tf.constant(arr(m.bias))[None,None,None,:]
            if isinstance(m,nn.BatchNorm2d):
                return tf.nn.batch_normalization(x,tf.constant(arr(m.running_mean)),tf.constant(arr(m.running_var)),
                       tf.constant(arr(m.bias)),tf.constant(arr(m.weight)),m.eps)
            if isinstance(m,nn.ReLU):return tf.nn.relu(x)
            if isinstance(m,nn.GELU):return tf.nn.gelu(x,approximate=False)
            raise ValueError(f'Unsupported bridge module {type(m).__name__}')
    class Bridge(tf.Module):
        @tf.function(input_signature=[tf.TensorSpec([1,shape[1],shape[2],shape[0]],tf.float32,name='image')])
        def __call__(self,x):
            x=layer(x,model.stem,'stem');x=layer(x,model.r0,'r0');x=layer(x,model.r1,'r1')
            x=tf.nn.avg_pool2d(x,ksize=shape[1]//4,strides=shape[1]//4,padding='VALID')
            x=tf.reshape(tf.transpose(x,[0,3,1,2]),[1,-1])
            return {'logits':tf.matmul(x,tf.constant(arr(model.head.weight).T))+tf.constant(arr(model.head.bias))}
    bridge=Bridge();maximum=0.
    for image in parity:
        with torch.no_grad():expected=model(torch.from_numpy(image[None])).numpy()
        got=bridge(tf.constant(image[None].transpose(0,2,3,1)))['logits'].numpy()
        maximum=max(maximum,parity_error(expected,got))
    if maximum>1e-4:raise RuntimeError(f'Bridge parity failed: max abs {maximum}')
    concrete=bridge.__call__.get_concrete_function()
    float_converter=tf.lite.TFLiteConverter.from_concrete_functions([concrete],bridge)
    float_path=out/'model.float.tflite'
    float_path.write_bytes(float_converter.convert())
    float_runtime=tf.lite.Interpreter(model_path=str(float_path),num_threads=1,
               experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    float_runtime.allocate_tensors()
    float_in=float_runtime.get_input_details()[0];float_out=float_runtime.get_output_details()[0]
    float_maximum=0.
    for image in parity:
        with torch.no_grad():expected=model(torch.from_numpy(image[None])).numpy()
        float_runtime.set_tensor(float_in['index'],image[None].transpose(0,2,3,1).astype(np.float32))
        float_runtime.invoke()
        float_maximum=max(float_maximum,parity_error(expected,float_runtime.get_tensor(float_out['index'])))
    if float_maximum>1e-4:raise RuntimeError(f'Float TFLite conversion parity failed: max abs {float_maximum}')
    converter=tf.lite.TFLiteConverter.from_concrete_functions([concrete],bridge)
    converter.optimizations=[tf.lite.Optimize.DEFAULT]
    converter.representative_dataset=lambda:([image[None].transpose(0,2,3,1)] for image in calibration)
    converter.target_spec.supported_ops=[tf.lite.OpsSet.TFLITE_BUILTINS_INT8,tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.inference_input_type=tf.int8;converter.inference_output_type=tf.int8
    flat=converter.convert();path=out/'model.tflite';path.write_bytes(flat)
    metadata={'path':str(path),'sha256':digest_file(path),'float_bridge_max_abs':maximum,
              'calibration_images':len(calibration),'tensorflow_version':tf.__version__,
              'parity_images':len(parity),'parity_scope':'separate caller inputs' if parity_images is not None else 'calibration subset',
              'float_tflite_max_abs':float_maximum,'float_tflite_sha256':digest_file(float_path),'npu_execution':False,
              'all_ops_integer':None,'scope':'INT8 boundary model; inspect internal types. Conversion is not NPU execution.'}
    write_json(out/'export.json',metadata)
    return metadata


def evaluate_tflite(path: str | Path,x: np.ndarray,y: np.ndarray)->dict:
    try:import tensorflow as tf
    except ImportError as exc:raise CompilerUnavailable('TFLite CPU runtime absent') from exc
    from .compiler import inspect_tflite
    if inspect_tflite(Path(path))['npu_partition_count']:
        raise ValueError('A compiled Ethos-U graph must not be executed in the CPU evaluator')
    if x.ndim!=4 or not len(x) or len(x)!=len(y) or not np.isfinite(x).all():
        raise ValueError('Evaluation inputs must be finite with one label per image')
    runtime=tf.lite.Interpreter(model_path=str(path),num_threads=1,
                               experimental_op_resolver_type=tf.lite.experimental.OpResolverType.BUILTIN_REF)
    runtime.allocate_tensors();i=runtime.get_input_details()[0];o=runtime.get_output_details()[0]
    predictions=[]
    for image in x:
        value=image[None].transpose(0,2,3,1)
        if np.issubdtype(i['dtype'],np.integer):
            scale,z=i['quantization']
            if scale<=0:raise ValueError('Invalid input scale')
            lim=np.iinfo(i['dtype']);value=np.clip(np.rint(value/scale)+z,lim.min,lim.max).astype(i['dtype'])
        runtime.set_tensor(i['index'],value);runtime.invoke();output=runtime.get_tensor(o['index'])
        if output.ndim!=2 or output.shape[0]!=1 or not np.isfinite(output).all():
            raise ValueError('TFLite output must contain finite batch-one classification logits')
        predictions.append(int(output.argmax(1)[0]))
    correct=int(np.equal(predictions,y).sum())
    return {'accuracy':correct/len(y),'correct':correct,'n':len(y),'predictions':predictions,
            'backend':'tflite_BUILTIN_REF_CPU','npu_execution':False}
