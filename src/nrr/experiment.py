"""End-to-end executable comparison, with an external evidence gate."""
from __future__ import annotations
import copy
from datetime import datetime,timezone
import math
from pathlib import Path
import json
import warnings
import numpy as np
import torch
from .artifacts import reserve_output,write_json,digest_file,digest_array,state_digest,environment
from .budget import positive_int,batch_charge
from .data import load_digits_split,load_npz
from .models import RegionCNN,make_candidate,forward_macs
from .training import train_teacher,batch_stream,fit_budgeted,evaluate
from .cpu_quant import quantize_cpu
from .compiler import check_external_dependencies,run_vela,require_external_baseline
from .report import render

METHODS=('continuation','operatorwise','regionwise','single_region','small_student')


def verify_artifacts(out: str | Path)->dict:
    p=Path(out);manifest=json.loads((p/'manifest.json').read_text(encoding='utf-8'))
    errors=[]
    if (p/'blocked.json').exists() or (p/'.running').exists():
        errors.append('run is blocked or incomplete')
    required={'result.json','report.md','report.html','data_manifest.json','calibration.json'}
    if not required.issubset(manifest.get('files',{})):
        errors.append('manifest lacks required completed-run artifacts')
    for name,expected in manifest['files'].items():
        target=(p/name).resolve()
        if p.resolve() not in target.parents:errors.append(f'unsafe path: {name}');continue
        if not target.is_file() or digest_file(target)!=expected['sha256']:errors.append(name)
    return {'ok':not errors,'files_checked':len(manifest['files']),'errors':errors}


def _checkpoint(model,path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'config':model.config,'state_dict':model.state_dict()},path)


def _cpu_evaluation(model,data,calibration,out: Path)->dict:
    old = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        result = _cpu_evaluation_impl(model,data,calibration,out)
        result['cpu_quantization']['strict_deterministic_algorithms'] = False
        result['cpu_quantization']['determinism_note'] = 'Quantized CPU resize lacks strict-mode support; no cross-runtime bit guarantee.'
        return result
    finally:
        torch.use_deterministic_algorithms(old, warn_only=warn)


def _cpu_evaluation_impl(model,data,calibration,out: Path)->dict:
    q,info=quantize_cpu(model,torch.from_numpy(data.x[calibration]))
    evaluated=evaluate(q,data,data.test)
    (out/'cpu_quant_graph.txt').write_text(info.pop('graph'),encoding='utf-8')
    x=torch.from_numpy(data.x[data.val[:7]])
    with warnings.catch_warnings(record=True) as ws:
        warnings.simplefilter('always')
        traced=torch.jit.trace(q,x[:1]);traced.save(str(out/'cpu_quantized.pt'))
        recovered=torch.jit.load(str(out/'cpu_quantized.pt')).eval()
        with torch.no_grad():error=float((q(x)-recovered(x)).abs().max())
    if error!=0:raise RuntimeError('CPU serialized output mismatch')
    return {'cpu_quant_test':evaluated,'cpu_quantization':info,
            'scripted_cpu_roundtrip':{'max_abs':error,'images':len(x),'scope':'same PyTorch CPU runtime',
                                      'warnings':sorted({str(w.message) for w in ws})}}


def run_experiment(out: str | Path,backend: str='cpu',teacher_steps: int=400,budget_steps: int=120,
                   seed: int=17,width: int=16,threads: int=1,data_path: str | None=None)->dict:
    if backend not in ('cpu','vela'):raise ValueError('backend must be cpu or vela')
    for name,v in [('teacher_steps',teacher_steps),('budget_steps',budget_steps),('threads',threads)]:positive_int(v,name)
    if not isinstance(seed,int) or isinstance(seed,bool) or not 0<=seed<2**32-1:raise ValueError('Invalid seed')
    if not isinstance(width,int) or width<4 or width%2:raise ValueError('width must be even and >=4')
    out=reserve_output(out);old_threads=torch.get_num_threads();old_deterministic=torch.are_deterministic_algorithms_enabled()
    old_warn=torch.is_deterministic_algorithms_warn_only_enabled()
    try:
        if backend=='vela':check_external_dependencies()
        torch.set_num_threads(threads);torch.use_deterministic_algorithms(True)
        data=load_npz(data_path) if data_path else load_digits_split(seed)
        write_json(out/'data_manifest.json',data.manifest())
        root=Path(__file__).resolve().parents[2]
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed);teacher=RegionCNN(data.x.shape[1],data.classes,width)
            teacher_log=train_teacher(teacher,data,teacher_steps,seed)
            _checkpoint(teacher,out/'teacher/model.pt');teacher_hash=state_digest(teacher)
            write_json(out/'teacher/training.json',teacher_log)
            calibration=np.random.default_rng(seed+101).choice(data.train,min(128,len(data.train)),replace=False)
            write_json(out/'calibration.json',{'split':'train','ids':calibration.tolist(),'sha256':digest_array(calibration,data.x[calibration])})
            external={'status':'not_run','host_operator_count':None,'npu_latency_ms':None,
                      'reason':'CPU mechanism check; no inference about NPU operator support'}
            if backend=='vela':
                from .tflite_bridge import convert_tflite
                exp=convert_tflite(teacher,data.x[calibration],out/'teacher/tflite')
                external=run_vela(exp['path'],out/'teacher/vela')
                require_external_baseline(external)
            tm=forward_macs(teacher,tuple(data.x.shape[1:]));cap=batch_charge(tm,tm,32)*budget_steps
            models={name:make_candidate(teacher,name,seed+7) for name in METHODS}
            costs=[batch_charge(forward_macs(m,tuple(data.x.shape[1:])),tm,32) for m in models.values()]
            stream=batch_stream(data.train,seed+3,cap//min(costs)+1,32)
            np.save(out/'shared_batch_stream.npy',stream,allow_pickle=False)
            records={};initial={}
            for name,model in models.items():
                destination=out/name;destination.mkdir()
                initial[name]=copy.deepcopy(model).eval()
                _checkpoint(model,destination/'before_training.pt')
                training=fit_budgeted(model,teacher,data,stream,cap,seed+7)
                _checkpoint(model,destination/'selected.pt')
                write_json(destination/'training.json',training)
                records[name]={'parameters':sum(p.numel() for p in model.parameters()),
                               'config':model.config,'training':training}
                if backend=='vela':
                    exp=convert_tflite(model,data.x[calibration],destination/'tflite')
                    records[name]['compiler']=run_vela(exp['path'],destination/'vela')
            result={'schema':'nrr.comparison.v1','created_at':datetime.now(timezone.utc).isoformat(),
                    'backend':backend,'seed':seed,'npu_hypothesis_tested':backend=='vela',
                    'scope':'single_custom_small_CNN_mechanism_pilot; not final portfolio evidence',
                    'external_compiler':external,'data':data.manifest(),'environment':environment(root),
                    'teacher_training':teacher_log,'calibration_ids':calibration.tolist(),
                    'protocol':{'methods':list(METHODS),'candidates_per_method':1,'hidden_search_trials':0,
                                'student_loss':'0.5 CE + 0.5 KL(T=2,T^2) + 0.1 final-region feature MSE',
                                'budget_cap_per_method':cap,'budget_unit':'conv/linear MAC proxy',
                                'shared_teacher_pretraining_excluded_and_reported':True,
                                'candidate_selection':'none; fixed controls; no winning method selected',
                                'checkpoint_selection':'validation only; fixed budget fractions; earliest tie',
                                'common_batch_stream_sha256':digest_array(stream)},'methods':records}
            baseline={'parameters':sum(p.numel() for p in teacher.parameters()),'fp32_test':evaluate(teacher,data,data.test)}
            baseline.update(_cpu_evaluation(teacher,data,calibration,out/'teacher'))
            result['baseline']=baseline
            for name,model in models.items():
                records[name]['before_training_test']=evaluate(initial[name],data,data.test)
                records[name]['fp32_test']=evaluate(model,data,data.test)
                records[name].update(_cpu_evaluation(model,data,calibration,out/name))
                if backend=='vela':
                    from .tflite_bridge import evaluate_tflite
                    records[name]['tflite_cpu_test']=evaluate_tflite(out/name/'tflite/model.tflite',data.x[data.test],data.y[data.test])
            if backend=='vela':
                baseline['tflite_cpu_test']=evaluate_tflite(out/'teacher/tflite/model.tflite',data.x[data.test],data.y[data.test])
            if state_digest(teacher)!=teacher_hash:raise RuntimeError('Teacher changed during comparison')
            result['teacher_preserved']=True
            write_json(out/'result.json',result);render(result,out)
            (out/'.running').unlink()
            files={str(p.relative_to(out)):{'sha256':digest_file(p),'bytes':p.stat().st_size} for p in sorted(out.rglob('*')) if p.is_file()}
            write_json(out/'manifest.json',{'files':files,'scope':'local artifact integrity, not provenance certification'})
            return result
    except Exception as exc:
        write_json(out/'blocked.json',{'status':'blocked','exception':type(exc).__name__,'reason':str(exc),
                                      'npu_hypothesis_tested':False,'no_success_result_should_be_inferred':True})
        raise
    finally:
        torch.set_num_threads(old_threads);torch.use_deterministic_algorithms(old_deterministic,warn_only=old_warn)
