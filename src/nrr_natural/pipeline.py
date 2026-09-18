"""One teacher per seed, fixed alternatives/budgets, all weights frozen before test."""
from __future__ import annotations
import contextlib,os,platform,importlib.metadata,time
from pathlib import Path
import numpy as np
import torch
from nrr.artifacts import digest_array,digest_file,state_digest,environment,write_json
from nrr.saved_probe import read_json
from nrr.models import RegionCNN,model_from_config,forward_macs
from nrr.training import train_teacher,fit_budgeted,batch_stream,evaluate
from nrr.experiment import METHODS,_checkpoint,_cpu_evaluation
from nrr.compiler import check_external_dependencies,require_external_baseline,run_vela,inspect_tflite
from nrr.tflite_bridge import convert_tflite,evaluate_tflite
from nrr.budget import batch_charge
from .models import candidate
from .protocol import StudyProtocol
from .tasks import run_task

def _load(path):
    blob=torch.load(path,map_location='cpu',weights_only=True)
    if not isinstance(blob,dict) or set(blob)!={'config','state_dict'}:raise ValueError('Invalid saved checkpoint')
    if any(not torch.is_tensor(v) or (v.is_floating_point() and not torch.isfinite(v).all()) for v in blob['state_dict'].values()):raise ValueError('Nonfinite checkpoint')
    with torch.random.fork_rng(devices=[]):model=model_from_config(blob['config'])
    model.load_state_dict(blob['state_dict'],strict=True);return model.eval()

def _fixed_json(path,value):
    if path.exists():
        if read_json(path)!=value:raise ValueError(f'Run identity differs: {path.name}')
    else:write_json(path,value)

def _runtime_identity(root):
    env=environment(root);versions={}
    for name in ('tensorflow-cpu','tensorflow','ethos-u-vela'):
        try:versions[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:versions[name]=None
    cpu=[];info=Path('/proc/cpuinfo')
    if info.exists():
        for line in info.read_text().splitlines():
            if line.startswith(('model name','flags')):
                cpu.append(line)
                if len(cpu)==2:break
    return {'python':env['python'],'torch':env['torch'],'numpy':env['numpy'],'platform':platform.platform(),'machine':platform.machine(),'cpu':cpu,'torch_build':torch.__config__.show(),'mkldnn_enabled':torch.backends.mkldnn.enabled,'onednn_isa':os.environ.get('ONEDNN_MAX_CPU_ISA'),'cpu_quant_engine':torch.backends.quantized.engine,'source_tree_sha256':env['source_tree_sha256'],'compiler_packages':versions}

@contextlib.contextmanager
def _lock(root):
    import fcntl
    with (root/'session.lock').open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:raise RuntimeError('This study is already running') from exc
        try:yield
        finally:fcntl.flock(f,fcntl.LOCK_UN)

def run_suite(data,protocol:StudyProtocol,out,backend='external'):
    if backend not in ('external','cpu-check'):raise ValueError('Unknown backend')
    if backend=='cpu-check' and data.scope!='synthetic_mechanism_check':raise ValueError('cpu-check is restricted to explicitly labelled synthetic mechanism fixtures')
    if data.x.shape[1:]!=(3,32,32):raise ValueError('Natural study requires RGB 32x32 inputs')
    root=Path(out)
    if root.is_symlink():raise ValueError('Output must not be a symlink')
    if root.exists() and not (root/'identity.json').is_file():
        names={f.name for f in root.iterdir()}
        if names-{'prepared_dataset.json','session.lock'}:raise ValueError('Refusing a non-study output directory')
    root.mkdir(parents=True,exist_ok=True)
    with _lock(root):return _run_suite(data,protocol,root,backend)

def _run_suite(data,p,root,backend):
    project=Path(__file__).resolve().parents[2]
    binding={'protocol_id':p.protocol_id,'backend':backend,'data_sha256':digest_array(data.x,data.y),'splits':{k:digest_array(getattr(data,k)) for k in ('train','val','test')},'runtime':_runtime_identity(project)}
    _fixed_json(root/'identity.json',binding);_fixed_json(root/'protocol.json',{'protocol_id':p.protocol_id,**p.to_dict()});_fixed_json(root/'data_manifest.json',data.manifest())
    previous_threads=torch.get_num_threads();torch.set_num_threads(p.threads);previous_determinism=torch.are_deterministic_algorithms_enabled();warn=torch.is_deterministic_algorithms_warn_only_enabled();torch.use_deterministic_algorithms(True)
    state={'status':'starting','phase':'environment','external_compiler_executed':False,'hardware_executed':False,'protocol_id':p.protocol_id,'data_scope':data.scope}
    def progress(phase,message):
        state.update(phase=phase,message=message);write_json(root/'state.json',state);print(f'[{phase}] {message}',flush=True)
    try:
        if backend=='external':
            check_external_dependencies()
            if importlib.metadata.version('ethos-u-vela')!='5.1.0':raise ValueError('Frozen protocol requires Vela 5.1.0')
            progress('shape_gate','RGB 32x32 untrained topology check (not accuracy evidence)')
            def gate(dest):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(p.seeds[0]);model=RegionCNN(3,data.classes,p.width).eval()
                cal=np.random.default_rng(p.split_seed).choice(data.train,min(p.calibration_images,len(data.train)),replace=False)
                exp=convert_tflite(model,data.x[cal],dest/'tflite',parity_images=data.x[data.val[:16]])
                comp=run_vela(exp['path'],dest/'vela');require_external_baseline(comp)
                return {'compiler':comp,'training_updates':0,'accuracy_evaluated':False,'scope':'untrained structural feasibility only','input_shape':[3,32,32]}
            _,shape_gate,_=run_task(root/'shape-gate',binding,gate);state['external_compiler_executed']=True
        else:shape_gate=None
        teachers={};checkpoints=[];train_records=[]
        for seed in p.seeds:
            progress('training',f'seed {seed}: teacher {p.teacher_steps} updates; completed tasks reused only after hash checks')
            def teacher_action(dest):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(seed);model=RegionCNN(3,data.classes,p.width);log=train_teacher(model,data,p.teacher_steps,seed,batch_size=p.batch_size)
                _checkpoint(model,dest/'model.pt');write_json(dest/'training.json',log);return {'training':log,'checkpoint_sha256':digest_file(dest/'model.pt')}
            task_id={**binding,'task':'teacher','seed':seed};tdir,tlog,_=run_task(root/'training'/f'seed-{seed}'/'teacher',task_id,teacher_action)
            teacher=_load(tdir/'model.pt');teacher_hash=state_digest(teacher);teachers[str(seed)]={'checkpoint':str((tdir/'model.pt').relative_to(root)),**tlog};tm=forward_macs(teacher,tuple(data.x.shape[1:]))
            calibration=np.random.default_rng(seed+101).choice(data.train,min(p.calibration_images,len(data.train)),replace=False)
            cpath=root/'training'/f'seed-{seed}'/'calibration.json';_fixed_json(cpath,{'ids':calibration.tolist(),'sha256':digest_array(calibration,data.x[calibration]),'split':'train'})
            caps={steps:batch_charge(tm,tm,p.batch_size)*steps for steps in p.budget_steps};costs=[batch_charge(forward_macs(candidate(teacher,m,seed+7),(3,32,32)),tm,p.batch_size) for m in METHODS];stream=batch_stream(data.train,seed+3,max(caps.values())//min(costs)+1,p.batch_size)
            stream_path=root/'training'/f'seed-{seed}'/'shared_batches.npy'
            if stream_path.exists():
                if not np.array_equal(np.load(stream_path,allow_pickle=False),stream):raise ValueError('Shared batch stream changed')
            else:np.save(stream_path,stream,allow_pickle=False)
            checkpoints.append({'seed':seed,'method':'teacher','budget_steps':None,'path':str((tdir/'model.pt').relative_to(root)),'sha256':digest_file(tdir/'model.pt'),'calibration_ids':calibration.tolist(),'training':tlog['training']})
            for steps,cap in caps.items():
                for method in METHODS:
                    progress('training',f'seed {seed} / budget {steps} / {method}')
                    def repair(dest):
                        model=candidate(teacher,method,seed+7);_checkpoint(model,dest/'before_training.pt');log=fit_budgeted(model,teacher,data,stream,cap,seed+7);_checkpoint(model,dest/'model.pt');write_json(dest/'training.json',log)
                        if state_digest(teacher)!=teacher_hash:raise RuntimeError('Teacher mutated')
                        return {'training':log,'checkpoint_sha256':digest_file(dest/'model.pt'),'shared_stream_sha256':digest_array(stream)}
                    arm_id={**binding,'task':'repair','seed':seed,'method':method,'budget_steps':steps,'teacher_sha256':tlog['checkpoint_sha256']}
                    d,log,_=run_task(root/'training'/f'seed-{seed}'/f'budget-{steps}'/method,arm_id,repair)
                    entry={'seed':seed,'method':method,'budget_steps':steps,'path':str((d/'model.pt').relative_to(root)),'sha256':digest_file(d/'model.pt'),'calibration_ids':calibration.tolist(),'training':log['training']};checkpoints.append(entry);train_records.append(entry)
        if len(checkpoints)!=p.expected_models:raise RuntimeError('Incomplete fixed candidate set; no test evaluation')
        freeze={'protocol_id':p.protocol_id,'data_sha256':binding['data_sha256'],'checkpoints':checkpoints,'policy':'all weights fixed before any test evaluation'};_fixed_json(root/'frozen.json',freeze)
        for row in checkpoints:
            if digest_file(root/row['path'])!=row['sha256']:raise ValueError('Frozen checkpoint integrity changed')
        progress('frozen',f'{len(checkpoints)} selected checkpoints frozen; final test evaluation starts now')
        rows=[];errors=[];test_hash=digest_array(data.test,data.x[data.test],data.y[data.test])
        for ix,entry in enumerate(checkpoints,1):
            seed,method,steps=entry['seed'],entry['method'],entry['budget_steps'];label=f'seed-{seed}_'+('teacher' if method=='teacher' else f'b{steps}_{method}');progress('evaluation',f'{ix}/{len(checkpoints)} {label}: CPU accuracy and compiler topology are separate')
            def inference(dest):
                model=_load(root/entry['path']);cal=np.asarray(entry['calibration_ids'],dtype=np.int64);tt=time.perf_counter();fp=evaluate(model,data,data.test);fp_seconds=time.perf_counter()-tt
                row={'seed':seed,'method':method,'budget_steps':steps,'checkpoint_sha256':entry['sha256'],'parameters':sum(v.numel() for v in model.parameters()),'fp32':fp,'tflite_cpu':None,'compiler':None,'training':entry['training'],'test_subset_sha256':test_hash,'test_ids':data.test.tolist(),'calibration_sha256':digest_array(cal,data.x[cal]),'timing':{'fp32_cpu_seconds':fp_seconds},'npu_hardware_executed':False}
                if backend=='external':
                    tt=time.perf_counter();exp=convert_tflite(model,data.x[cal],dest/'tflite',parity_images=data.x[data.val[:16]]);row['timing']['quantization_export_parity_seconds']=time.perf_counter()-tt;tt=time.perf_counter();q=evaluate_tflite(exp['path'],data.x[data.test],data.y[data.test]);row['timing']['tflite_cpu_seconds']=time.perf_counter()-tt;q.update(subset_sha256=test_hash,ids=data.test.tolist());row['tflite_cpu']=q;row['export']=exp;row['precompile_graph']=inspect_tflite(Path(exp['path']));tt=time.perf_counter();comp=run_vela(exp['path'],dest/'vela');row['timing']['compile_seconds']=time.perf_counter()-tt;row['compiler']=comp
                    if method=='teacher':require_external_baseline(comp)
                else:
                    tt=time.perf_counter();row['cpu_check']=_cpu_evaluation(model,data,cal,dest);row['timing']['torch_cpu_quant_seconds']=time.perf_counter()-tt
                return row
            eid={**binding,'task':'evaluation','checkpoint_sha256':entry['sha256'],'seed':seed,'method':method,'budget_steps':steps,'test_subset_sha256':test_hash}
            try:
                _,row,_=run_task(root/'evaluation'/label,eid,inference);rows.append(row)
            except Exception as exc:
                errors.append({'key':label,'exception':type(exc).__name__,'reason':str(exc)});progress('evaluation',f'{label} failed; preserving evidence, continuing remaining fixed arms')
        result={'schema':'nrr.natural-study.v1','protocol_id':p.protocol_id,'protocol':p.to_dict(),'status':('observed' if backend=='external' else 'mechanism_check_complete') if not errors else 'partial','external_compiler_executed':backend=='external','npu_hardware_executed':False,'backend':backend,'records':rows,'errors':errors,'teachers':teachers,'shape_gate':shape_gate,'data':data.manifest(),'test_subset_sha256':test_hash,'test_labels':data.y[data.test].tolist(),'test_ids':data.test.tolist(),'run_identity':binding,'scope':data.scope,'expected_models':p.expected_models,'interpretation':'Successful-path MAC budgets only. Three seeds are not independent copies of the test dataset.','failed_attempts':[str(f.relative_to(root)) for f in root.rglob('attempt.json') if read_json(f).get('status')=='failed']}
        from .aggregate import aggregate
        from .report import render
        result['aggregate']=aggregate(result);write_json(root/'summary.json',result);render(result,root);state.update(status=result['status'],phase='complete' if not errors else 'partial',records_completed=len(rows),expected_models=p.expected_models);write_json(root/'state.json',state);return result
    except Exception as exc:
        state.update(status='blocked',exception=type(exc).__name__,reason=str(exc));write_json(root/'state.json',state);raise
    finally:
        torch.set_num_threads(previous_threads);torch.use_deterministic_algorithms(previous_determinism,warn_only=warn)
