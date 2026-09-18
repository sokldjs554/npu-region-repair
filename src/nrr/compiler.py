"""Read actual Vela outputs, not a manually invented NPU support table."""
from __future__ import annotations
from pathlib import Path
import importlib.util
import shutil
import re
import subprocess
from .artifacts import digest_file,write_json


class CompilerUnavailable(RuntimeError): pass


def check_external_dependencies(executable: str='vela')->None:
    if shutil.which(executable) is None: raise CompilerUnavailable(f'{executable} is not installed; NPU evidence is unavailable')
    if importlib.util.find_spec('tensorflow') is None: raise CompilerUnavailable('TensorFlow is not installed; TFLite export cannot be verified')


def graph_evidence(nodes: list[dict])->dict:
    ids={};producers={}
    for node in nodes:
        if node['device'] not in ('host','npu') or node['id'] in ids:raise ValueError('Invalid device or duplicate node ID')
        ids[node['id']]=node
        for t in node['outputs']:
            if t in producers:raise ValueError('Multiple tensor producers')
            producers[t]=node['id']
    order = {node['id']: i for i, node in enumerate(nodes)}
    adjacency={i:set() for i in ids};boundary=set()
    for node in nodes:
        for t in node['inputs']:
            if t not in producers:continue
            before=producers[t]
            if order[before] >= order[node['id']]:
                raise ValueError('Graph is not in topological order or contains a cycle')
            a=ids[before]['device'];b=node['device']
            if a==b: adjacency[before].add(node['id']);adjacency[node['id']].add(before)
            else:boundary.add((t,a,b))
    unseen={i for i,n in ids.items() if n['device']=='host'};components=[]
    while unseen:
        todo=[min(unseen)];group=[]
        while todo:
            i=todo.pop()
            if i not in unseen:continue
            unseen.remove(i);group.append(i);todo.extend(adjacency[i]&unseen)
        components.append(sorted(group))
    return {'host_operator_count':sum(n['device']=='host' for n in nodes),
            'npu_operator_count':sum(n['device']=='npu' for n in nodes),
            'npu_partition_count':sum(n['device']=='npu' for n in nodes),
            'count_scope':'post_compilation_graph_not_original_model_ops',
            'host_components':components,'boundary_tensor_count':len(boundary),
            'boundary_tensors':[{'tensor':t,'from':a,'to':b} for t,a,b in sorted(boundary)],'operators':nodes}


def inspect_tflite(path: Path)->dict:
    try:from tensorflow.lite.python import schema_py_generated as schema
    except ImportError as exc:raise CompilerUnavailable('TensorFlow flatbuffer schema is absent') from exc
    blob=path.read_bytes();model=schema.Model.GetRootAsModel(blob,0)
    if model.SubgraphsLength()!=1:raise ValueError('First experiment accepts single-subgraph models only')
    enum={v:k for k,v in vars(schema.BuiltinOperator).items() if isinstance(v,int)}
    sg=model.Subgraphs(0);nodes=[]
    def name(ix):
        b=sg.Tensors(ix).Name()
        return b.decode('utf-8',errors='replace') if b else str(ix)
    for ix in range(sg.OperatorsLength()):
        op=sg.Operators(ix);code=model.OperatorCodes(op.OpcodeIndex());builtin=code.BuiltinCode()
        custom=code.CustomCode()
        custom=custom.decode('utf-8',errors='replace') if custom else None
        device='npu' if builtin==schema.BuiltinOperator.CUSTOM and custom=='ethos-u' else 'host'
        if builtin==schema.BuiltinOperator.CUSTOM and custom!='ethos-u':
            raise ValueError(f'Unrecognized custom operator: {custom}')
        inputs=[int(op.Inputs(i)) for i in range(op.InputsLength()) if op.Inputs(i)>=0]
        outputs=[int(op.Outputs(i)) for i in range(op.OutputsLength()) if op.Outputs(i)>=0]
        nodes.append({'id':ix,'op':custom or enum.get(builtin,str(builtin)),'device':device,'inputs':inputs,
                      'outputs':outputs,'output_names':[name(t) for t in outputs],
                      'output_dtypes':[int(sg.Tensors(t).Type()) for t in outputs]})
    return graph_evidence(nodes)


def run_vela(path: str | Path,out: str | Path,executable: str='vela',accelerator: str='ethos-u55-256')->dict:
    if shutil.which(executable) is None:raise CompilerUnavailable(f'{executable} is not installed; no compiler result was generated')
    path=Path(path).resolve();out=Path(out)
    if not path.is_file():raise ValueError('Input flatbuffer does not exist')
    if out.exists() and any(out.iterdir()):raise FileExistsError(f'Compiler output already populated: {out}')
    out.mkdir(parents=True,exist_ok=True)
    command=[executable,str(path),'--accelerator-config',accelerator,'--output-dir',str(out.resolve()),'--verbose-operators']
    version=subprocess.run([executable,'--version'],capture_output=True,text=True,check=True,timeout=30).stdout.strip()
    meta={'command':command,'version':version,'input_sha256':digest_file(path),
          'target':accelerator,'hardware_execution':False,'npu_latency_ms':None,'status':'started'}
    write_json(out/'command.json',meta)
    try:
        completed=subprocess.run(command,capture_output=True,text=True,timeout=180)
    except subprocess.TimeoutExpired as exc:
        def text(value):
            return value.decode('utf-8',errors='replace') if isinstance(value,bytes) else value or ''
        (out/'stdout.txt').write_text(text(exc.stdout),encoding='utf-8')
        (out/'stderr.txt').write_text(text(exc.stderr),encoding='utf-8')
        write_json(out/'command.json',{**meta,'exit_code':None,'status':'timeout','compiler_executed':True})
        raise RuntimeError('Vela timed out; partial stdout and stderr preserved') from exc
    (out/'stdout.txt').write_text(completed.stdout,encoding='utf-8')
    (out/'stderr.txt').write_text(completed.stderr,encoding='utf-8')
    meta={'command':command,'version':version,'exit_code':completed.returncode,'input_sha256':digest_file(path),
          'target':accelerator,'hardware_execution':False,'npu_latency_ms':None,
          'status':'completed' if completed.returncode==0 else 'failed','compiler_executed':True}
    write_json(out/'command.json',meta)
    if completed.returncode:raise RuntimeError(f'Vela failed ({completed.returncode}); see {out}/stderr.txt')
    files=list(out.glob('*_vela.tflite'))
    if len(files)!=1:raise RuntimeError('Expected exactly one compiled Vela TFLite file')
    evidence=inspect_tflite(files[0])
    report={**meta,**evidence,'status':'observed','output_sha256':digest_file(files[0]),
            'compiled_model':files[0].name,'scope':'Compiler topology, not measured NPU latency'}
    write_json(out/'evidence.json',report)
    return report


def require_external_baseline(report: dict | None)->None:
    if not report or report.get('status')!='observed' or not report.get('input_sha256') or not report.get('command'):
        raise ValueError('Actual compiler evidence is required before NPU comparison')
    if report.get('host_operator_count',0)<=0 or report.get('npu_operator_count',0)<=0:
        raise ValueError('Baseline does not demonstrate mixed NPU/host execution regions')
    names=' '.join(name for op in report.get('operators',[]) if op['device']=='host' for name in op.get('output_names',[]))
    if not all(re.search(r'(?:^|[\s;/])' + region + r'(?:[;/]|$)', names) for region in ('r0','r1')):
        raise ValueError('Cannot attribute observed host operations to BOTH candidate regions; stop rather than invent a mapping')
