#!/usr/bin/env python3
"""CPU-only natural-image study launcher; preserve evidence even on failure.

Uses the verified v0.2.1 pip-less-venv bootstrap. Raw CIFAR data and installed
packages are never included in result ZIPs. Completed tasks can resume in-place.
"""
from __future__ import annotations
import argparse
from datetime import datetime,timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import zipfile

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools.run_external_probe import bootstrap_environment


def bundle_evidence(session:Path,target:Path):
    tmp=target.with_suffix('.partial.zip')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for folder in ('logs','evidence'):
            for p in sorted((session/folder).rglob('*')):
                if p.is_symlink():raise ValueError('Symlink in study evidence')
                if p.is_file() and p.name!='session.lock':z.write(p,'nrr-natural-evidence/'+str(p.relative_to(session)))
        # Transport diagnostics survive failure. Raw data/partials are excluded.
        for p in sorted((session/'cache/download-evidence').rglob('*')):
            if p.is_symlink():raise ValueError('Symlink in download evidence')
            if p.is_file() and p.suffix in ('.json','.jsonl','.log'):
                z.write(p,'nrr-natural-evidence/downloads/'+str(p.relative_to(session/'cache/download-evidence')))
        if (session/'runner.json').exists():z.write(session/'runner.json','nrr-natural-evidence/runner.json')
    tmp.replace(target)


def launch(project:Path,session:Path,install:bool=True,protocol:Path|None=None)->int:
    if Path(session).is_symlink():raise ValueError('Session must not be a symlink')
    project=project.resolve();session=session.resolve()
    protocol=(protocol or project/'configs/cifar10_natural_v1.json').resolve()
    if not (project/'src/nrr_natural/pipeline.py').is_file():raise ValueError('Natural study source is missing')
    if not protocol.is_file():raise ValueError('Frozen protocol file is missing')
    if session.is_symlink():raise ValueError('Session must not be a symlink')
    if session.exists() and any(session.iterdir()) and not (session/'runner.json').is_file():
        raise ValueError('Refusing to use a non-study output directory')
    (session/'logs').mkdir(parents=True,exist_ok=True);(session/'evidence').mkdir(exist_ok=True)
    attempt=1
    while (session/'logs'/f'attempt-{attempt:03d}').exists():attempt+=1
    logs=session/'logs'/f'attempt-{attempt:03d}';logs.mkdir()
    if (session/'runner.json').exists():
        (session/'evidence'/f'previous-runner-{attempt-1:03d}.json').write_bytes((session/'runner.json').read_bytes())
    state={'schema':'nrr.natural-launch.v1','launcher_version':'0.3.1','status':'starting','commands':[],
           'attempt':attempt,'started_at':datetime.now(timezone.utc).isoformat(),'training_completed':False,
           'external_execution_confirmed':False,'hardware_executed':False,'gpu_requested':False,
           'install_requested':install,'protocol_file':protocol.name,'python':sys.version,
           'scope':'natural-image repeated-seed execution; inspect status and actual recorded sample counts'}
    env=os.environ.copy();env.pop('PYTHONHOME',None)
    env.update(PYTHONPATH=str(project/'src'),PYTHONUNBUFFERED='1',CUDA_VISIBLE_DEVICES='',
               OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1')
    bundle=session.parent/(session.name+'-results.zip')
    def save():
        tmp=session/'runner.tmp';tmp.write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        tmp.replace(session/'runner.json')
    def command(label,argv,timeot=900):
        print(f'[{label}] starting',flush=True)
        path=logs/(label+'.log')
        rec={'step':label,'argv':list(map(str,argv)),'log':str(path.relative_to(session)),'exit_code':None}
        state['commands'].append(rec);state['current_step']=label;save()
        proc=None;started=time.monotonic();last_heartbeat=started;position=0
        try:
            with path.open('w',encoding='utf-8') as logfile:
                proc=subprocess.Popen(argv,cwd=project,env=env,stdout=logfile,stderr=subprocess.STDOUT,start_new_session=True)
                while proc.poll() is None:
                    time.sleep(.5)
                    now=time.monotonic()
                    if label in ('natural-study','prepare-cifar'):
                        with path.open(encoding='utf-8',errors='replace') as reader:
                            reader.seek(position);chunk=reader.read();position=reader.tell()
                        for line in chunk.splitlines():
                            if line.startswith(('[training]','[shape_gate]','[frozen]','[evaluation]','[cifar]')):print(line,flush=True)
                    if now-last_heartbeat>30:
                        print(f'[{label}] running · elapsed {int(now-started)}s',flush=True);last_heartbeat=now
                    if now-started>timeout:
                        os.killpg(proc.pid,signal.SIGTERM)
                        try:proc.wait(timeout=10)
                        except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
                        rec.update(exit_code=124,status='timeout');return 124
                rec['exit_code']=proc.returncode;return proc.returncode
        except KeyboardInterrupt:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=10)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            rec.update(exit_code=130,status='interrupted');raise
        except OSError as exc:
            with path.open('a') as log:log.write(f'\n{type(exc).__name__}: {exc}\n')
            rec.update(exit_code=127,status='launch_error');return 127
        finally:
            rec['seconds']=time.monotonic()-started;save()
            if rec['exit_code'] not in (None,0):
                print(f'[{label}] failed: exit {rec["exit_code"]}; last log lines:',flush=True)
                print('\n'.join(path.read_text(errors='replace').splitlines()[-12:]),flush=True)
    code=3
    try:
        if os.name!='posix' or not (3,11)<=sys.version_info[:2]<=(3,13):
            raise RuntimeError('Use Linux/Colab CPU, Python 3.11–3.13 for the pinned dependencies')
        python=Path(sys.executable);driver=[str(python),'-m','pip']
        if install:
            state['status']='installing';save();venv=session/'.venv'
            if venv.exists():
                python=venv/'bin/python'
                check='import sys,pathlib; assert pathlib.Path(sys.prefix).resolve()==pathlib.Path(sys.argv[1]).resolve(); assert sys.prefix!=sys.base_prefix'
                if command('verify-existing-venv',[str(python),'-I','-c',check,str(venv)],timeout=120):
                    raise RuntimeError('Existing environment is not the expected isolated interpreter')
                driver=[sys.executable,'-m','pip','--isolated','--python',str(python)]
            else:python,driver=bootstrap_environment(venv,command)
            env['PATH']=str(python.parent)+os.pathsep+env.get('PATH','')
            install_base=driver+['install','--disable-pip-version-check','--timeout','30','--retries','1']
            if command('install-torch',install_base+['--only-binary=:all:','--index-url','https://download.pytorch.org/whl/cpu','torch==2.10.0']):
                raise RuntimeError('CPU PyTorch installation failed')
            if command('install-compiler',install_base+['--only-binary=:all:','numpy==2.2.6','scikit-learn>=1.4,<2',
                       'tensorflow-cpu==2.20.0','ethos-u-vela==5.1.0','setuptools>=68','wheel']):
                raise RuntimeError('Compiler dependency installation failed')
            if command('install-project',install_base+['--no-deps','--no-build-isolation','-e',str(project)]):
                raise RuntimeError('Local project installation failed')
        check='import torch,tensorflow as tf,importlib.metadata as m; from nrr.compiler import check_external_dependencies; check_external_dependencies(); print(torch.__version__,tf.__version__,m.version("ethos-u-vela")); assert m.version("ethos-u-vela")=="5.1.0"'
        if command('dependency-imports',[str(python),'-c',check],timeout=120):
            raise RuntimeError('Compiler environment not ready; no new training or natural-image result was produced')
        command('package-versions',driver+['freeze'],timeout=120)
        state['status']='preparing_data';save()
        data_code=command('prepare-cifar',[str(python),'-m','nrr_natural','prepare','--cache',str(session/'cache'),
                                       '--out',str(session/'prepared'),'--protocol',str(protocol)],timeout=900)
        if data_code:
            state['data_preparation_exit_code']=data_code
            detail='whole preparation exceeded 900s' if data_code==124 else 'download/read/split failed'
            raise RuntimeError(f'CIFAR preparation blocked: {detail}; inspect prepare-cifar.log and downloads; no synthetic substitution')
        (session/'evidence'/'dataset.json').write_bytes((session/'prepared/dataset.json').read_bytes())
        state['status']='running_study';save()
        code=command('natural-study',[str(python),'-m','nrr_natural','run','--data',str(session/'prepared/data.npz'),
                      '--protocol',str(protocol),'--out',str(session/'evidence/study')],timeout=10800)
        path=session/'evidence/study/summary.json'
        if path.is_file():
            result=json.loads(path.read_text())
            state.update(status=result['status'],training_completed=(session/'evidence/study/frozen.json').is_file(),
                         external_execution_confirmed=result['external_compiler_executed'],
                         completed_models=len(result['records']),expected_models=result['expected_models'])
            if result['status']!='observed' and code==0:code=2
        else:
            state.update(status='blocked',training_completed=(session/'evidence/study/frozen.json').is_file(),
                         reason='Study has not produced a complete summary; inspect study/state.json and task attempts')
            if code==0:code=2
    except KeyboardInterrupt:
        state.update(status='interrupted',reason='User/runtime interrupted; completed checkpoints preserved');code=130
    except Exception as exc:
        state.update(status='blocked',exception=type(exc).__name__,reason=str(exc));code=3
    finally:
        state.update(exit_code=code,finished_at=datetime.now(timezone.utc).isoformat());save()
        bundle_evidence(session,bundle)
        print(json.dumps({'status':state['status'],'exit_code':code,'evidence_zip':str(bundle)},ensure_ascii=False),flush=True)
    return code


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--session',type=Path,required=True);p.add_argument('--install',action='store_true')
    p.add_argument('--protocol',type=Path)
    a=p.parse_args();return launch(a.project,a.session,a.install,a.protocol)

if __name__=='__main__':raise SystemExit(main())
