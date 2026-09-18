#!/usr/bin/env python3
"""User-run CPU launcher. Installs only with --install; always packs failure logs.

python tools/run_external_probe.py --install --session /path/to/new-session
No remote repository, GPU, credentials or training is used.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
from collections.abc import Callable
import zipfile


def reserve_session(path: Path) -> None:
    if path.is_symlink() or path.exists():
        raise FileExistsError(f'Choose a new session directory: {path}')
    path.mkdir(parents=True)
    (path/'logs').mkdir();(path/'evidence').mkdir()


def create_bundle(session: Path, target: Path) -> None:
    paths=[]
    for folder in ('logs','evidence'):
        root=session/folder
        if root.exists():
            for path in root.rglob('*'):
                if path.is_symlink():raise ValueError(f'Refusing symlink in evidence: {path}')
                if path.is_file():paths.append(path)
    if (session/'runner.json').is_file():paths.append(session/'runner.json')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(paths):
            z.write(path,'nrr-external-evidence/'+str(path.relative_to(session)))



def bootstrap_environment(
    directory: Path,
    command: Callable[..., int],
) -> tuple[Path, list[str]]:
    """Create an isolated env without ensurepip and manage it using host pip.

    pip's --python option (>=22.3) supports pip-less environments. We never
    install into the notebook kernel or silently fall back to its packages.
    Every subprocess, including venv creation, goes through the log callback.
    """
    if directory.exists() or directory.is_symlink():
        raise FileExistsError(f'Choose a new virtual environment: {directory}')
    host = str(sys.executable)
    if command('create-venv', [host, '-m', 'venv', '--without-pip', str(directory)], timeout=120):
        raise RuntimeError('Virtual environment creation failed; see create-venv.log')
    bindir = directory / ('Scripts' if os.name == 'nt' else 'bin')
    # Do not resolve this executable: it may be a venv symlink to the host.
    python = bindir / ('python.exe' if os.name == 'nt' else 'python')
    check = (
        'import importlib.util,json,pathlib,sys; '
        'expected=pathlib.Path(sys.argv[1]).resolve(); '
        'print(json.dumps({"python":sys.version,"prefix":sys.prefix,"base_prefix":sys.base_prefix})); '
        'assert pathlib.Path(sys.prefix).resolve()==expected, "incorrect install target"; '
        'assert sys.prefix!=sys.base_prefix, "not a virtual environment"; '
        'assert importlib.util.find_spec("pip") is None, "unexpected inherited pip"'
    )
    if command('verify-isolation', [str(python), '-I', '-c', check, str(directory)], timeout=120):
        raise RuntimeError('Environment isolation verification failed; see verify-isolation.log')
    # --isolated ignores user pip config / PIP_TARGET etc. while --python fixes
    # the install target. No pip bootstrap (or download of get-pip.py) is used.
    pip_driver = [host, '-m', 'pip', '--isolated', '--python', str(python)]
    if command('check-pip-driver', pip_driver + ['--version'], timeout=120):
        raise RuntimeError('The host needs pip >=22.3 with --python support; see check-pip-driver.log')
    return python, pip_driver


def launch(project: Path, session: Path, install: bool, saved_run: Path | None = None) -> int:
    project=project.resolve();session=session.resolve()
    if not (project/'src/nrr/probe_run.py').is_file():raise ValueError('Project does not include the saved-checkpoint probe')
    saved_run=(saved_run or project/'verification/pilot-final-seed17').resolve()
    if session==saved_run or saved_run in session.parents:
        raise ValueError('Session must be outside the original saved run')
    reserve_session(session)
    state={'schema':'nrr.external-launch.v1','status':'starting','started_at':datetime.now(timezone.utc).isoformat(),
           'commands':[],'install_requested':install,'training_requested':False,'gpu_requested':False,
           'external_execution_confirmed':False,'launcher_version':'0.2.1',
           'runtime':{'python':sys.version,'executable':sys.executable,'platform':sys.platform}}
    bundle=session.parent/(session.name+'-results.zip')
    env=os.environ.copy()
    env.pop('PYTHONHOME', None)  # An inherited prefix can defeat a venv interpreter.
    env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',TF_NUM_INTRAOP_THREADS='1',TF_NUM_INTEROP_THREADS='1',
               CUDA_VISIBLE_DEVICES='',PYTHONPATH=str(project/'src'),PYTHONUNBUFFERED='1')
    # Do not record environment variables: they may contain unrelated secrets.
    def save():
        (session/'runner.json').write_text(json.dumps(state,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    def command(label,argv,timeout=900):
        print(f'[{label}]',flush=True)
        record={'step':label,'argv':[str(a) for a in argv],'log':f'logs/{label}.log','exit_code':None}
        state['current_step']=label
        state['commands'].append(record);save()
        try:
            with (session/record['log']).open('w',encoding='utf-8') as log:
                result=subprocess.run(argv,cwd=project,env=env,stdout=log,stderr=subprocess.STDOUT,
                                      timeout=timeout,check=False)
            record['exit_code']=result.returncode
            return result.returncode
        except subprocess.TimeoutExpired:
            record.update(exit_code=124,status='timeout')
            with (session/record['log']).open('a',encoding='utf-8') as log:
                log.write(f'\nTimed out after {timeout} seconds.\n')
            return 124
        except OSError as exc:
            record.update(exit_code=127,status='launch_error')
            with (session/record['log']).open('a',encoding='utf-8') as log:
                log.write(f'\n{type(exc).__name__}: {exc}\n')
            return 127
        finally:
            save()
            if record['exit_code'] not in (None, 0):
                print(f"[{label}] failed ({record['exit_code']}); log: {session/record['log']}",flush=True)
    code=3
    try:
        if not (3,11)<=sys.version_info[:2]<=(3,13):
            raise RuntimeError('Use Python 3.11–3.13 for these pinned CPU wheels')
        python=Path(sys.executable)
        pip_driver=[str(python),'-m','pip']
        if install:
            state['status']='installing';save()
            directory=session/'.venv'
            python,pip_driver=bootstrap_environment(directory,command)
            env['PATH']=str(python.parent)+os.pathsep+env.get('PATH','')
            state['bootstrap']='venv-without-pip + host-pip-python'
            state['installation_target']=str(directory)
            save()
            base=pip_driver+['install','--disable-pip-version-check','--timeout','30','--retries','1']
            if command('install-torch',base+['--only-binary=:all:','--index-url','https://download.pytorch.org/whl/cpu','torch==2.10.0']):
                raise RuntimeError('CPU PyTorch installation failed; see install-torch.log')
            if command('install-compiler',base+['--only-binary=:all:','numpy==2.2.6','scikit-learn>=1.4,<2',
                       'tensorflow-cpu==2.20.0','ethos-u-vela==5.1.0','pytest>=8,<10','setuptools>=68','wheel']):
                raise RuntimeError('Compiler dependency installation failed; see install-compiler.log')
            if command('install-project',base+['--no-deps','--no-build-isolation','-e',str(project)]):
                raise RuntimeError('Local project installation failed; see install-project.log')
        state['status']='preflight';save()
        code=command('saved-input-preflight',[str(python),'-m','nrr','preflight','--saved-run',str(saved_run),
                                              '--out',str(session/'evidence'/'inputs')],timeout=120)
        if code:raise RuntimeError('Saved input verification failed; original files were not changed')
        check='import torch,tensorflow as tf; import importlib.metadata as m; print("torch",torch.__version__); print("tensorflow",tf.__version__); print("vela",m.version("ethos-u-vela"))'
        code=command('dependency-imports',[str(python),'-c',check],timeout=120)
        if code:raise RuntimeError('Compiler modules did not import; no external result was produced')
        command('package-versions',pip_driver+['freeze'],timeout=120)
        state['status']='probing';save()
        code=command('saved-model-compiler-probe',[str(python),'-m','nrr','probe','--saved-run',str(saved_run),
                         '--out',str(session/'evidence'/'probe')],timeout=1200)
        result_path=session/'evidence'/'probe'/'probe.json'
        if result_path.is_file():
            result=json.loads(result_path.read_text(encoding='utf-8'))
            state.update(status=result['status'],probe_exit_code=result['exit_code'],
                         external_execution_confirmed=result['compiler_executed'],
                         baseline_gate=result.get('baseline_gate'),reason=result.get('reason'))
        else:
            state.update(status='failed',reason='Probe process ended without a final result; inspect subprocess log')
            if code==0:code=2
    except Exception as exc:
        state.update(status='blocked',reason=str(exc),exception=type(exc).__name__,
                     failed_step=state.get('current_step','runtime-check'))
        code=3 if state.get('status')=='blocked' else 2
    finally:
        state.update(exit_code=code,finished_at=datetime.now(timezone.utc).isoformat())
        save();create_bundle(session,bundle)
        print(json.dumps({'status':state['status'],'exit_code':code,'evidence_zip':str(bundle)},ensure_ascii=False),flush=True)
        print('The ZIP contains evidence/logs even on failure. It contains no virtual environment.',flush=True)
    return code


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    p.add_argument('--session',type=Path,required=True)
    p.add_argument('--saved-run',type=Path)
    p.add_argument('--install',action='store_true',help='create a dedicated venv and download fixed CPU dependencies')
    args=p.parse_args(argv)
    try:return launch(args.project,args.session,args.install,args.saved_run)
    except (ValueError,OSError) as exc:
        print(str(exc),file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
