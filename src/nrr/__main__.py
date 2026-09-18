from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .experiment import run_experiment,verify_artifacts
from .compiler import CompilerUnavailable


def main(argv=None)->int:
    parser=argparse.ArgumentParser(description='Budgeted operator/region replacement, with explicit CPU/NPU evidence separation')
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('run')
    run.add_argument('--out',required=True)
    run.add_argument('--backend',choices=('cpu','vela'),default='cpu')
    run.add_argument('--teacher-steps',type=int,default=400)
    run.add_argument('--budget-steps',type=int,default=120,help='MAC-proxy cap expressed in equivalent baseline batches, not each arm step count')
    run.add_argument('--seed',type=int,default=17)
    run.add_argument('--width',type=int,default=16)
    run.add_argument('--threads',type=int,default=1)
    run.add_argument('--data',dest='data_path',help='NPZ with x/y/train/val/test; NCHW float32')
    check=sub.add_parser('verify');check.add_argument('out')
    for command in ('preflight','probe'):
        action=sub.add_parser(command)
        action.add_argument('--saved-run',required=True)
        action.add_argument('--out',required=True)
        action.add_argument('--data',dest='data_path')
        if command=='probe':
            action.add_argument('--executable',default='vela')
            action.add_argument('--accelerator',default='ethos-u55-256')
    args=vars(parser.parse_args(argv));command=args.pop('command')
    try:
        if command=='preflight':
            from .saved_probe import preflight
            result=preflight(**args)
            print(json.dumps({'status':result['status'],'compiler_executed':False,'path':str(Path(args['out'])/'preflight.json')}))
            return 0
        if command=='probe':
            from .probe_run import run_saved_probe
            result=run_saved_probe(**args)
            print(json.dumps({'status':result['status'],'compiler_executed':result['compiler_executed'],'reason':result.get('reason'),'path':str(Path(args['out'])/'probe.json')}))
            return result['exit_code']
        if command=='verify':
            result=verify_artifacts(args['out']);print(json.dumps(result,ensure_ascii=False));return 0 if result['ok'] else 1
        result=run_experiment(**args)
        print(json.dumps({'result':str(Path(args['out'])/'result.json'),'report':str(Path(args['out'])/'report.html'),'npu_hypothesis_tested':result['npu_hypothesis_tested'],'methods':{k:{'fp32':v['fp32_test']['accuracy'],'cpu_quant':v['cpu_quant_test']['accuracy'],'updates':v['training']['updates']} for k,v in result['methods'].items()}}))
        return 0
    except CompilerUnavailable as exc:
        print(f'External comparison blocked: {exc}',file=sys.stderr);return 3
    except (ValueError,OSError,RuntimeError) as exc:
        print(f'Experiment failed: {exc}',file=sys.stderr);return 2


if __name__=='__main__':raise SystemExit(main())
