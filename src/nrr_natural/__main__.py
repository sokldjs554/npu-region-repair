"""CLI for the fixed natural-image protocol. Network access is explicit in prepare."""
import argparse
from dataclasses import fields
import json
from pathlib import Path
import sys
import numpy as np
from nrr.artifacts import digest_file,write_json
from nrr.data import Dataset
from nrr.saved_probe import read_json
from .protocol import StudyProtocol
from .cifar import prepare,ARCHIVE_SHA256
from .pipeline import run_suite


def load_protocol(path=None):
    if path is None:return StudyProtocol()
    d=read_json(Path(path));kw={f.name:d[f.name] for f in fields(StudyProtocol) if f.name in d}
    for n in ('seeds','budget_steps'):
        if n in kw:kw[n]=tuple(kw[n])
    p=StudyProtocol(**kw)
    if d.get('protocol_id',p.protocol_id)!=p.protocol_id:raise ValueError('Protocol checksum differs')
    if set(d)-set(p.to_dict())-{'protocol_id'}:raise ValueError('Unknown protocol fields')
    for key,value in d.items():
        if key!='protocol_id' and p.to_dict()[key]!=value:raise ValueError(f'Protocol field differs: {key}')
    return p


def load_prepared(path,protocol):
    path=Path(path);meta=read_json(path.parent/'dataset.json')
    if meta.get('protocol_id')!=protocol.protocol_id:raise ValueError('Prepared data belongs to another protocol')
    if meta.get('archive_sha256')!=ARCHIVE_SHA256 or meta.get('npz_sha256')!=digest_file(path):
        raise ValueError('Prepared dataset integrity differs')
    with np.load(path,allow_pickle=False) as z:
        arrays={k:z[k].copy() for k in ('x','y','train','val','test')}
    data=Dataset(**arrays,source=meta['source'],scope=meta['scope'])
    if data.manifest()['data_sha256']!=meta['data_sha256']:raise ValueError('Prepared tensor hash differs')
    if len(data.train)!=10*protocol.train_per_class or len(data.val)!=10*protocol.val_per_class:
        raise ValueError('Prepared split size differs')
    if len(data.test)!=(protocol.test_limit or 10000):raise ValueError('Prepared test size differs')
    return data,meta


def main(argv=None):
    parser=argparse.ArgumentParser(description='Fixed CIFAR-10 repeated-seed model-repair study')
    sub=parser.add_subparsers(dest='command',required=True)
    pp=sub.add_parser('prepare');pp.add_argument('--cache',required=True);pp.add_argument('--out',required=True)
    pp.add_argument('--protocol');pp.add_argument('--no-download',action='store_true')
    rp=sub.add_parser('run');rp.add_argument('--data',required=True);rp.add_argument('--out',required=True)
    rp.add_argument('--protocol');rp.add_argument('--backend',choices=['external','cpu-check'],default='external')
    args=parser.parse_args(argv)
    try:
        p=load_protocol(args.protocol)
        if args.command=='prepare':
            out=Path(args.out)
            if out.exists():
                load_prepared(out/'data.npz',p)
                print('Reusing verified prepared data',flush=True)
            else:prepare(args.cache,out,p,download=not args.no_download)
            print(json.dumps({'data':str(out/'data.npz'),'protocol_id':p.protocol_id}));return 0
        data,meta=load_prepared(args.data,p)
        out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
        binding=out/'prepared_dataset.json'
        if binding.exists() and read_json(binding)!=meta:raise ValueError('Run prepared-data identity differs')
        write_json(binding,meta)
        result=run_suite(data,p,out,args.backend)
        print(json.dumps({'status':result['status'],'records':len(result['records']),'expected':p.expected_models}))
        return 0 if result['status'] in ('observed','mechanism_check_complete') else 2
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        return 3

if __name__=='__main__':raise SystemExit(main())
