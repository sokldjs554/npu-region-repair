"""Small, fail-closed artifact utilities. Nothing mutates source checkpoints."""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import numpy as np
import torch


def reserve_output(path: str | Path) -> Path:
    p=Path(path)
    if p.is_symlink() or (p.exists() and (not p.is_dir() or any(p.iterdir()))):
        raise FileExistsError(f'Output must be a new or empty non-symlink directory: {p}')
    p.mkdir(parents=True, exist_ok=True)
    with (p/'.running').open('x',encoding='utf-8') as f: f.write('in progress\n')
    return p


def write_json(path: str | Path, value: object) -> None:
    p=Path(path)
    text=json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    p.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix=p.name+'.',suffix='.tmp',dir=p.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f: f.write(text)
        os.replace(tmp,p)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def digest_file(path: str | Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()


def digest_array(*arrays: np.ndarray) -> str:
    h=hashlib.sha256()
    for a in arrays:
        a=np.ascontiguousarray(a)
        h.update(str(a.dtype).encode()); h.update(str(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def state_digest(model: torch.nn.Module) -> str:
    h=hashlib.sha256()
    for key,value in sorted(model.state_dict().items()):
        h.update(key.encode())
        if torch.is_tensor(value): h.update(digest_array(value.detach().cpu().numpy()).encode())
        else: h.update(repr(value).encode())
    return h.hexdigest()


def environment(root: Path) -> dict:
    try:
        commit=subprocess.run(['git','rev-parse','HEAD'],cwd=root,capture_output=True,text=True,check=True,timeout=3).stdout.strip()
    except (OSError,subprocess.SubprocessError): commit=None
    sources={str(p.relative_to(root)):digest_file(p) for p in sorted((root/'src').rglob('*.py'))}
    return {'python':sys.version,'platform':platform.platform(),'torch':str(torch.__version__),'numpy':np.__version__,'cuda_used':False,'threads':torch.get_num_threads(),'git_commit':commit,'source_sha256':sources,'source_tree_sha256':hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest()}
