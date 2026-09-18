"""Hash-verified completed tasks; incomplete attempts are preserved, never silently reused."""
from pathlib import Path
import time
from nrr.artifacts import digest_file, write_json
from nrr.saved_probe import read_json

def run_task(root: Path,identity: dict,action):
    root=Path(root)
    if root.is_symlink():raise ValueError('Task root is a symlink')
    root.mkdir(parents=True,exist_ok=True);done=root/'complete.json'
    if done.is_file():
        seal=read_json(done)
        if seal.get('identity')!=identity:raise ValueError('Completed task identity differs')
        name=seal.get('attempt','')
        if not name.startswith('attempt-') or '/' in name or '\\' in name:raise ValueError('Unsafe task attempt')
        folder=root/name
        actual={str(p.relative_to(folder)):digest_file(p) for p in folder.rglob('*') if p.is_file() and not p.is_symlink()}
        if any(p.is_symlink() for p in folder.rglob('*')) or actual!=seal.get('files'):raise ValueError('Completed task integrity differs')
        if not (folder/'record.json').is_file() or read_json(folder/'record.json')!=seal.get('record'):raise ValueError('Completed task record integrity differs')
        return folder,seal['record'],True
    index=1
    while (root/f'attempt-{index:03d}').exists():index+=1
    folder=root/f'attempt-{index:03d}';folder.mkdir();started=time.perf_counter()
    write_json(folder/'attempt.json',{'identity':identity,'status':'running','attempt_number':index})
    try:
        record=action(folder);write_json(folder/'record.json',record)
        write_json(folder/'attempt.json',{'identity':identity,'status':'complete','attempt_number':index,'seconds':time.perf_counter()-started})
        files={str(p.relative_to(folder)):digest_file(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        write_json(done,{'identity':identity,'attempt':folder.name,'files':files,'record':record})
        return folder,record,False
    except BaseException as exc:
        write_json(folder/'attempt.json',{'identity':identity,'status':'failed','attempt_number':index,'seconds':time.perf_counter()-started,'exception':type(exc).__name__,'reason':str(exc),'failed_attempt_compute':'not fully metered; not part of successful-path MAC comparison'})
        raise
