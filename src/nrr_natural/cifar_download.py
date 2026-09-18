"""Checksum-pinned HTTPS transfer with a hard per-source subprocess deadline.

Standard library only, so a stalled DNS/connect/read cannot trap the supervisor.
TLS is verified. Source failures never weaken the expected byte count or SHA256.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import urllib.parse
import urllib.request


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def _https(url: str) -> None:
    p = urllib.parse.urlsplit(url)
    if p.scheme != 'https' or not p.hostname or p.username or p.password:
        raise ValueError('Only credential-free HTTPS sources are accepted')


class _HTTPSRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _https(newurl)  # reject downgrades BEFORE making the next request
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _worker(url: str, destination: Path, expected_bytes: int, socket_timeout: float) -> None:
    _https(url)
    opener = urllib.request.build_opener(_HTTPSRedirect())
    request = urllib.request.Request(url, headers={'Accept-Encoding':'identity', 'User-Agent':'nrr-cifar/0.3.1'})
    with opener.open(request, timeout=socket_timeout) as response:
        _https(response.geturl())
        if response.status != 200:
            raise ValueError(f'Expected HTTP 200, received {response.status}')
        length = response.headers.get('Content-Length')
        if length is not None and int(length) != expected_bytes:
            raise ValueError(f'Content-Length differs from pinned archive: {length}')
        print(json.dumps({'http_status':response.status, 'final_url':response.geturl(),
                          'content_length':length}), flush=True)
        size = 0
        with destination.open('xb') as target:
            while block := response.read1(65536):
                size += len(block)
                if size > expected_bytes:
                    raise ValueError('Download exceeds pinned archive size')
                target.write(block)
                target.flush()
        if size != expected_bytes:
            raise ValueError(f'Truncated archive: {size}/{expected_bytes} bytes')


def _stop(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=2)


def fetch_archive(cache: str | Path, name: str, expected_sha256: str, expected_bytes: int,
                  urls: tuple[str, ...], *, attempt_timeout: float = 180,
                  socket_timeout: float = 15, progress_interval: float = 5) -> Path:
    cache = Path(cache).absolute()
    if not name or Path(name).name != name or name in ('.','..'):
        raise ValueError('Archive name must be a single filename')
    if len(expected_sha256) != 64 or any(c not in '0123456789abcdef' for c in expected_sha256):
        raise ValueError('Expected SHA256 must be lowercase hexadecimal')
    if not isinstance(expected_bytes, int) or isinstance(expected_bytes, bool) or expected_bytes <= 0:
        raise ValueError('Expected archive bytes must be positive')
    for value in (attempt_timeout, socket_timeout, progress_interval):
        if not math.isfinite(value) or value <= 0:
            raise ValueError('Timeouts and progress interval must be finite and positive')
    if not urls:
        raise ValueError('At least one HTTPS source is required')
    for url in urls:
        _https(url)
    if any(p.is_symlink() for p in (cache, *cache.parents)):
        raise ValueError('Cache path must not contain a symlink')
    cache.mkdir(parents=True, exist_ok=True)
    path = cache / name
    if path.is_symlink():
        raise ValueError('Archive must not be a symlink')
    if path.exists():
        if not path.is_file() or path.stat().st_size != expected_bytes or sha256(path) != expected_sha256:
            raise ValueError('Cached archive checksum/size differs; existing file preserved')
        print('[cifar] '+json.dumps({'event':'cache_verified','bytes':expected_bytes,'sha256':expected_sha256}),flush=True)
        return path
    history = cache / 'download-evidence'
    if history.is_symlink():
        raise ValueError('Download evidence must not be a symlink')
    history.mkdir(exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix='attempt-', dir=history))
    events = run / 'events.jsonl'
    errors = []
    def emit(event: str, **kw):
        record = {'event':event,'time':datetime.now(timezone.utc).isoformat(),**kw}
        text = json.dumps(record, ensure_ascii=False, allow_nan=False)
        with events.open('a', encoding='utf-8') as log:
            log.write(text+'\n')
        print('[cifar] '+text, flush=True)
    emit('download_start',expected_bytes=expected_bytes,sha256=expected_sha256,
         source_count=len(urls),attempt_timeout=attempt_timeout)
    for index,url in enumerate(urls):
        folder = run / f'source-{index+1}'
        folder.mkdir()
        partial = folder / 'archive.partial'
        log_path = folder / 'worker.log'
        proc = None
        start = time.monotonic()
        emit('source_start',source=index+1,url=url,bytes=0)
        reason = None
        try:
            argv = [sys.executable,'-u',str(Path(__file__).resolve()),'--worker',
                    '--url',url,'--destination',str(partial),'--expected-bytes',str(expected_bytes),
                    '--socket-timeout',str(min(socket_timeout,attempt_timeout))]
            with log_path.open('w',encoding='utf-8') as log:
                proc = subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
                next_progress = start
                while proc.poll() is None:
                    now = time.monotonic()
                    if now - start >= attempt_timeout:
                        reason='attempt_timeout'
                        _stop(proc)
                        raise TimeoutError('Whole-source transfer deadline exceeded')
                    if now >= next_progress:
                        size = partial.stat().st_size if partial.is_file() else 0
                        emit('download_progress',source=index+1,bytes=size,expected_bytes=expected_bytes,
                             elapsed_seconds=round(now-start,2))
                        next_progress=now+progress_interval
                    time.sleep(min(.05,progress_interval))
            if proc.returncode != 0:
                raise RuntimeError(f'worker_exit_{proc.returncode}: '+log_path.read_text(errors='replace')[-1200:])
            emit('hash_check',source=index+1,bytes=partial.stat().st_size)
            if partial.stat().st_size != expected_bytes or sha256(partial) != expected_sha256:
                raise ValueError('Downloaded archive checksum/size differs')
            if path.is_symlink() or (path.exists() and (not path.is_file() or sha256(path) != expected_sha256)):
                raise ValueError('Archive target changed during transfer; target preserved')
            partial.replace(path)
            emit('download_verified',source=index+1,url=url,bytes=expected_bytes,
                 sha256=expected_sha256,elapsed_seconds=round(time.monotonic()-start,2))
            return path
        except (OSError,ValueError,RuntimeError) as exc:
            reason=reason or type(exc).__name__
            errors.append({'source':index+1,'url':url,'reason':reason,'message':str(exc)})
            emit('source_failed',**errors[-1],bytes=partial.stat().st_size if partial.is_file() else 0,
                 elapsed_seconds=round(time.monotonic()-start,2))
        finally:
            if proc is not None:
                _stop(proc)
            partial.unlink(missing_ok=True)
    emit('download_failed',errors=errors)
    raise RuntimeError('No verified CFAR archive downloaded; see download-evidence. '+
                        ' | '.join(f'{e["url"]}: {e["reason"]}' for e in errors))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker',action='store_true',required=True)
    parser.add_argument('--url',required=True)
    parser.add_argument('--destination',type=Path,required=True)
    parser.add_argument('--expected-bytes',type=int,required=True)
    parser.add_argument('--socket-timeout',type=float,required=True)
    a=parser.parse_args()
    try:
        _worker(a.url,a.destination,a.expected_bytes,a.socket_timeout)
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}',file=sys.stderr,flush=True)
        return 2
    return 0


if __name__=='__main__':
    raise SystemExit(main())
