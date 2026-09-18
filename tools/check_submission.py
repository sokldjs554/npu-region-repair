from __future__ import annotations
import csv, json, re, sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
DOCS=ROOT/'docs'

def fail(msg:str):
    raise SystemExit(f'submission check failed: {msg}')

def main()->int:
    required=[ROOT/'README.md',DOCS/'index.html',DOCS/'NATURAL_RESULTS.md',DOCS/'results/summary.json',DOCS/'results/records.csv']
    for p in required:
        if not p.is_file() or p.stat().st_size==0: fail(f'missing/empty {p.relative_to(ROOT)}')
    agg=json.loads((DOCS/'results/summary.json').read_text())
    if agg.get('status')!='observed': fail('aggregate status')
    if agg.get('expected_models')!=33 or agg.get('observed_records')!=33: fail('model count')
    if not agg.get('external_compiler_executed') or agg.get('npu_hardware_executed'): fail('compiler/hardware provenance')
    rows=list(csv.DictReader((DOCS/'results/records.csv').open()))
    if len(rows)!=33: fail(f'records.csv has {len(rows)} rows')
    page=(DOCS/'index.html').read_text()
    for needle in ['6 → 0','−26.22%','+2.01%p','−0.83%p','실제 NPU latency']:
        if needle not in page: fail(f'missing page claim {needle}')
    # Local relative hrefs from the static landing page must exist. Anchors are ignored.
    for href in re.findall(r'href="([^"]+)"',page):
        if href.startswith(('#','http://','https://')): continue
        target=(DOCS/href.split('#')[0]).resolve()
        if not target.exists(): fail(f'broken docs link: {href}')
    readme=(ROOT/'README.md').read_text()
    stale=['자연 이미지의 본 실험과 방법의 우위 검증은 아직 남아 있다','현재 단계: 소형 모델의 외부 컴파일 제약']
    for text in stale:
        if text in readme: fail(f'stale README text: {text}')
    if (ROOT/'LICENSE').exists() or (ROOT/'LICENSE.md').exists(): fail('unexpected license file')
    print('submission check: OK (33 records, observed compiler evidence, links and claims verified)')
    return 0

if __name__=='__main__':
    raise SystemExit(main())
