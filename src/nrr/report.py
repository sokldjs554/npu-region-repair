"""Render stored records, keeping CPU measurements and compiler evidence separate."""
from __future__ import annotations
from pathlib import Path
import html

LABELS = {
    'continuation': '원형 추가 학습',
    'operatorwise': '연산자별 대체',
    'regionwise': '두 구간 병목 블록 대체',
    'single_region': '한 구간만 대체',
    'small_student': '소형 학생 모델 새 학습',
}


def _score(record: dict) -> str:
    return f"{record['correct']}/{record['n']} ({100 * record['accuracy']:.2f}%)"


def _html_table(headers: list[str], rows: list[list[str]]) -> str:
    head = ''.join('<th>' + html.escape(x) + '</th>' for x in headers)
    body = ''.join('<tr>' + ''.join('<td>' + html.escape(x) + '</td>' for x in row) + '</tr>' for row in rows)
    return '<table><thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table>'


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '|' + '|'.join(['---'] * len(headers)) + '|'] +
                     ['| ' + ' | '.join(row) + ' |' for row in rows])


def render(result: dict, out: Path) -> None:
    rows = []
    for key, method in result['methods'].items():
        training = method['training']
        rows.append([LABELS[key], f"{method['parameters']:,}", str(training['updates']), str(training['images']),
                     f"{100 * training['budget_spent'] / training['budget_cap']:.2f}%",
                     _score(method['fp32_test']), _score(method['cpu_quant_test'])])
    headers = ['방법', '파라미터', '학습 step', '학습 장수', 'MAC 예산 사용', 'FP32 시험 정확도', '양자화 CPU 시험 정확도']
    source, scope = result['data']['source'], result['data']['scope']
    external = result['external_compiler']; status = external['status']
    compiler_rows = []
    if status == 'observed':
        for label, record in [('원형 모델', external)] + [(LABELS[k], m.get('compiler', {})) for k, m in result['methods'].items()]:
            compiler_rows.append([label, str(record.get('host_operator_count', '미확인')),
                                  str(len(record['host_components'])) if 'host_components' in record else '미확인',
                                  str(record.get('boundary_tensor_count', '미확인'))])
    compiler_headers = ['모델', 'host_operator_count', 'host 연결 구간 수', '경계 텐서 수']
    compiler_text = (_md_table(compiler_headers, compiler_rows) if compiler_rows else
                     '미실행: host 연산 개수, 경계 텐서 수 및 NPU 시간은 확인되지 않았습니다. CPU 성공으로 대신 판정하지 않습니다.')
    compiler_html = (_html_table(compiler_headers, compiler_rows) if compiler_rows else '<p>' + compiler_text + '</p>')
    baseline = result['baseline']
    introduction = '단일 소형 모델의 첫 비교입니다. 최종 자연 이미지 연구 성과나 실제 NPU 성능 입증이 아닙니다.'
    budget = ('같은 상한의 Conv/Linear MAC 대리 예산을 사용합니다. 학생 순전파의 3배와 교사 순전파를 학습 이미지마다 청구합니다. '
              '정규화·메모리·실제 역전파 커널·검증·보정 비용을 모두 같게 맞춘 실험은 아닙니다. 작은 모델에는 더 많은 업데이트가 허용됩니다.')
    selection = ('후보는 방법별 한 개로 미리 고정했습니다. 검증셋으로 체크포인트를 선택하고 모든 방법의 선택을 끝낸 뒤 같은 시험셋을 평가했습니다. '
                 '보정은 학습 분할에서만 수행했습니다. 시험 결과로 승자나 하이퍼파라미터를 선택하지 않았습니다.')
    cpu_note = ('PyTorch x86의 실제 양자화 Conv/Linear 모듈을 실행했습니다. 다른 연산은 부동소수점으로 남을 수 있습니다. '
                '저장 후 같은 PyTorch 런타임에서 출력도 대조했으며, 이를 완전 정수 NPU 실행이나 독립 외부 엔진 검증으로 해석하지 않습니다.')
    limitation = ('외부 컴파일 결과가 있어도 그것은 컴파일러의 그래프 분할 관측일 뿐 실제 칩에서 잰 지연시간이 아닙니다. '
                  'Arm의 지원 조건을 모빌린트의 조건으로 대신하지 않습니다. 작은 단일 시드 결과로 방법의 우월성이나 신규성을 주장하지 않습니다.')
    baseline_text = f"원형 모델 FP32: {_score(baseline['fp32_test'])} · 양자화 CPU: {_score(baseline['cpu_quant_test'])}"
    md = ['# NPU Region Repair — 첫 실행 기록', '', '**' + introduction + '**', '',
          f"실행 모드: `{result['backend']}` · seed {result['seed']} · NPU 가설 검증 필드: `{result['npu_hypothesis_tested']}`",
          f"데이터 출처: {source}", f"데이터 범위: `{scope}`", '', baseline_text, '', _md_table(headers, rows), '',
          '## 비교 조건', '', budget, '', selection, '', cpu_note, '',
          '## 외부 컴파일러 상태', '', f"상태: `{status}`", '', compiler_text, '', limitation, '',
          '## 후속 검증', '',
          '실제 컴파일러가 혼합 실행 구간과 후보 위치를 확인하지 못하면 연구용 외부 비교를 중단합니다. '
          '자연 이미지와 확보된 원형 모델에 대한 검증 및 여러 시드 비교는 별도 단계입니다.', '',
          '원본 수치·예측·조건은 `result.json`, 파일 무결성 목록은 `manifest.json`입니다. 해시는 서명이나 학술 검증이 아닙니다.']
    (out / 'report.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    page = '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>NPU Region Repair · 첫 실행 기록</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f6f8;color:#192330;font:16px/1.7 system-ui,sans-serif}main{max-width:1280px;margin:auto;padding:40px 24px}
h1{font-size:34px;margin:0 0 12px}h2{font-size:21px;margin-top:28px}small{letter-spacing:.06em;color:#516072}.notice{border-left:4px solid #b87922;background:#fff4e3;padding:18px 22px;margin:24px 0}
section{background:white;border:1px solid #dce1e6;border-radius:12px;padding:24px;margin:20px 0}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;white-space:nowrap}th,td{padding:12px;border-bottom:1px solid #e3e6e8;text-align:right}th:first-child,td:first-child{text-align:left}th{font-size:13px;color:#516072}a{color:#175392}p{max-width:1120px;overflow-wrap:anywhere}code{background:#eef0f3;padding:2px 5px;overflow-wrap:anywhere}
@media(max-width:480px){main{padding:24px 12px}section{padding:16px}h1{font-size:28px}}
</style><main><small>MODEL REPAIR / FIRST COMPARISON</small><h1>NPU Region Repair</h1>
<p>연산자별 대체와 구간 단위 대체를 같은 학습 예산 기준으로 비교하기 위한 별도 프로젝트</p>'''
    page += '<div class="notice"><strong>' + introduction + '</strong><br>실행: ' + html.escape(result['backend']) + ' · seed ' + str(result['seed']) + '</div>'
    page += '<p>데이터: ' + html.escape(source) + '<br>범위: <code>' + html.escape(scope) + '</code></p>'
    page += '<section><h2>학습·평가 비교</h2><p>' + html.escape(baseline_text) + '</p><div class="scroll">' + _html_table(headers, rows) + '</div></section>'
    page += '<section><h2>예산을 읽는 방법</h2><p>' + budget + '</p><h2>선택·양자화 조건</h2><p>' + selection + '</p><p>' + cpu_note + '</p></section>'
    page += '<section><h2>외부 컴파일러 상태</h2><p><code>' + html.escape(status) + '</code></p><div class="scroll">' + compiler_html + '</div><p>' + limitation + '</p>'
    page += '<p><a href="report.md">상세 기록</a> · <a href="result.json">측정 JSON</a> · <a href="manifest.json">파일 무결성</a></p></section></main></html>'
    (out / 'report.html').write_text(page, encoding='utf-8')
