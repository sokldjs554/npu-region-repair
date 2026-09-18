"""An offline report of observed compiler topology, never a speedup claim."""
from html import escape
from pathlib import Path


def render_probe(report: dict, out: Path) -> None:
    rows=[]
    for name, row in report.get('records', {}).items():
        compiler=row.get('compiler', {})
        fp=row.get('fp32_cpu', {})
        quant=row.get('tflite_cpu', {})
        def score(value):
            return f"{value['correct']}/{value['n']}" if 'correct' in value else '—'
        values=[name,row.get('status','—'),score(fp),score(quant),
                compiler.get('host_operator_count','—'),compiler.get('npu_partition_count','—'),
                compiler.get('boundary_tensor_count','—')]
        rows.append('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in values)+'</tr>')
    heading=escape(report.get('status','unknown'))
    details=escape(report.get('reason',''))
    gate=report.get('baseline_gate', {})
    gate_text=escape(gate.get('reason','미확인'))
    (out/'report.html').write_text('''<!doctype html><html lang="ko"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>npu-region-repair · 외부 컴파일 검사</title>
<style>body{font:16px/1.65 system-ui,sans-serif;max-width:1160px;margin:32px auto;padding:0 20px}
h1{font-size:28px;line-height:1.4}.scroll{overflow:auto}table{border-collapse:collapse;min-width:760px;width:100%}
th,td{padding:12px;text-align:left;border-bottom:1px solid #bbb}.note{border-left:4px solid #777;padding:10px 18px}
pre{white-space:pre-wrap;overflow-wrap:anywhere}small{display:block;margin-top:24px}</style>
<h1>저장된 모델의 외부 컴파일 검사</h1><p>상태: <strong>'''+heading+'''</strong></p><p>'''+details+'''</p>
<div class="note">재학습 0회. 원래 저장된 체크포인트와 분할을 사용합니다.<br>
정확도는 CPU에서 측정합니다. Vela 출력은 컴파일 결과이며, NPU 지연시간·전력·칩 실행 정확도 측정이 아닙니다.<br>
NPU 분할 수는 컴파일 후 ethos-u 사용자 연산의 수입니다. 원래 신경망 연산자 수와 다릅니다.</div>
<h2>동일 시험셋 비교</h2><div class="scroll"><table><thead><tr><th>모델</th><th>상태</th><th>FP32 정답</th>
<th>TFLite CPU 정답</th><th>CPU 잔류 연산</th><th>NPU 분할</th><th>경계 텐서</th></tr></thead><tbody>'''
+''.join(rows)+'''</tbody></table></div><h2>원형의 구간 귀속 판정</h2><pre>'''+gate_text+'''</pre>
<p>연산 구간을 확인하지 못한 경우 결과는 미확정으로 남습니다. 누락된 값은 0이 아니라 —로 표시됩니다.
이 한 번의 예비 검사는 자연 이미지 성능, 실제 NPU 성능 또는 새로운 방법의 우위를 입증하지 않습니다.</p>
<p><a href="probe.json">원시 검사 결과 JSON</a> · <a href="inputs/preflight.json">원본 자료 확인</a></p>
<small>모델별 변환·컴파일 원본 파일과 로그는 같은 결과 폴더에 보존됩니다.</small></html>''',encoding='utf-8')
