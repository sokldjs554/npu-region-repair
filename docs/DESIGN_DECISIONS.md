# NPU Region Repair 설계 판단 기록

이 문서는 결과표보다 **왜 이런 비교를 설계했는지**를 설명합니다.

## 1. 왜 compiler fallback을 모델 연구 문제로 봤나

INT8 변환이 성공해도 모든 연산이 NPU에서 실행되는 것은 아닙니다. compiler가 지원하지 않는 연산은 host에 남고, 그 경계 때문에 실행 graph가 여러 partition으로 나뉠 수 있습니다.

그래서 문제를 단순히 "변환 성공/실패"로 두지 않고:

**compiler fallback 관측 → 문제 구간 식별 → 모델 구조 변경 → recovery training → 정확도/크기/topology 재비교**

로 다시 정의했습니다.

## 2. 왜 네 가지 대안을 비교했나

한 가지 repair 방법만 실행하면 "좋아졌다"는 결과는 얻을 수 있어도 **왜 좋아졌는지**를 구분하기 어렵습니다.

- **continuation**: 구조를 바꾸지 않았을 때의 기준
- **operator-wise**: 미지원 연산만 최소 변경
- **region-wise**: 연결된 문제 구간을 더 작은 block으로 교체
- **single-region**: 일부 구간만 고쳤을 때의 중간 대조
- **same-size student**: region-wise와 크기는 같지만 기존 weight를 재사용하지 않는 강한 baseline

특히 same-size student를 넣은 이유는 region-wise의 이점이 **작은 모델 자체 때문인지, 기존 weight reuse 때문인지** 분리하기 위해서입니다.

## 3. 왜 동일 recovery budget을 썼나

구조마다 한 step의 계산량이 다르기 때문에 단순히 같은 epoch 수로 비교하면 작은 모델이 더 적은 계산으로 유리하거나, 반대로 더 많은 update를 받아 유리해질 수 있습니다.

그래서 Conv/Linear MAC proxy로 budget을 맞추고, 같은 batch stream을 공유하도록 했습니다.

코드 시작점:
- `src/nrr_natural/pipeline.py`
- `src/nrr/budget.py`
- `src/nrr/training.py`

## 4. 왜 3 seeds × 2 budgets인가

한 seed의 결과만으로 weight reuse 효과를 말하면 초기화와 학습 노이즈에 의존할 수 있습니다.

두 budget을 둔 이유도 동일합니다. **짧은 적응 예산에서 유리한 방법이 충분히 학습시키면 계속 유리한지** 확인하고 싶었습니다.

실제로 region-wise는 same-size student 대비:
- budget 80: 평균 **+2.01%p**
- budget 320: 평균 **-0.83%p**

로 방향이 뒤집혔습니다.

이 때문에 "region-wise가 항상 우수하다"는 결론을 내리지 않았습니다.

## 5. 왜 checkpoint를 전부 고정한 뒤 test를 봤나

후보별 test 결과를 보면서 checkpoint나 학습 설정을 바꾸면 test leakage가 생깁니다.

그래서 pipeline은:

1. teacher/candidate 학습
2. validation 기준 선택
3. 모든 candidate checkpoint freeze
4. hash 확인
5. 그 이후에만 final test evaluation

순서를 코드에서 강제합니다.

코드 시작점:
- `src/nrr_natural/pipeline.py`
- `src/nrr_natural/protocol.py`

## 6. 왜 Arm Vela를 사용했나

Mobilint NPU를 흉내 내기 위해 선택한 것이 아닙니다.

접근 가능한 외부 NPU compiler를 이용해 **operator support와 partition이라는 실제 compiler 제약이 모델 설계에 어떤 질문을 만드는지** 재현하기 위해 사용했습니다.

따라서 Vela의 cycle/SRAM 추정치를 Mobilint 성능으로 일반화하지 않습니다.

## 7. 왜 region-wise를 최종 승자로 부르지 않나

region-wise는:
- host op 6 → 0
- partition 7 → 1
- parameters -26.22%
- operator-wise 대비 Vela estimated cycles 약 -6.55%

라는 장점이 있었습니다.

하지만 정확도는 operator-wise가 더 높았습니다. 즉 이 결과는 **정확도 하나의 승자**가 아니라 model size·compiler topology·recovery budget 사이의 trade-off입니다.

## 8. 실제 Mobilint NPU가 있다면 다음 실험

1. 동일 모델을 qb Compiler로 변환해 실제 unsupported/fusion boundary 확인
2. operator-wise / region-wise를 동일 calibration set으로 quantize
3. qb Runtime에서 latency·throughput·memory 반복 측정
4. 실제 partition/fallback 비용과 Vela에서 본 구조적 신호가 같은 방향인지 비교
5. budget별 recovery accuracy와 실제 NPU latency를 Pareto frontier로 정리
6. 고객 모델에서는 데이터 분포 변화까지 포함해 calibration/repair rule 재평가

## 9. 5분 코드 검토 경로

| 질문 | 먼저 볼 파일 | 확인할 내용 |
|---|---|---|
| 비교 조건을 어떻게 고정했나 | `src/nrr_natural/protocol.py` | seeds, budgets, split |
| 전체 실험 순서는 어떻게 강제했나 | `src/nrr_natural/pipeline.py` | train → freeze → test |
| 모델 대안은 어디서 정의하나 | `src/nrr/experiment.py` | repair methods |
| compiler 결과는 어떻게 읽나 | `src/nrr/compiler.py` | Vela invocation/topology |
| TFLite 정확도는 어디서 재나 | `src/nrr/tflite_bridge.py` | export/reference CPU |
| 공개 숫자의 원본은 무엇인가 | `docs/results/summary.json` | 33-model aggregate |
| 무엇을 주장하지 않는가 | `docs/VALIDATION.md` | hardware/customer-data boundary |

## 한 문장으로 정리

**컴파일러 제약을 모델 구조와 학습 예산의 문제로 되돌리고, 어떤 repair가 어떤 조건에서 유효한지 비교한 프로젝트입니다.**
