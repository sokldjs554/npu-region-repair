# NPU Region Repair

> **제한된 회복 학습 예산에서 컴파일러 제약을 반영한 모델 구간 복구**  
> NPU 지원 제약으로 생긴 CPU fallback 구간을 관측하고, 모델을 얼마나 바꿔야 하는지 학습 예산·정확도·모델 크기·컴파일 토폴로지로 비교한 연구·개발 프로젝트입니다.

[![tests](https://github.com/sokldjs554/npu-region-repair/actions/workflows/tests.yml/badge.svg)](https://github.com/sokldjs554/npu-region-repair/actions/workflows/tests.yml) [![Compiler](https://img.shields.io/badge/compiler-Vela%205.1.0-informational)](#측정-항목) [![Hardware](https://img.shields.io/badge/NPU%20hardware-not%20measured-lightgrey)](#범위와-한계)

**공개 데모:** <https://sokldjs554.github.io/npu-region-repair/> — 저장된 33개 모델 결과를 seed, recovery budget, method별로 비교할 수 있습니다.

## 핵심 요약

- **문제:** INT8 모델인데도 Vela 컴파일 결과에서 CPU fallback 6개와 NPU partition 7개가 남았습니다.
- **방법:** 연산자 대체, 연결 구간 대체, 단일 구간 대체, 동일 크기 학생 모델을 같은 회복 학습 예산으로 비교했습니다.
- **결과:** region-wise는 host op를 0으로 만들고 파라미터를 **26.22% 감소**시켰습니다. 낮은 예산에서는 same-size student보다 평균 **+2.01%p**, 높은 예산에서는 **-0.83%p**로 이점이 사라졌습니다.
- **해석:** 한 방법의 우위를 주장하기보다 **정확도·모델 크기·적응 예산·컴파일 구조 사이의 trade-off**를 확인한 프로젝트입니다.
- **범위:** Arm Vela 컴파일 관측이며, 실제 Mobilint NPU latency/power 측정은 아닙니다.

## 왜 이런 비교를 했나

- compiler fallback을 단순 변환 실패가 아니라 **모델 구조를 다시 설계해야 하는 신호**로 봤습니다.
- operator-wise / region-wise / single-region / same-size student를 같이 둬서 **구조 변경 효과와 weight reuse 효과를 분리**했습니다.
- 동일 epoch가 아니라 MAC proxy 기반 recovery budget을 써서 서로 다른 크기의 모델을 가능한 한 같은 적응 비용에서 비교했습니다.
- 모든 checkpoint를 먼저 고정한 뒤 final test를 실행해 **test 결과를 보고 설정을 바꾸는 누수**를 막았습니다.

왜 Vela를 썼는지, 왜 3 seeds × 2 budgets인지, 실제 Mobilint NPU가 있다면 무엇부터 다시 측정할지는 [설계 판단 기록](docs/DESIGN_DECISIONS.md)에 정리했습니다.

### 5분 코드 검토 경로

`protocol.py` → `pipeline.py` → `experiment.py` → `compiler.py` → `tflite_bridge.py` → `summary.json`

각 단계에서 확인할 질문은 [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md#9-5분-코드-검토-경로)에 적었습니다.

## 문제 정의

INT8 모델이라고 해서 모든 연산이 NPU에서 실행되는 것은 아닙니다. 연산 종류와 shape가 타깃 컴파일러의 지원 범위를 벗어나면 그래프 일부가 CPU에 남고, NPU 실행 구간도 여러 조각으로 나뉠 수 있습니다.

이 프로젝트는 그 상황에서 다음 질문을 실험합니다.

> **지원되지 않는 연산을 없애기 위해 모델을 바꿀 때, 기존 가중치를 얼마나 보존해야 제한된 회복 학습 예산에서 정확도와 모델 크기를 함께 지킬 수 있는가?**

단순히 pruning/quantization을 적용해 표를 만드는 대신, **외부 컴파일러에서 실제 fallback을 확인 → 대체 전략 설계 → 같은 예산 규칙으로 회복 학습 → TFLite CPU 정확도와 컴파일 결과를 함께 비교**하는 흐름으로 구성했습니다.

## 핵심 결과

고정 CIFAR-10 실험에서 학습 시드 `17 / 29 / 43`, 회복 예산 `80 / 320`, 교사 모델 3개와 후보 모델 30개를 사용해 **33/33 모델을 학습·선택·INT8 변환·Vela 컴파일**했습니다. 실제 NPU 하드웨어는 실행하지 않았습니다.

| 관찰 | 결과 |
|---|---|
| 원형 모델의 컴파일 제약 | `host op 6` · `NPU partition 7` · 경계 텐서 12 |
| 완전 NPU 후보 | operatorwise / regionwise / small-student 모두 `host op 0` · `partition 1` |
| regionwise 모델 크기 | `7,690 → 5,674 params` (**−26.22%**) |
| 낮은 회복 예산 | regionwise − same-size student = **+2.01%p 평균**, 3개 시드 모두 양수 |
| 높은 회복 예산 | regionwise − same-size student = **−0.83%p 평균** — 이점이 사라짐 |
| regionwise vs operatorwise | 정확도는 낮지만, 구조 MACs 감소와 Vela 추정 cycle 약 **6.55% 감소** |

### 동일 예산 규칙에서의 정확도

TFLite CPU 정확도의 시드 3개 평균입니다. `±`는 세 학습 시드의 표준편차이며 통계적 신뢰구간은 아닙니다.

| 회복 예산 | Continuation | Operator-wise | **Region-wise** | Single region | Small student |
|---:|---:|---:|---:|---:|---:|
| 80 | 48.82 ± 1.07% | 46.97 ± 0.89% | **42.50 ± 1.72%** | 46.34 ± 1.25% | 40.49 ± 0.17% |
| 320 | 49.24 ± 1.35% | 48.86 ± 1.72% | **46.08 ± 1.33%** | 48.55 ± 1.48% | 46.91 ± 0.42% |

이 결과는 **“region-wise가 항상 가장 좋다”**는 결론을 지지하지 않습니다. 오히려 기존 가중치 재사용의 이점이 **적응 예산에 따라 달라진다**는 조건부 결과를 보여줍니다.

## 실험 설계

```text
CIFAR-10 고정 분할
      │
      ▼
Teacher RegionCNN ── Vela probe ──▶ CPU fallback 관측
      │
      ├─ continuation       : 구조 유지
      ├─ operatorwise       : 비지원 연산 교체, 폭 유지
      ├─ regionwise         : 연결 구간을 더 작은 블록으로 교체
      ├─ single_region      : 문제 구간 하나만 교체
      └─ small_student      : 동일 크기 학생 모델 강력 기준선
      │
      ▼
동일 MAC-proxy 적응 예산
      │
      ▼
validation 기준 checkpoint 선택
      │
      ▼
FP32 CPU → INT8 TFLite CPU → Vela compile topology
```

- **데이터셋:** CIFAR-10 공식 데이터, 고정 train subset 10,000 / validation 2,000 / official test 10,000.
- **시드:** 17, 29, 43.
- **교사 모델:** 시드당 3,000 updates.
- **적응 예산:** 기준 80 / 320 update-equivalent Conv/Linear MAC proxy.
- **모델 선택:** validation split만 사용합니다. Test label은 모든 후보 checkpoint가 고정된 뒤에만 사용합니다.
- **컴파일 경로:** TensorFlow 2.20.0 → INT8 TFLite → Vela 5.1.0, target `ethos-u55-256`.

## 측정 항목

이 프로젝트는 값의 출처를 섞지 않습니다.

| 지표 | 출처 |
|---|---|
| FP32 정확도 | PyTorch CPU |
| INT8 정확도 | TFLite `BUILTIN_REF` CPU |
| host op / NPU partition / boundaries | Vela 컴파일 그래프 분석 |
| `cycles_total`, SRAM | Vela 분석 추정값 |
| 실제 NPU latency / energy | **측정하지 않음** |

원형 모델에서 관측된 CPU 잔류 연산은 정규화 계산과 GELU 경로였습니다. 이는 특정 Arm 조건에서의 관측이며, 모든 NPU나 모빌린트 제품의 지원표로 일반화하지 않습니다.

## 결과 확인

- **결과 대시보드:** [`docs/index.html`](docs/index.html)
- **상세 실험 결과:** [`docs/NATURAL_RESULTS.md`](docs/NATURAL_RESULTS.md)
- **기계 판독용 요약:** [`docs/results/summary.json`](docs/results/summary.json)
- **모델별 결과표:** [`docs/results/records.csv`](docs/results/records.csv)
- **실험 프로토콜:** [`docs/NATURAL_STUDY.md`](docs/NATURAL_STUDY.md)
- **관련 연구 / 차별화 범위:** [`docs/RELATED.md`](docs/RELATED.md)
- **검증 기록:** [`docs/VALIDATION.md`](docs/VALIDATION.md)
- **설계 판단 기록:** [`docs/DESIGN_DECISIONS.md`](docs/DESIGN_DECISIONS.md)

공개 저장소에는 모델별 약 30 MB에 이르는 prediction dump 전체 대신 핵심 검증 자료만 남겼습니다. 전체 실행 번들의 SHA-256은 `docs/NATURAL_RESULTS.md`에 기록되어 있습니다.

## 재현 방법

### CPU 검증

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
python -m pip install -e '.[dev]'
python -m pytest -q -rs
```

### 자연 이미지 실험

```bash
python tools/run_natural_study.py --install --session runs/natural-study
```

외부 컴파일러 경로에서는 TensorFlow 2.20.0과 `ethos-u-vela==5.1.0`을 사용합니다.

## 저장소 구조

```text
src/nrr/             초기 비교 실험 + compiler bridge
src/nrr_natural/     CIFAR-10 반복 시드 실험
configs/             고정된 실험 설정
tools/               재현 가능한 실험 실행기
tests/               CPU, 데이터 분할, 변환 계약 및 guard 테스트
docs/                결과 대시보드, 프로토콜, 한계 및 관련 연구
```

## 범위와 한계

이 저장소는 CIFAR-10 SOTA benchmark가 아니라 **통제된 모델 적응 실험**을 보여주기 위한 프로젝트입니다.

- 모델은 소형 커스텀 RegionCNN이며, 여러 시드와 컴파일러 조건을 반복 비교할 수 있도록 학습 데이터는 고정 10k subset을 사용했습니다.
- 세 개의 학습 시드는 학습 변동성을 보기 위한 것이며 30,000개의 독립 테스트 샘플을 의미하지 않습니다.
- Vela cycle 값은 추정치이며 실제 latency나 power 측정값이 아닙니다.
- Mobilint SDK나 Mobilint 하드웨어를 사용하지 않았습니다.
- 고객 데이터나 고객 모델을 사용하지 않았습니다.
- 세 개 시드 결과만으로 학술적 최초성이나 통계적 우월성을 주장하지 않습니다.

이 프로젝트에서 중요한 것은 특정 방법의 우위를 주장하는 것이 아니라 **측정된 trade-off와 이를 재현하는 도구**입니다. 더 단순한 operator-wise replacement가 더 높은 정확도를 유지할 수 있다는 부정적 결과도 그대로 포함했습니다.

## 관련 프로젝트

[`npuloop`](https://sokldjs554.github.io/npuloop/)은 fake-quantization과 명시적인 integer execution의 차이를 다루고 NumPy/C++ 정수 검증을 제공합니다. `NPU Region Repair`는 컴파일러 제약 아래에서 모델 구조를 수정하고 회복 학습하는 문제에 집중한 별도 저장소이며, npuloop의 결과를 이 프로젝트의 결과처럼 재사용하지 않습니다.
