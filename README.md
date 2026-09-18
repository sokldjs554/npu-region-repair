# NPU Region Repair

> **Compiler-guided model repair under a limited adaptation budget**  
> NPU 지원 제약으로 생긴 CPU fallback 구간을 관측하고, 모델을 얼마나 바꿔야 하는지 학습 예산·정확도·모델 크기·컴파일 토폴로지로 비교한 연구·개발 프로젝트입니다.

[![tests](https://github.com/sokldjs554/npu-region-repair/actions/workflows/tests.yml/badge.svg)](https://github.com/sokldjs554/npu-region-repair/actions/workflows/tests.yml) [![Compiler](https://img.shields.io/badge/compiler-Vela%205.1.0-informational)](#what-was-measured) [![Hardware](https://img.shields.io/badge/NPU%20hardware-not%20measured-lightgrey)](#scope-and-limits)

![NPU Region Repair result dashboard](docs/preview.png)

## Why this project

INT8 모델이라고 해서 모든 연산이 NPU에서 실행되는 것은 아닙니다. 연산 종류와 shape가 타깃 컴파일러의 지원 범위를 벗어나면 그래프 일부가 CPU에 남고, NPU 실행 구간도 여러 조각으로 나뉠 수 있습니다.

이 프로젝트는 그 상황에서 다음 질문을 실험합니다.

> **지원되지 않는 연산을 없애기 위해 모델을 바꿀 때, 기존 가중치를 얼마나 보존해야 제한된 회복 학습 예산에서 정확도와 모델 크기를 함께 지킬 수 있는가?**

단순히 pruning/quantization을 적용해 표를 만드는 대신, **외부 컴파일러에서 실제 fallback을 확인 → 대체 전략 설계 → 같은 예산 규칙으로 회복 학습 → TFLite CPU 정확도와 컴파일 결과를 함께 비교**하는 흐름으로 구성했습니다.

## Headline results

고정 CIFAR-10 실험에서 학습 시드 `17 / 29 / 43`, 회복 예산 `80 / 320`, 교사 3개와 후보 30개를 사용해 **33/33 모델을 학습·선택·INT8 변환·Vela 컴파일**했습니다. 실제 NPU 하드웨어는 실행하지 않았습니다.

| 관찰 | 결과 |
|---|---|
| 원형 모델의 컴파일 제약 | `host op 6` · `NPU partition 7` · 경계 텐서 12 |
| 완전 NPU 후보 | operatorwise / regionwise / small-student 모두 `host op 0` · `partition 1` |
| regionwise 모델 크기 | `7,690 → 5,674 params` (**−26.22%**) |
| 낮은 회복 예산 | regionwise − same-size student = **+2.01%p 평균**, 3개 시드 모두 양수 |
| 높은 회복 예산 | regionwise − same-size student = **−0.83%p 평균** — 이점이 사라짐 |
| regionwise vs operatorwise | 정확도는 낮지만, 구조 MACs 감소와 Vela 추정 cycle 약 **6.55% 감소** |

### Accuracy under the same budget rule

TFLite CPU 정확도의 시드 3개 평균입니다. `±`는 세 학습 시드의 표준편차이지 통계적 신뢰구간이 아닙니다.

| 회복 예산 | Continuation | Operator-wise | **Region-wise** | Single region | Small student |
|---:|---:|---:|---:|---:|---:|
| 80 | 48.82 ± 1.07% | 46.97 ± 0.89% | **42.50 ± 1.72%** | 46.34 ± 1.25% | 40.49 ± 0.17% |
| 320 | 49.24 ± 1.35% | 48.86 ± 1.72% | **46.08 ± 1.33%** | 48.55 ± 1.48% | 46.91 ± 0.42% |

이 결과는 **“region-wise가 항상 가장 좋다”**는 결론을 지지하지 않습니다. 오히려 기존 가중치 재사용의 이점이 **적응 예산에 따라 달라진다**는 조건부 결과를 보여줍니다.

## Experiment design

```text
CIFAR-10 fixed split
      │
      ▼
Teacher RegionCNN ── Vela probe ──▶ CPU fallback observed
      │
      ├─ continuation       : structure unchanged
      ├─ operatorwise       : unsupported ops replaced, width preserved
      ├─ regionwise         : connected regions replaced with smaller blocks
      ├─ single_region      : one problematic region replaced
      └─ small_student      : same-size student trained as a strong baseline
      │
      ▼
Same MAC-proxy adaptation budget
      │
      ▼
validation checkpoint selection
      │
      ▼
FP32 CPU → INT8 TFLite CPU → Vela compile topology
```

- **Dataset:** CIFAR-10 official data, fixed train subset 10,000 / validation 2,000 / official test 10,000.
- **Seeds:** 17, 29, 43.
- **Teacher:** 3,000 updates per seed.
- **Adaptation budgets:** baseline 80 / 320 update-equivalent Conv/Linear MAC proxy.
- **Model selection:** validation split only. Test labels are used after every candidate checkpoint is frozen.
- **Compiler:** TensorFlow 2.20.0 → INT8 TFLite → Vela 5.1.0, target `ethos-u55-256`.

## What was measured

이 프로젝트는 값의 출처를 섞지 않습니다.

| Metric | Source |
|---|---|
| FP32 accuracy | PyTorch CPU |
| INT8 accuracy | TFLite `BUILTIN_REF` CPU |
| host op / NPU partition / boundaries | Vela-compiled graph inspection |
| `cycles_total`, SRAM | Vela analytical estimates |
| actual NPU latency / energy | **not measured** |

원형에서 관측된 CPU 잔류 연산은 정규화 계산과 GELU 경로였습니다. 특정 Arm 조건에서의 관측이며, 모든 NPU나 모빌린트 제품의 지원표로 일반화하지 않습니다.

## Explore the results

- **Result dashboard:** [`docs/index.html`](docs/index.html)
- **Detailed experiment result:** [`docs/NATURAL_RESULTS.md`](docs/NATURAL_RESULTS.md)
- **Slim machine-readable evidence:** [`docs/results/aggregate.json`](docs/results/aggregate.json)
- **Per-model table:** [`docs/results/records.csv`](docs/results/records.csv)
- **Protocol:** [`docs/NATURAL_STUDY.md`](docs/NATURAL_STUDY.md)
- **Prior compiler probe:** [`docs/COMPILER_OBSERVED.md`](docs/COMPILER_OBSERVED.md)
- **Related work / novelty boundary:** [`docs/RELATED.md`](docs/RELATED.md)
- **Validation snapshot:** [`docs/VALIDATION.md`](docs/VALIDATION.md)

The public repository intentionally keeps compact evidence instead of the 30 MB per-image prediction dump. The complete returned execution bundle is preserved separately with its SHA-256 in `docs/NATURAL_RESULTS.md`.

## Reproduce

### CPU checks

```bash
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0
python -m pip install -e '.[dev]'
python -m pytest -q -rs
```

### Natural-image study

```bash
python tools/run_natural_study.py --install --session runs/natural-study
```

The notebook version is [`notebooks/natural_image_study.ipynb`](notebooks/natural_image_study.ipynb). The external compiler path uses TensorFlow 2.20.0 and `ethos-u-vela==5.1.0`.

### Saved-model compiler probe

```bash
python tools/run_external_probe.py --install --session runs/external-probe
```

This probe uses the compact pilot artifacts under `verification/pilot-final-seed17/`. It does not retrain them before compilation.

## Repository layout

```text
src/nrr/             first comparison + compiler bridge
src/nrr_natural/     CIFAR-10 repeated-seed study
configs/             frozen experiment configuration
notebooks/           self-contained Colab/CPU execution notebooks
tools/               repeatable study launchers
tests/               CPU, data split, conversion contract and guard tests
docs/                result dashboard, protocol, limits and related work
verification/        compact pilot evidence required by tests/probe
```

## Scope and limits

This repository demonstrates a **controlled model-adaptation experiment**, not a CIFAR-10 SOTA benchmark.

- The model is a small custom RegionCNN and the training set is a fixed 10k subset to make multi-seed/compiler comparison tractable.
- Three training seeds describe training variation; they are not 30,000 independent test samples.
- Vela cycle values are estimates, not real latency or power.
- No Mobilint SDK or Mobilint hardware was used.
- No customer data/model was used.
- The project does not claim academic firstness or statistical superiority from three seeds.

The useful result is the **measured trade-off and the tooling that reproduces it**, including the negative finding that a simpler operator-wise replacement can retain higher accuracy.

## Related project

[`npuloop`](https://sokldjs554.github.io/npuloop/) studies the separate question of fake-quantization versus explicit integer execution and provides NumPy/C++ integer verification. `NPU Region Repair` is a new repository focused on model modification and recovery training under compiler constraints; it does not reuse npuloop's results as its own.
