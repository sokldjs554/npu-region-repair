# Related work and novelty boundary

This project does **not** claim that model replacement, knowledge distillation, compiler-aware optimization, or latency-constrained search are new ideas. The contribution being tested is narrower: **after observing a real compiler fallback region, compare operator-wise replacement, connected-region replacement, and a same-size student under the same declared adaptation-budget rule, then verify both accuracy and compiler topology.**

## Closest references

### LANA — Latency Aware Network Acceleration (ECCV 2022)

LANA learns layer replacement candidates with knowledge distillation and searches combinations under latency constraints. Therefore, "replacement + distillation + budget" is not a new combination introduced here.

This project differs in experimental question and evidence: candidate methods are frozen before the natural-image study; the trigger is an observed mixed CPU/NPU compiler graph; and the comparison includes a same-size student and operator-wise replacement rather than assuming region replacement is preferable.

- https://research.nvidia.com/publication/2022-10_lana-latency-aware-network-acceleration

### Arm Vela

Vela compiles quantized TFLite graphs for Ethos-U and leaves unsupported work outside compiled NPU regions. It is used here as an **external compiler observation**, not as a model of Mobilint hardware and not as a hardware latency measurement.

- https://arm-software.github.io/CMSIS-Ethos-U/main/vela/index.html

### TensorFlow Lite integer quantization

The external bridge uses TensorFlow 2.20.0 full-integer TFLite conversion and CPU reference execution. TFLite accuracy and Vela graph partitioning are recorded as separate evidence.

- https://ai.google.dev/edge/litert/models/post_training_integer_quant

### LiteRT Quantization Debugger

Layer-level quantization error analysis and mixed execution concerns already exist in deployment tooling. This repository does not claim layer inspection itself as a novel method.

- https://ai.google.dev/edge/litert/models/quantization_debugger

## What the CIFAR-10 study actually adds

The observed study produced three useful findings within its fixed scope:

1. **Fallback removal is not unique to region replacement.** Operator-wise replacement, region-wise replacement and the same-size student all reached `host op 0 / NPU partition 1` in the chosen Vela target.
2. **Region replacement trades accuracy for size/cost.** It reduced parameters from 7,690 to 5,674 and lowered Vela estimated cycles relative to operator-wise replacement, but operator-wise replacement retained higher accuracy.
3. **Weight reuse is budget-dependent.** At budget 80, region-wise replacement exceeded the same-size student in all three seeds (+2.01%p mean); at budget 320 the mean difference reversed to −0.83%p.

These are conditional observations from one custom small CNN, one dataset setup and three initialization seeds. They are not a proof of general superiority or academic novelty.
