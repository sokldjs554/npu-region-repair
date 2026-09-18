# Validation snapshot

This public submission snapshot keeps source code and compact natural-image evidence. Large per-image prediction arrays from the observed run are not duplicated in the repository.

## Fresh local checks for this submission snapshot

- `tools/check_submission.py`: **PASS** — 33 records, compiler/hardware provenance, key claims and local links checked.
- `python -m compileall -q src tools`: **PASS**.
- Compact public test suite: **104 passed, 2 skipped** in one fresh run. The pilot-only saved-model regression tests are not part of this public snapshot.
- The two skips are TensorFlow-dependent external conversion/parity tests in the current validation container. The observed CIFAR-10 result itself was produced in the returned TensorFlow 2.20.0 / Vela 5.1.0 environment and is preserved through its aggregate evidence and returned-bundle SHA-256.
- Static landing page: 33 rows rendered without JavaScript errors; desktop width had no page overflow and mobile 390px width had no page overflow in a Playwright `set_content` check.

## Evidence provenance

The natural-image result is `status=observed`, `expected_models=33`, `observed_records=33`, `errors=[]`, `external_compiler_executed=true`, and `npu_hardware_executed=false`.

The complete returned execution bundle is intentionally not stored in this compact public package. Its SHA-256 is recorded in [NATURAL_RESULTS.md](NATURAL_RESULTS.md).
