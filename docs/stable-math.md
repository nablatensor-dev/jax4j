# Stable math and statistical reductions (batch 1)

Added to the working checkout on 2026-11-05. These methods are not available in
previously published jars until a new release is published.

## API

- Elementwise `square()`, `log1p()`, `expm1()`, `softplus()`.
- `mean(axis)` and `mean(axis, keepDims)` complement existing whole-array `mean()`.
- `var()` and `std()` reduce the whole tensor. Axis forms accept `(axis)`,
  `(axis, keepDims)`, or `(axis, correction, keepDims)`.
- `logsumexp()` reduces the whole tensor; `(axis)` and `(axis, keepDims)` reduce
  one dimension.
- `softmax(axis)` and `logSoftmax(axis)` normalize along one dimension without
  changing the input shape.

Negative axes follow the existing reduction rules. Whole-array results retain
existing shape [1]; axis reductions drop the axis unless keepDims is true.
Correction defaults to zero (population variance); correction one uses N-1.
Negative correction is rejected. Empty means/variances and variances whose
correction makes the divisor nonpositive produce NaN. Standard deviation is
sqrt(variance); its derivative at zero is not generally defined.

Variance uses a centered two-pass calculation, not E[x²]-E[x]². F32 input,
reduction and division rounding still apply: a constant field can have a small
nonzero residual variance when its computed mean rounds. This does not promise
arbitrary-precision or overflow-free statistics over every finite F32 input.

## Numerical implementations

CPU uses Math.log1p/Math.expm1. CUDA uses log1pf/expm1f. GLSL lacks these
intrinsics: Vulkan uses a degree-12 log1p series for |x|<1/8 and a degree-10
expm1 series for |x|<1/2, evaluated in Horner form; outside those intervals it
uses log/exp. expm1 preserves signed zero. Softplus uses
max(x,0)+log1p(exp(-abs(x))), avoiding overflow for large positive x.

Logsumexp shifts by the maximum, clamped to finite F32 bounds so all-negative-
infinity input yields negative infinity and positive-infinity input yields
positive infinity. NaNs propagate. Empty logsumexp is negative infinity.
Softmax/logSoftmax shift by the unmodified maximum; infinite/NaN logits can
produce NaN, following ordinary unmasked normalization semantics. Finite
extreme wrong-class logits have finite logSoftmax even when softmax underflows.
No masking API is introduced here.

All operations preserve device placement. The new primitives have forward- and
reverse-mode derivative rules, fused/JIT support, and existing region lowering
support on CPU/Vulkan. CUDA device scan remains unsupported and is tested as an
explicit error; ordinary eager/JIT/autodiff operations work on CUDA. Browser
WebGPU coverage was not tested or added in this batch. ROCm compiles but was not
hardware-tested. New tensors and transform result leaves remain caller-owned.

## Reproduction

Local Java 25:

```sh
JAVA_HOME=/opt/zulu25.30.17-ca-jdk25.0.1-linux_x64 \
  /opt/apache-maven/bin/mvn -Pcuda,rocm -Djax4j.vulkan.tests=true test
```

zbox: NVIDIA GeForce GTX 1060, driver 580.178.04; isolated source/tooling under
`/tmp/jax4j-batch1-20261105`, Java 25.0.1:

```sh
mvn -Pcuda -pl jax4j-engine-cuda -am clean test -Djax4j.cuda.tests=true
```

Final full-suite results: local CPU/core 63 passed and 1 skipped; Vulkan 45 passed
with no skips; zbox CPU/core 63 passed and 1 skipped; CUDA 33 passed with no
skips. All suites had zero failures/errors. Local WebGPU hardware tests were
not enabled (2 non-hardware tests passed and 34 were skipped). CUDA and ROCm
modules also compile locally; ROCm hardware was not exercised.

StableMathTest covers independent double references, near-zero relative
accuracy, approximation boundaries, signed zero, extreme inputs, nonfinite and
empty reductions, axis/correction semantics, analytic gradients/JVPs, finite
differences, tied logits, replay with fresh inputs, CPU/Vulkan scan, and CUDA's
explicit scan limitation. Vulkan diagnostics assert zero host transfers during
operations. CUDA has no matching public transfer diagnostics; its tests require
the CUDA backend and check device outputs, without claiming measured transfer
counts. Empty CUDA buffers now avoid zero-byte allocations and zero-work
launches. Empty-axis broadcasting and scalar reduction cotangents were also
corrected; fused CPU SIGN support was completed for reduction gradients.

## Next: batch 2 (not implemented)

- Comparisons and masks: <, >, equality, logical operations, where, isfinite, isnan.
- Indexing: slicing, take, concatenation, dynamic slice/update.
- Other math: sin, cos, hypot, general powers.
- Constructors: meshgrid, indices, eye, logspace, geomspace.
- Other reductions: prod, argmin, top-K, segment reductions.
- Other conveniences: oneHot, diff, roll, pad, general-axis stacking.
