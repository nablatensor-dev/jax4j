# Array operations: batch 2

Introduced in 0.2.2. The site's pinned browser jar is unchanged. CPU, native Vulkan and CUDA are the validation targets.

## Public API

| Family | Methods |
| --- | --- |
| Comparison and logic | `Tensor.lt/le/gt/ge/eq/ne/and/or/xor(Tensor or double)`, `not()`, `isfinite()`, `isnan()` |
| Selection | `Tensor.where(mask, yes, no)` and `Tensors.where(...)`, with broadcasting |
| Elementary math | `sin()`, `cos()`, `hypot(Tensor or double)`, `pow(Tensor or double)` |
| Static indexing | `slice(axis, start, stop[, step])`, `take(indices, axis)` |
| Runtime regions | `dynamicSlice(starts, sizes...)`, `dynamicUpdateSlice(update, starts)` |
| Joining | `Tensor.concatenate(parts, axis)`, `Tensor.stack(parts, axis)`, and Tensors aliases |
| Reductions and ranking | `prod()`, `prod(axis[, keepDims])`, `argmin(axis)`, `topK(k)`, `segmentSum(ids, count)` |
| Array conveniences | `oneHot(classes)` (also Tensors alias), `diff(axis)`, `diff(order, axis)`, `roll(shift, axis)`, `pad(widths[, value])` |
| Constructors | `Tensors.eye`, `logspace`, `geomspace`, `meshgrid`, `indices` |

Binary elementwise methods broadcast like the existing arithmetic operations.
Negative axes follow existing normalization. Runtime indices are Tensor inputs,
so captured JIT plans use new index values on replay; they are not downloaded
and frozen at capture time.

## Masks, indices and bounds

This batch follows the existing F32 compute-buffer model. It does **not** add
BOOL or integer storage kernels. Comparison/predicate results are F32 0 or 1;
logical operations and where interpret zero as false and any nonzero value,
including NaN, as true. Equality with NaN is false; inequality with NaN is true.
`where` selects a value rather than multiplying masks, so an unselected NaN
cannot contaminate its selected output. It does not make derivatives of an
otherwise invalid branch mathematically well defined.

Index tensors must contain integral F32 values. Runtime take wraps negative
indices once; out-of-bounds, fractional, NaN and infinite indices yield NaN.
The selected dimension cannot exceed 2^24, the exact consecutive-integer range
used by this library's F32 index convention. Take output shape inserts all index
tensor dimensions at the selected axis, replacing that axis. Repeated indices
accumulate in its reverse derivative.

Slices use Python-style explicit integer start/stop bounds, exclusive stop and
nonzero positive or negative step. Negative bounds count from the end. There is
no omitted-bound sentinel: to reverse a length-N axis, use
`slice(axis, N-1, -N-1, -1)`.

Dynamic start vectors contain one index per input dimension. Slice sizes and
update shapes are static and cannot exceed the input. Integral start positions
are clamped to [0, inputExtent-regionExtent]. Negative starts clamp to zero;
this is deliberately different from take's negative-index wrapping. Fractional
or nonfinite starts yield all-NaN slices or leave an update's base unchanged.
Region dimensions are limited to 2^24. Updates are immutable and return a new
tensor. Their gradients route to the selected update region and the untouched
base region; start indices have zero derivatives.

Segment IDs are a vector matching data's leading dimension. The result has
shape [segmentCount, ...remainingDataDimensions]. Empty segments are zero.
Negative, out-of-range, fractional and nonfinite IDs are dropped. One-hot
appends a class dimension; invalid labels produce all-zero rows. Class and
segment counts are nonnegative and limited to 2^24.

## Reduction and array conventions

Products use multiplicative identity one for empty reduced axes. Their
first-order derivative multiplies the other entries, so one or several zero
entries behave correctly without division by zero.

TopK operates on the last dimension, returning a `Tensor.TopK(values, indices)`
record with both result tensors owned by the caller. Values are descending;
NaNs sort first and ties prefer the smaller original index. Indices are F32.
K may be zero, but cannot exceed the last-axis length. `argmin` composes
negation with existing argmax and inherits that backend's argmax NaN policy;
it is not a new general sorting primitive.

Diff defaults to order one; order zero returns an independent copy. Roll wraps
around the chosen axis. Pad accepts nonnegative [before, after] pairs per axis
and a constant fill (zero by default). Concatenation joins an existing axis;
stacking inserts a new axis. Empty lists and incompatible shapes/devices fail.

Eye supports rectangular shapes and a diagonal offset. Logspace includes both
endpoints with a positive finite base (default 10); geomspace includes both
endpoints and accepts finite nonzero endpoints of the same sign, including
negative pairs. Count zero is empty and count one returns the start value.
These constructors generate host data and upload once, as arange/linspace do.

Meshgrid returns a caller-owned `List<Tensor>`, defaults to matrix indexing
`ij`, and accepts `xy` plus a sparse flag. Sparse grids retain singleton axes;
dense grids broadcast. Axes must be rank-one tensors on the same device.
`indices(device, dims...)` returns one dense coordinate tensor per dimension.
No dimensions yields an empty list. Close grids, TopK leaves and other result
trees with `TreeUtil.closeAll`; inputs remain borrowed and usable.

## Execution and derivative coverage

Elementwise operations execute eagerly or fused on CPU/Vulkan/CUDA. Static
slice, roll and pad reuse existing gather/scatter metadata. CUDA now has those
static gather/scatter implementations as well. Array kernels handle true
selection, runtime indexing/update, concatenation, product, topK, oneHot and
segment sums without host reads. Their Vmap rule batches on the device.
CPU and Vulkan region lowering also supports them inside structured scan.
CUDA's pre-existing device scan restriction remains unchanged.

First-order JVP and reverse derivatives are implemented for array-kernel data
inputs. Discrete masks, index inputs and ranking indices have zero derivatives.
Higher-order differentiation through these array kernels fails explicitly with
UnsupportedOperationException. Static gather/scatter conveniences retain their
existing transform rules. Elementwise powers use the real domain; differentiating
with respect to an exponent at a negative base has no real-valued log derivative.
Hypot's derivative at the origin is undefined. New APIs do not promise full JAX
transform or dtype parity.

The deterministic reference-style GPU ranking kernel computes ranks with
pairwise comparisons; its cost is quadratic in row width for each selected
rank. Scatter/segment accumulation and generic first-order derivative kernels
also use explicit scans, not optimized sorting, atomics or segmented-reduction
algorithms. They establish functional device-local support, not a performance
claim for large arrays.

ROCm's elementwise paths compile, but runtime array kernels are not implemented
there and fail explicitly. Browser/native WebGPU hardware support is unverified;
the browser bridge was not extended. No silent GPU-to-CPU fallback is used.

## Validation

Validated 2026-11-05 with Java 25.0.1. Final full-suite results:

| Machine/backend | Passed | Skipped | Failures/errors |
| --- | ---: | ---: | ---: |
| Local CPU/core | 79 | 1 | 0 |
| Local Vulkan (AMD Radeon Graphics, RADV PHOENIX) | 62 | 0 | 0 |
| zbox CPU/core | 79 | 1 | 0 |
| zbox CUDA (GTX 1060) | 48 | 0 | 0 |

The local CUDA/ROCm modules also compile. WebGPU hardware tests were not enabled
(2 non-hardware tests passed, 34 were skipped); ROCm hardware was not tested.

Batch2Test runs the same core contract fixtures on CPU, Vulkan and CUDA:
comparison/broadcast semantics, unselected NaNs, borrowed-input survival, math
references, slices, join shapes, runtime index replay, repeated-index gradients,
clamped updates, zero-aware products, topK ties/NaNs, empty and invalid labels,
segment sums, grids, padding and first-order derivatives. CPU/Vulkan additionally
exercise selection inside scan; mapped selection and its gradient run on all
three. Vulkan diagnostics check zero host transfers during selection/indexing.

Local Java 25:

```sh
JAVA_HOME=/opt/zulu25.30.17-ca-jdk25.0.1-linux_x64 \
  /opt/apache-maven/bin/mvn -Pcuda,rocm -Djax4j.vulkan.tests=true test
```

On zbox (NVIDIA GTX 1060, driver 580.178.04), sources are isolated under
`/tmp/jax4j-batch2-20261105/jax4j`; Java/Maven tooling is reused from the earlier
batch-1 temporary directory:

```sh
mvn -Pcuda -pl jax4j-engine-cuda -am clean test -Djax4j.cuda.tests=true
```

CUDA has no public transfer-diagnostics API matching Vulkan's; tests require the
CUDA backend and verify results/device placement, without claiming measured
CUDA transfer counts. No release, deployment, or automatic commit is performed.
