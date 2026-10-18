# GPU implementation status

Validated on 2026-10-18 with Java 25.0.1. The active Maven coordinates use
`com.nablatensor`; the parent artifact is `jax4j`. Java packages use `com.jax4j`.

## wgpu extension

The wgpu-native tensor backend is now implemented and hardware validated, using
shared core kernels and independent accelerator runtimes. The final three-module
build records **118 passed, 1 existing CPU test skipped**, including 36 wgpu and
31 Vulkan tests. See [wgpu setup and speed comparison](wgpu.md),
[validation](wgpu-validation.json), and [raw timings](wgpu-vulkan-benchmark.json).
This supersedes the earlier Vulkan-only deferral of wgpu below.

## Implemented

- Core combines the tensor API, CPU kernels, autodiff, NN, scalar recording,
  CPU replay/JIT and operation helpers. Duplicate resources were checked and
  CPU service providers merged. Core has no accelerator dependency.
- Vulkan combines tensor runtime/shaders and the scalar replay adapter in
  `jax4j-engine-vulkan`, depending only on core.
- CUDA and ROCm each combine tensor runtime and scalar replay in
  `jax4j-engine-cuda` and `jax4j-engine-rocm`, depending only on core.
  Optional profiles `cuda` and `rocm` include them in the reactor; native execution
  remains unverified.
- Optional examples contain the MNIST MLP and generic primitive/random checks.
- `Jit.compile` caches captured fused-kernel sequences by backend, shape, dtype
  and device. `Jit.compileTree` accepts Tensor pytrees and preserves structured
  outputs such as value-and-gradient records. New tensor values remain runtime
  inputs. Compilation assumes pure Java functions and fixed captured constants.
- Multi-input capture makes parameters and optimizer state compiler inputs.
  Shared expression nodes and materialized boundaries are reused. Traced
  convolution, gradient, axis-reduction and reshape operations forward real
  buffers correctly, and failed traces release owned intermediates.
- `Vmap` supports arbitrary axes, multiple mapped/shared arguments, structured
  outputs, nested maps, and primitive batching including convolution derivatives.
  It has no Java dispatch loop over examples for tensor kernels.
- Reverse-mode gradients compose with capture and leading-axis batching.
  Seeds preserve the tracing context; returned gradients survive tape cleanup.
  Legacy `Var` tapes reject nesting; `Transforms` supports nested tensor autodiff.
- Adam retains moments and performs fused updates on the parameter device.
- Vulkan reshape, slicing, stacking, fill, batched matrix multiplication and
  counter-based random generation execute on device. Uniform streams match
  CPU exactly; normal streams use an explicit float tolerance. Split/fold-in
  key semantics are the existing project mix64 algorithm, not JAX's random stream.
- GPU SPI defaults reject implicit host fallbacks for random generation,
  reshape, slice and stack. Vulkan counters report dispatches, compilations,
  allocations, transfer bytes and live/peak device allocations.
- Queued Vulkan resources are released only after completion. Synchronization
  covers the process-wide runtime across backend instances.

## Evidence

2026-10-18 extension: **82 passed, 1 skipped**, including **31 Vulkan tests** on
AMD Radeon Graphics (RADV PHOENIX). The skipped CPU case is an existing test.
An isolated source snapshot was used to avoid an IDE concurrently repopulating
workspace build outputs; its sources were checked byte-for-byte against this tree.
[Validation record](vulkan-p0-p1-validation.json) includes per-suite totals,
source checksum, commands and device assertions. The updated
[fusion benchmark](vulkan-benchmark-p0-p1.json) measured 0.621 ms fused versus
4.482 ms eager for the existing 36-operation workload, with 20 versus 720
dispatches and zero transfers in the fused timed region. This measures that
workload, not structured-region throughput or a cross-device performance budget.


Core tests cover reusable capture with changing values, captured parameters,
axis boundaries, host-read rejection, mapped gradients, structured outputs and
pinned **JAX 0.11.2** CPU fixtures. The JAX fixture generator explicitly forces CPU.

Six Vulkan hardware tests cover fused dispatch/allocation/transfer counts,
device-local shape operations, batched matmul, compiled mapped gradients,
Adam updates, PRNG parity and deferred resource release. They fail if Vulkan
is explicitly requested and unavailable. They do not initialize other accelerators.

[Measured Vulkan benchmark](vulkan-benchmark.json): AMD Radeon Graphics
(RADV PHOENIX), F32, 262,144 elements, 36 chained operations, 20 synchronized
iterations. The saved run measured 20 fused dispatches/allocations versus 720
for eager execution, with zero host transfers inside the fused timed region.
Cold capture/compilation is reported separately. This compares the current
eager and fused paths; it is not a pre-migration performance baseline or a
cross-device regression budget. Peak memory is process-wide, including warmup.

## Remaining roadmap work

The P0/P1 transformation APIs have been extended with arbitrary-axis/multi-input
batching, tensor JVP/VJP/Jacobians/higher derivatives, structured device regions,
custom derivatives, recomputation checkpoints and functional compiled Adam state.
See [the API and validation contracts](p0-p1-transforms.md) for the implementation,
primitive coverage, numerical conventions, and explicitly unsupported cases.
The earlier implementation/evidence bullets above describe the initial phase;
the new tests and pinned-JAX fixtures extend that evidence.

The full roadmap is **not complete**. Parallel device-region lowering, efficient
higher-order scan lowering, graph-aware views/matmul epilogues, allocation pooling,
selectable per-device numerical policies, and broad device-specific performance
budgets remain. P2 sharding, collectives and export remain unimplemented.
OpenCL and CUDA/ROCm restoration remain deferred. The wgpu tensor backend is
implemented; its setup, contracts and limits are documented separately.

## Maven migration

All artifacts below use group `com.nablatensor` and version `0.1.0-SNAPSHOT`.

| Previous artifacts | Destination |
| --- | --- |
| `jax4j-parent` | `jax4j` (POM) |
| `jax4j-tensor`, `jax4j-backend-cpu`, `jax4j-autodiff`, `jax4j-nn`, `jax4j-core`, `jax4j-cpu`, `jax4j-jit-cpu`, `jax4j-ops` | `jax4j-core` |
| `jax4j-backend-vulkan`, `jax4j-vulkan` | `jax4j-engine-vulkan` |
| `jax4j-examples`, `jax4j-bench`, `jax4j-validate` | `jax4j-examples` (optional) |

Original donor revisions and licenses remain in the import manifest, LICENSE,
and NOTICE. Mechanical moves are recorded in `module-moves.json`; this phase
adds original implementation changes without importing a new donor revision.
