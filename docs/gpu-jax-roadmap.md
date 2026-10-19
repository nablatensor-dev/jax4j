# JAX4J: GPU restoration, JAX features, and module consolidation

Status: Vulkan-first implementation in progress; source audit dated 2026-10-17.
Update: [wgpu-native tensor execution and measured Vulkan comparison](wgpu.md)
are now implemented and validated; earlier source inventory entries remain
historical context.
P0/P1 tensor API extensions and Vulkan validation dated 2026-10-18 are described
in [the implementation contracts](p0-p1-transforms.md). Donor/source inventory
and initial-state descriptions below are historical context, not current API limits.
See [implementation status](implementation-status.md) for completed work, measured
Vulkan evidence and remaining requirements. CUDA and ROCm are excluded from the
active reactor and have not been run. Future accelerator plans below are retained
as design targets, not implementation claims.

## Direction and mandatory requirements

Build a general Java 25 tensor, differentiation, and compilation library with a
small module structure. Preserve the fluent chained Tensor API. **Chained
operations MUST remain visible to the compiler and MUST remain fusible on GPUs.**
A migration that replaces a fused chain with eager per-operation launches,
host round trips, or implicit CPU execution is a performance regression and must
not be accepted merely because numerical tests pass.

The objective is uncompromised GPU throughput subject to explicit numerical
semantics. No architecture can promise maximum speed on every workload; prove
performance on the target devices. Prefer the fastest validated kernel plan,
which can include several kernels where a single giant kernel would spill
registers or reduce occupancy.

## Audited sources and provenance

Paths refer to this PC. Pin each imported change to its donor commit and record
original/destination paths, modifications, license headers, and validation in
[the migration manifest](migration-manifest.json). Preserve LICENSE and NOTICE.
Do not infer source equivalence from matching artifact names.

| Source | Audited revision / location | Role |
| --- | --- | --- |
| JAX4J | 32b70d2ca8916bb9675bcadb9368556011c5d0f7; ${HOME}/Projects/lowprio/jax4j | Current target; tensor API, CPU backend, autodiff, NN, policies |
| Current NablaTensor | 26006a665b074d4e231c8dfa405467afa90bc595; ${HOME}/Projects/highprio/nablatensor | Preferred donor for newer GPU runtimes, shared kernels, replay improvements, module consolidation pattern |
| TensorJIT / JAX4J | 22f83155fc666ac2d4172200963fcb6da4b94406; ${HOME}/Projects/highprio/nablatensor-old | General tensor transformations, CPU/autodiff lineage, NN and GPU examples |
| SonicTensor | ${HOME}/.m2/repository/com/sonictensor; 0.1.0-SNAPSHOT | Historical binary API reference; tensor snapshot metadata updated 2026-08-30 13:41:30 UTC |
| Original Python JAX | Official documentation linked below; pin a JAX version for parity fixtures | Behavioral specification and independent oracle, not Java source to copy directly |

See [ecosystem and migration](ecosystem-and-migration.md) for the existing source
snapshot provenance. SonicTensor source JARs were not present in the earlier
audit; JAX4J was reconstructed from NablaTensor's initial snapshot plus TensorJIT
sources. Do not add a dependency on the cached SonicTensor JARs.

## What exists now and where to import improvements

The default reactor now contains consolidated core and Vulkan engine modules.
Examples are optional. Other accelerator sources are preserved outside
the active reactor. Compilation alone is not evidence of device execution.

Donor paths below are relative to the corresponding audited repository root.

| Capability | JAX4J today | Preferred donor and concrete paths | Integration work / limitations |
| --- | --- | --- | --- |
| CUDA tensor execution | Present in jax4j-engine-cuda; optional cuda profile | NablaTensor nablatensor-engine-cuda/src/main/java/com/nablatensor/backend/cuda/: CudaBackend, CudaBuffer, CudaJit, CudaRuntime | Diff against JAX4J; port compatible runtime and compiler fixes; adapt enums/SPI; validate fused tensor and gradient execution on NVIDIA |
| ROCm/HIP tensor execution | Present in jax4j-engine-rocm; optional rocm profile | NablaTensor nablatensor-engine-rocm/.../backend/rocm/: HipRuntime, HipCompute, HipBuffer, RocmBackend | Port compatible runtime/kernel fixes, capability checks and availability gating; validate on supported AMD hardware |
| Vulkan tensor execution | Present in jax4j-backend-vulkan; legacy profile | NablaTensor nablatensor-engine-vulkan/.../backend/vulkan/: VulkanRuntime, VulkanCompute, VulkanBuffer, VulkanBackend, VulkanShader, VulkanShaders | Port shader/runtime fixes with matching SPI; verify tensor operation coverage, buffer barriers and resource lifetimes; replay shaders alone do not provide general tensors |
| wgpu-native / WGSL | No equivalent JAX4J module | NablaTensor nablatensor-engine-wgpu/.../backend/wgpu/: WgpuRuntime, WgpuCompute; .../engine/wgpu/: WgpuAadCodegen, WgpuAadEngine, WgpuAadKernel | Reuse native runtime and dispatch patterns. Implement JAX4J DeviceBuffer, ComputeBackend, tensor WGSL generation and fused-expression lowering. The donor is a scalar tape replay engine, not a complete tensor backend |
| OpenCL | No dedicated JAX4J module | NablaTensor nablatensor-engine-opencl; OpenClAadCodegen and runtime/engine code | Optional later compatibility backend; requires tensor lowering and SPI integration, not just copying tape replay |
| Shared GPU tensor kernels | JAX4J has earlier tensor/backend infrastructure | NablaTensor nablatensor-tensor/.../spi/: GpuKernel, GpuKernels, GpuLaunch, DevicePrelude, ElementwiseKernels, MatmulKernels, ReductionKernels, RandomKernels, ConvKernels, BroadcastLayout, AxisReduction | Port a coherent dependency set; audit dtype, layout and broadcast semantics; do not mix old and new opcodes by ordinal |
| Scalar tape replay | Existing jax4j-core and cuda/rocm/vulkan replay modules | NablaTensor nablatensor-core: GpuAadExecutable, AadCheckpointPlan, CudaAadCodegen; accelerator engine modules | Useful for checkpoint/replay techniques. Scalar tape adjoints are not a replacement for tensor autodiff or a general tensor graph |
| Tensor autodiff | AutoDiff, Tape, Var already copied | TensorJIT tensorjit-autodiff; compare upstream fixes and tests | Preserve existing behavior, then add compiler-visible backward graphs and GPU-resident optimizer updates |
| JIT surface / batching | JAX4J has tensor/Jit, FusingBackend, ExprBuffer, expr/Expr | TensorJIT tensorjit-transform: Transforms, JitFunction, VmapFunction; tensorjit-core Jit/FusingBackend | Import useful API patterns selectively. Old VmapFunction loops over slices; replace with batching rules before calling it accelerated vectorization |
| NN / training helpers | Dense, masked pooling, CPU Adam, policy inference | TensorJIT tensorjit-nn and tensorjit-examples/MnistTrainGpu.java | Audit layers and optimizers individually; CPU Adam is not a GPU training step; keep updates on device |
| Linalg, tree structures, PRNG | Existing tensor Linalg, tree utilities, PrngKey | Compare TensorJIT tensorjit-core and current NablaTensor nablatensor-tensor | Reuse compatible improvements; establish dtype, shape and random-key contracts before extending |

Native wgpu here means wgpu-native through Java FFM. Browser WebGPU needs its own
host integration; it does not follow automatically from this import. CUDA, HIP,
Vulkan and WGSL have different precision and feature limits: expose capabilities
explicitly, especially fp64 and reduction support.

## Keep operation chains: the GPU performance contract

Preserve both readable method chaining and its underlying expression graph.
Fluent syntax alone is insufficient if every method immediately dispatches a kernel.
An existing example of the intended capture boundary is:

~~~java
Tensor y = Jit.apply(x -> x.mul(x).add(3.0).tanh(), input);
~~~

For a supported, same-shape elementwise chain, capture the complete expression
and emit one fused kernel with inputs and one final output. Keep intermediates
in registers where practical. Temporary named Java variables and helper/layer
calls must not end graph capture. Captured parameter tensors must be represented
as device inputs with correct ownership; they must not trigger hidden downloads.

Current JAX4J Jit.apply wraps one input in FusingBackend, traces the body each call,
and asks the real backend to materialize the final expression. Backend compiled
kernel caching is distinct from caching the traced execution plan. FusingBackend
currently materializes at operations including matmul, transpose, reductions,
slicing, stacking and broadcasting. Some other methods delegate directly; their
handling of traced inputs needs an explicit audit. This is a starting point,
not a whole-program compiler or proof that arbitrary NN/autodiff compositions fuse.

Required implementation properties:

1. Represent dependencies in a typed graph: operation, shape, dtype, layout,
   device, inputs and effects. Track shared subexpressions and outputs; avoid
   recomputing or materializing the same expensive subtree at each consumer.
2. Preserve graphs through forward computation, reverse-mode differentiation,
   batching, layers and optimizer updates. A Java loop of eager gradient kernels
   is not the final compiled training design.
3. Permit materialization only at explicit reads, device transfers, outputs, or
   justified kernel boundaries. Keep matmul/reduction boundaries initially;
   add bias/activation epilogues, reduction fusion and layout-aware views where
   measured beneficial. A transpose should not require a copy when a view works.
4. No implicit host reads, scalar extraction, synchronization, or CPU fallback
   inside a GPU chain. Unsupported operations fail with a clear capability error;
   an explicitly requested fallback must report transfers and lost fusion.
5. Cache reusable plans and kernels with keys covering operation semantics,
   shapes, dtypes, layouts, device/compiler capabilities and numerical options.
   Runtime tensor values stay inputs; only explicitly static values specialize.
6. Keep buffers resident across repeated calls and full training steps. Reuse
   allocations with explicit ownership. For asynchronous dispatch, defer release
   until device completion; closing a Java wrapper must not free in-flight memory.
7. Do not force one kernel for every graph. Split when dependencies, shared memory,
   register pressure or occupancy justify it, while preserving device residency.
8. Keep precision and fast-math policy explicit. Fusion may change rounding;
   compare with stated tolerances and never silently trade away required accuracy.

Performance acceptance must include warmed, synchronized timings, cold compilation
separately, kernel/dispatch counts, allocation counts, host-device bytes and peak
memory. Benchmark the same shapes, dtypes and outputs against the pre-change
baseline on each available device. Include short and long elementwise chains,
shared branches, matmul+bias+activation, reductions, batched policies and a complete
forward/backward/optimizer step. A large supported elementwise chain should use
one dispatch and no intermediate device buffers; any exception needs measured
justification. Gate merges on agreed device-specific regression budgets rather
than an unsupported claim of universally optimal GPU speed.

## JAX features to implement or adapt

These are semantic targets; they do not imply Python API compatibility, XLA
integration, or automatic portability of Python source. Local import candidates
accelerate implementation but do not establish equivalent semantics.

| Priority | Feature | Reuse / required work | Acceptance |
| --- | --- | --- | --- |
| P0 | jit with reusable compiled plans | Extend JAX4J Jit/Expr; import compatible donor kernel lowerings | Repeated same-signature calls reuse plans; full chains remain fusible and device-resident |
| P0 | grad and value_and_grad composed with jit | Existing AutoDiff/Tape/Var; introduce differentiable tensor graph rules | Finite-difference and pinned JAX parity checks; compiled backward chains; parameter trees retain structure |
| P0 | True vmap | Start from TensorJIT API, implement per-primitive batching rules and axis metadata | No per-example Java dispatch loop; in_axes/out_axes and batched gradients behave correctly |
| P0 | Shape/dtype/broadcast contracts and pytrees | Existing Shape, DType, tree utilities; extend where missing | Structure-preserving flatten/unflatten; errors before dispatch; nested parameter inputs and outputs |
| P1 | Explicit functional random keys | Existing PrngKey plus on-device kernels | split/fold-in semantics defined; no hidden mutable RNG; test jit/vmap reproducibility; exact JAX streams only if algorithm matches |
| P1 | Structured cond, scan, while | New graph nodes and backend lowering | Control flow stays on device; define static-shape and differentiation restrictions; avoid unrolling long loops by default |
| P1 | jvp, vjp, Jacobians and higher derivatives | Extend autodiff primitive rules | Test transformations and their compositions; expose unsupported cases explicitly |
| P1 | custom derivatives and rematerialization | Extend primitive registration; learn from NablaTensor checkpoint planning | Tensor-shape-aware custom JVP/VJP contracts and numerically checked memory/recompute tradeoffs |
| P1 | Asynchronous dispatch and explicit completion | Adapt backend queues and lifetime tracking | Explicit await/blockUntilReady equivalent; valid results and buffer lifetimes under concurrent calls |
| P2 | Multi-device placement, sharding and collectives | New execution model; runtime/device discovery is only a building block | Start with explicit mesh/shards and collective semantics; test gradients and device transfers |
| P2 | Ahead-of-time export / StableHLO / XLA path | Separate feasibility study and optional toolchain | Define ABI, supported operations, packaging and measurable gain before adding dependencies |

Do not mark pmap implemented because a class comment lists it: the audited old
transform module contains JitFunction, Transforms and VmapFunction, without a
pmap implementation. For new multi-device work, consult current JAX sharding and
shard_map semantics instead of treating a historical pmap wrapper as the target.

## Simplify Maven modules aggressively

Adopt NablaTensor's useful rule: **one module per accelerator, containing its
runtime, buffers, kernels and any optional replay adapter together**. Go further
for the general library: tensor API, transformations, autodiff, NN and the CPU
implementation are packages in one core artifact. Package boundaries and tests
provide separation without multiplying Maven artifacts.

Proposed reactor:

| Artifact | Contents / dependency direction |
| --- | --- |
| jax4j-core | Tensor API, shapes/dtypes/devices, backend SPI/registry, graph/IR, JIT, transforms, autodiff, trees, PRNG, linalg, NN/optimizers, CPU interpreter/kernels and CPU JIT; no accelerator dependency |
| jax4j-engine-cuda | CUDA runtime, buffers, compiler/cache, tensor kernels, optional scalar replay adapter; depends on core |
| jax4j-engine-rocm | HIP equivalents; depends on core |
| jax4j-engine-vulkan | Vulkan runtime and shader compilation, tensor and optional replay kernels; depends on core |
| jax4j-engine-wgpu | wgpu-native runtime, WGSL tensor lowering and optional replay; depends on core |
| jax4j-examples | Examples, benchmark harnesses and parity runners; depends on core, accelerators opt-in |

Six primary child modules, plus the parent POM. Add engine-opencl only if there
is a concrete supported-device need. Keep general numerical examples and
validation harnesses in the optional examples module.
A dedicated app artifact may be retained if downstream deployment requires it,
but it should not split the core library back into tiny artifacts.

Concrete consolidation map:

| Existing JAX4J modules | Destination |
| --- | --- |
| tensor + backend-cpu + autodiff + nn | core, preserving com.nablatensor.jax4j.tensor/autodiff/nn packages |
| current core scalar recording/replay contracts + cpu + jit-cpu + simd + ops | core packages where reusable; optional SIMD selected at runtime/build profile without making CPU require incubator flags |
| backend-cuda + cuda | engine-cuda |
| backend-rocm + rocm | engine-rocm |
| backend-vulkan + vulkan | engine-vulkan |
| New wgpu import | engine-wgpu |
| examples + bench + validate harnesses | examples; reusable neural components stay in core |

Implementation rules:

- The existing core artifact includes scalar replay. Inventory and merge
  its classes deliberately before reusing its artifactId for the general core.
  Detect duplicate types/resources; do not overwrite the old directory blindly.
- Preserve Java packages during file movement; update artifact dependencies and
  scripts in one pass. Publish a migration table for changed Maven coordinates.
  Update launchers and classpaths together with moved source code.
- Merge META-INF/services provider files correctly. Core includes the always-
  available CPU provider; each GPU artifact registers only its own provider.
  Accelerators must not depend on each other or force native library loading
  during ordinary CPU startup. Missing toolchains/devices must be reported cleanly.
- Profiles select optional dependencies and hardware checks, not alternative
  implementations of public Tensor/autodiff classes. All accelerator Java sources
  should compile in a supported CI lane without requiring a physical GPU.
- Put operation helpers, code generation, dispatch and diagnostics in packages;
  do not create one Maven module per concern. Keep backend-neutral contracts in
  core and native implementation details in the owning accelerator artifact.

## Execution order and completion criteria

1. Capture existing CPU/parity checks, source provenance and representative GPU
   chain baselines. Inventory public coordinates, service descriptors and scripts.
2. Consolidate modules as mechanical moves first. CPU package/tests, policy parity
   and dependent launchers must work before changing kernel behavior.
3. Restore CUDA/ROCm/Vulkan tensor paths using pinned donor diffs. Prove fused
   forward and backward execution and runtime availability checks on real devices;
   mark unavailable hardware as unverified, not passed.
4. Bring shared graph/kernel improvements into core. Add reusable plan capture,
   real batching and compiled autodiff/optimizer support with fusion regression
   checks. Audit traced constants, views, error cleanup and shared subexpressions.
5. Add wgpu as a tensor backend: runtime first, then elementwise/fusion, matmul,
   reductions, gradients and full training. Keep a capability matrix per device.
6. Extend JAX semantics in priority order. Keep the independent pinned Python JAX
   reference for outputs/gradients; report precision-dependent tolerances.

A backend is ready only when it has numerical checks, gradient checks, chained-
operation dispatch evidence, warmed end-to-end benchmarks, absence/device error
checks and memory-lifetime validation. Compilation alone is an intermediate step.

## Primary JAX references

- [Composable grad/vmap transformations](https://docs.jax.dev/en/latest/101/transformations.html)
- [JIT compilation](https://docs.jax.dev/en/latest/201/jit.html)
- [API reference: transformations and device operations](https://docs.jax.dev/en/latest/jax.html)
- [Functional state and purity](https://docs.jax.dev/en/latest/stateful-computations.html)
- [Explicit PRNG keys](https://docs.jax.dev/en/latest/101/random.html)
- [Asynchronous dispatch and correct timing](https://docs.jax.dev/en/latest/async_dispatch.html)
- [Manual parallelism with shard_map](https://docs.jax.dev/en/latest/201/shard-map.html)
- [Migrating pmap](https://docs.jax.dev/en/latest/migrate_pmap.html)

These references describe the upstream behavior to study; proposed JAX4J behavior
and priorities above are project design decisions.
