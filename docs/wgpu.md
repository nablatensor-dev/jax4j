# wgpu-native tensor execution

`jax4j-engine-wgpu` adds `Device.wgpu()` and `Backend.WGPU`. It runs the tensor
API, fused capture, batching, autodiff, structured control flow and functional
Adam through wgpu-native. The default reactor includes core, Vulkan and wgpu.
Explicit CPU use does not load GPU libraries. Explicit unavailable wgpu requests
fail instead of falling back to CPU.

## Native setup

The Java FFM bindings are pinned to **wgpu-native 29.0.1.1**, whose C ABI is checked
before device creation. Linux x86_64 is the validated platform. The installer
checks the official release archive's SHA-256 and installs locally:

```sh
scripts/install-wgpu-native.sh
export JAX4J_WGPU_NATIVE_PATH="$PWD/.native/wgpu-native/lib/libwgpu_native.so"
mvn test -Djax4j.wgpu.tests=true -Djax4j.vulkan.tests=true
```

Alternatively set `-Djax4j.wgpu.native_path=/absolute/path/libwgpu_native.so`.
The system property takes precedence over the environment variable. Discovery
also supports the previously installed NablaTensor cache and the system loader.
No native binary or native JAR is committed. The wgpu module has no dependency
on the Vulkan module or shaderc. Its headless adapter request selects Vulkan;
this backend is native wgpu execution, not browser WebGPU.

A diagnostic is available after compilation:

```sh
java --enable-native-access=ALL-UNNAMED \
  -cp jax4j-core/target/classes:jax4j-engine-wgpu/target/classes \
  com.jax4j.backend.wgpu.WgpuInfo
```

## Architecture and contracts

The core `GpuTensorBackend`, `GpuRuntime` and `GpuBuffer` share tensor kernels,
shape validation and buffer ownership without creating an accelerator dependency
on another accelerator. Vulkan retains its runtime, shaderc compilation and
public diagnostics. wgpu owns its C ABI binding, adapter/device/queue, pipelines,
staging readback and diagnostics.

Cooperative matmul and reduction kernels use explicit WGSL with
`workgroupBarrier()`. Remaining primitive, fused and structured-region shaders
use wgpu-native's GLSL extension and Naga validation/lowering. That path requires
the native distribution's GLSL frontend, as supplied by the pinned release.
No native subgroup extensions are required. A large training benchmark exposed
incorrect cooperative synchronization in the original GLSL path; the WGSL
kernels and large, repeated reduction/matmul/training regression checks address it.

Commands are encoded in bounded batches and submitted on explicit completion or
readback. Buffer destruction is deferred until submitted work has completed.
Dispatch parameters use immutable, bounded cached storage buffers. Tensor data
transfers and parameter uploads have separate diagnostics. `liveDeviceBytes`
tracks tensor buffers; persistent pipeline/parameter and staging resources are
excluded. The parameter cache is capped at 1024 entries and retired only after
queue completion.

The current numerical contract is F32 on device 0. Shared-key random generation
has the same deterministic project stream as CPU/Vulkan. The existing
[P0/P1 restrictions](p0-p1-transforms.md), including serial structured-region
lowering and static expansion for higher-order scans, also apply to wgpu.

## Validation

The final clean isolated build recorded **118 passed, 1 existing CPU test skipped**:
31 Vulkan tests, 36 wgpu tests, and 52 core cases including the skip. The exact
source snapshot was checked against the workspace. See
[the validation record](wgpu-validation.json) for commands, suite totals,
source checksum, native version and library SHA-256.

The wgpu checks include the same JAX parity/transformation/queue tests as Vulkan,
plus multi-workgroup reductions at 257/65,536/262,144 elements, nonuniform tiled
matmul with tail dimensions, large functional training, missing-native-library
CPU startup, and rejection of explicit unavailable-device requests. The official
installed binary matches the previously cached binary byte-for-byte.

## Speed comparison

Measured on **AMD Radeon Graphics (RADV PHOENIX)**. wgpu itself uses Vulkan on this
machine. Both runtimes execute the same F32 workloads and dispatch counts. Input
uploads and numerical verification downloads are outside timing. Each measured
call waits for completion; final-output cleanup is excluded for both backends.
Results use 10 warmup calls, 15 alternating-order samples and 5 calls per sample.

| Workload | Vulkan median ms | wgpu median ms | Faster backend |
| --- | ---: | ---: | --- |
| Fused 36-operation chain, 262,144 elements | 0.402 | 0.447 | Vulkan 1.11x |
| Eager same chain | 2.870 | 2.271 | wgpu 1.26x |
| 256 x 256 matmul | 0.233 | 0.362 | Vulkan 1.56x |
| Compiled gradient, 262,144 elements | 0.714 | 0.495 | wgpu 1.44x |
| Compiled gradient + functional Adam step | 1.351 | 0.923 | wgpu 1.46x |
| Scan, 128 steps x 32 carry elements | 0.854 | 1.433 | Vulkan 1.68x |

Every warmed timed region recorded zero tensor transfers, zero parameter uploads
and zero new shader compilations. Primary numerical outputs were checked before
timing. Cold function times are separate and exclude runtime/device and static
kernel initialization. These measurements describe this run on this GPU; they
are not a cross-device performance claim. Both scan implementations currently use
serial device regions. Raw timings, allocation/dispatch counts and cold times are
in [the benchmark JSON](wgpu-vulkan-benchmark.json).

Reproduce after building:

```sh
java --enable-native-access=ALL-UNNAMED \
  -cp jax4j-core/target/classes:jax4j-engine-vulkan/target/classes:jax4j-engine-wgpu/target/classes \
  com.jax4j.tensor.bench.GpuComparison
```

Properties `size`, `matrix`, `scanLength`, `warmup`, `samples`, and `iterations`
control the comparison. Initial source/runtime provenance is recorded in
`migration-manifest.json`; the prior consolidation/P0/P1 work was committed as
`21ac4c9` before this backend was added.
