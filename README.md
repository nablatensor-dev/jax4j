# JAX4J

Java 25 tensors, automatic differentiation and neural networks under
`com.nablatensor:jax4j:0.1.0`. The fluent Tensor API supports fused
CPU, Vulkan and wgpu-native execution. There are no dependencies on SonicTensor,
NablaTensor or TensorJIT JARs.

Release maintainers: see [the release guide](docs/releasing.md).

## Maven Central

JAX4J 0.1.0 is published under `com.nablatensor` on
[Maven Central](https://central.sonatype.com/artifact/com.nablatensor/jax4j-core/0.1.0).

```xml
<dependency>
  <groupId>com.nablatensor</groupId>
  <artifactId>jax4j-core</artifactId>
  <version>0.1.0</version>
</dependency>
<dependency>
  <groupId>com.nablatensor</groupId>
  <artifactId>jax4j-engine-vulkan</artifactId>
  <version>0.1.0</version>
  <scope>runtime</scope>
</dependency>
```

Use `jax4j-engine-wgpu` for wgpu-native. The parent artifact `jax4j` is a POM.
CUDA, ROCm and SIMD artifacts are also available; this publication validated
only Vulkan and wgpu. The Maven Central binary JARs match the GitHub v0.1.0
release assets byte for byte. Finance APIs are maintained separately in
[NablaTensor Quant](https://github.com/nablatensor-dev/quant).

## Build and validate

```sh
export JAVA_HOME=/opt/zulu25.30.17-ca-jdk25.0.1-linux_x64
export PATH="$JAVA_HOME/bin:$PATH"
mvn test
mvn -Djax4j.vulkan.tests=true test
mvn -Djax4j.wgpu.tests=true -Djax4j.vulkan.tests=true test
mvn -Pexamples test
```

Vulkan hardware tests are explicit: when enabled, unavailable Vulkan is a test
failure. CPU tests never need a GPU. Native execution uses Java FFM with
`--enable-native-access=ALL-UNNAMED` (configured for Maven tests).

| Artifact | Contents |
| --- | --- |
| `jax4j-core` | Tensor API, CPU kernels, graph capture, JIT, batching, autodiff, pytrees, PRNG, linalg, NN and Adam; scalar recording and CPU replay |
| `jax4j-engine-cuda` | Optional CUDA runtime, tensor kernels and scalar replay (`-Pcuda`) |
| `jax4j-engine-rocm` | Optional ROCm/HIP runtime, tensor kernels and scalar replay (`-Procm`) |
| `jax4j-engine-vulkan` | Vulkan runtime, buffers, tensor shaders and scalar replay adapter |
| `jax4j-engine-wgpu` | wgpu-native tensor runtime, WGSL cooperative kernels and native GLSL/Naga lowering |
| `jax4j-examples` | Optional examples and numerical validation harnesses (`-Pexamples`) |

Java packages use `com.jax4j`; tensor factories are available through `Tensors`.
Select a default backend with `-Djax4j.tensor.backend=cpu` or `JAX4J_BACKEND=cpu`.
The scalar API uses `Nabla.Evaluator` / `TypedEvaluator`, `Evaluation` /
`TypedEvaluation`, `valueOnly()`, `value()`, `gradient()` and `gradients()`.

Core, Vulkan and wgpu form the default reactor. CUDA and ROCm each combine their
tensor runtime and scalar replay engine in one module. Build them with `-Pcuda`
and `-Procm`, or both with `-Pcuda,rocm`; their hardware execution is unverified.
SIMD scalar replay remains optional under `-Psimd`.

For native installation, API usage, validation and measured speed comparisons,
see [wgpu support](docs/wgpu.md).

## Fused computation

```java
var function = Jit.compile(x -> x.mul(x).add(3).tanh());
try (Tensor output = function.apply(input)) {
    output.blockUntilReady();
}
```

The first call captures the pure Java function. Calls with the same backend,
shape, dtype and device replay the captured kernels with new runtime values.
Host reads inside compiled functions fail explicitly. Java side effects and
scalar constants are evaluated during capture; closed-over tensors must stay alive.
Use `Jit.compileTree` for dynamic Tensor parameter trees and structured outputs,
including `AutoDiff.ValueAndGrad`. Use multi-input `Jit.apply` to capture all
parameters of a layer or update.

`Transforms` provides tensor gradients, JVP/VJP, Jacobians, Hessians, structured
parameter gradients and nested differentiation. `Vmap.apply` supports arbitrary
axes and multiple mapped/shared inputs; `Vmap.applyTree` preserves structured
outputs. Compose these with `Jit.compile`/`compileTree` for cached execution.

`ControlFlow` provides device conditional, while and scan regions, including
structured state. `CustomDerivative` supplies custom JVP/VJP and recomputation
checkpoints. Dynamic while loops support forward mode; reverse mode is rejected.

Use `FunctionalAdam` with explicit moment trees and a Tensor step counter inside
compiled training functions. The mutable `Adam` remains available outside tracing.
Vulkan defers buffer frees until pending commands finish.

See [P0/P1 APIs, restrictions and validation](docs/p0-p1-transforms.md). Tensor
execution is F32. Device regions currently use serial shader lowering; optimized
parallel regions and broad performance budgets remain future work.

## Reproduction and provenance

- [Roadmap and implementation status](docs/gpu-jax-roadmap.md)
- [Maven migration and validation](docs/implementation-status.md)
- [Original source provenance](docs/ecosystem-and-migration.md)
- [Per-file import manifest](docs/migration-manifest.json)
- [Mechanical source moves](docs/module-moves.json)

`python/jax_parity.py` regenerates checked-in fixtures with JAX **0.11.2**, forced
to CPU. Java parity tests use the fixtures without requiring Python.

For synchronized Vulkan timing and dispatch/allocation/transfer counts:

```sh
mvn -DskipTests package
java --enable-native-access=ALL-UNNAMED \
  -cp jax4j-core/target/classes:jax4j-engine-vulkan/target/classes \
  com.jax4j.backend.vulkan.VulkanBenchmark
```
