# JAX4J

[Website](https://nablatensor.com/)

Java 25 tensors, automatic differentiation and neural networks under
`com.nablatensor:jax4j:0.2.1`. The fluent Tensor API supports fused
CPU, CUDA, Vulkan and wgpu-native execution. There are no dependencies on SonicTensor,
NablaTensor or TensorJIT JARs.

**Java. Differentiated.** Tensors → autodiff → JIT → gradient descent, live in JShell.

[![JAX4J in JShell: colorful live demo of tensors, automatic differentiation, JIT replay and gradient descent](demo/jax4j.gif)](demo/README.md)

## Hello, JAX4J! A little calculus, hold the pencil.

A common task in machine learning: measure an error and find which way to move
to reduce it. Here, the loss is the mean of the squared errors; JAX4J works out
the gradient for you.

```java
import com.nablatensor.jax4j.tensor.Tensor;
import com.nablatensor.jax4j.tensor.Tensors;
import com.nablatensor.jax4j.tensor.Transforms;
import java.util.Arrays;

try (var errors = Tensors.array(new float[] {1, 2, 3, 4})) {
    var result = Transforms.valueAndGrad(x -> x.mul(x).mean(), errors);
    try (Tensor loss = result.value(); Tensor gradient = result.grad()) {
        System.out.println("Loss: " + loss.toFloatArray()[0]);
        System.out.println("Gradient: " + Arrays.toString(gradient.toFloatArray()));
    }
}
// Loss: 7.5
// Gradient: [0.5, 1.0, 1.5, 2.0]
```

`mul(x).mean()` squares each error and averages the results. The gradient tells
us how each error affects the loss: take a small step in the opposite direction
to bring it down. Same Java expression, automatic derivatives. Runs on CPU with
just `jax4j-core`; no GPU required.

For JIT compilation, batching and more, see the
[transformation guide](docs/transforms.md).

## Why JAX4J exists

JAX4J grew out of earlier automatic differentiation work in the
SonicTensor/NablaTensor lineage, incorporating TensorJIT CPU and autodiff sources.
It brings that foundation into a self-contained Java tensor and neural-network
library, motivated by three practical frustrations:

- **Fewer options in Java.** Python has a rich ecosystem for tensors, automatic
  differentiation and machine learning. Java developers have far fewer choices;
  JAX4J aims to make these capabilities available directly on the JVM.
- **Large Python environments.** Python machine-learning dependencies installed
  in project-local `.venv` directories can consume gigabytes of disk space,
  especially when repeated across projects. A Java library with optional GPU
  backends offers a more compact way to get started.
- **Overlooked integrated GPUs.** GPU support in existing tools often leaves AMD
  integrated GPUs underserved. Modern AMD machines are particularly appealing
  development systems: advanced chip fabrication helps deliver excellent power
  efficiency and quiet operation. Making useful compute available on their
  integrated GPUs through Vulkan and wgpu is a central motivation for JAX4J.

## Releases

[JAX4J 0.2.1](https://github.com/nablatensor-dev/jax4j/releases/tag/v0.2.1)
is published on Maven Central. Module JARs and checksums are also available
in the GitHub release.

## Maven Central

JAX4J 0.2.1 is available under `com.nablatensor` on
[Maven Central](https://central.sonatype.com/artifact/com.nablatensor/jax4j-core/0.2.1).

```xml
<dependency>
  <groupId>com.nablatensor</groupId>
  <artifactId>jax4j-core</artifactId>
  <version>0.2.1</version>
</dependency>
<dependency>
  <groupId>com.nablatensor</groupId>
  <artifactId>jax4j-engine-vulkan</artifactId>
  <version>0.2.1</version>
  <scope>runtime</scope>
</dependency>
```

Use `jax4j-engine-wgpu` for wgpu-native. The parent artifact `jax4j` is a POM.
CUDA, ROCm and SIMD artifacts are included in the GitHub release.
Version 0.2.1 validated Vulkan and wgpu hardware; CUDA, ROCm and SIMD
execution remain unverified for this version. Finance APIs are maintained separately in
[NablaTensor Quant](https://github.com/nablatensor-dev/quant).

## Build and validate

Requires JDK 25+ and Maven.

```sh
mvn test
mvn -Pcuda -Djax4j.cuda.tests=true test
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

Version 0.2.1 corrects the Java package namespace. Update imports when upgrading from earlier releases.
Java packages use `com.nablatensor.jax4j`; tensor factories are available through `Tensors`.
Select a default backend with `-Djax4j.tensor.backend=cpu` or `JAX4J_BACKEND=cpu`.

Core, Vulkan and wgpu build by default. Enable optional backends with
`-Pcuda`, `-Procm` or `-Psimd`; combine profiles as needed.
ROCm hardware remains unverified. CUDA scalar replay does not yet support
multi-output tapes, uniform draws or named random streams.

For native installation, API usage, validation and measured speed comparisons,
see [wgpu support](docs/wgpu.md).
