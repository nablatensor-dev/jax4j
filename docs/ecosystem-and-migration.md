# Ecosystem and source migration

## Local libraries known to this project

| Project | Local location | Role in JAX4J |
| --- | --- | --- |
| SonicTensor | `${HOME}/.m2/repository/com/sonictensor/`, `0.1.0-SNAPSHOT` | Cached tensor JAR inspected as the requested API baseline; not a runtime dependency |
| JAX4J / TensorJIT | `${HOME}/Projects/highprio/nablatensor-old` | CPU kernels, quantization support types, and tensor autodiff copied as Java source |
| NablaTensor | `${HOME}/Projects/highprio/nablatensor` | Initial public source snapshot used to recover the SonicTensor/NablaTensor-lineage code |
| Python JAX | Was absent from the default Python; installed here in `.venv` | Independent Python reference, autodiff training experiment, and synchronized CPU layout measurements |

The original donor projects are not modified by this migration.
Normal Maven dependencies such as JUnit/ASM remain ordinary external dependencies;
implementation code does not require any of the three upstream project's JARs.

## Provenance and limits

The cached SonicTensor artifacts did **not** include source JARs, and the inspected
local Git histories did not contain a recoverable pre-rename SonicTensor commit.
Consequently this is not claimed to be a byte-for-byte extraction of that cached
release. The Java sources come from NablaTensor's initial public snapshot
`2ddbe777752422ed5ba48b611e79017007119153`, whose tensor API corresponds to the
inspected SonicTensor API. Source packages and Maven artifact names are migrated to
`com.nablatensor` / `jax4j-*`, with Java packages under `com.jax4j`.
The retained sources provide tensor, autodiff, neural-network and accelerator functionality.

TensorJIT's CPU implementation, supporting quantization interfaces, and autodiff
implementation are copied directly from the local source checkout. The exact
checkout revision, original paths, destination paths, and SonicTensor JAR checksum
are recorded in `migration-manifest.json`. Existing copyright headers and the
source snapshot's LICENSE/NOTICE are preserved. New code is in the same source tree.

New ML work includes Dense over the final tensor dimension, presence-aware pooling,
a small CPU Adam optimizer, and all five Java inference layouts. No special 3D
convolution kernel is needed for the enemy records. CPU reshape shares immutable
heap storage instead of cloning it; uploads/downloads retain their copy boundaries.
The Java real-time policy path uses preallocated primitive scratch arrays rather
than allocating tensor wrappers for every forward pass.

## Scope of validation

- The ML reactor's tests cover shapes/tree utilities, migrated autodiff, shared
  3D Dense gradients, Adam regression, masked pooling gradients, and empty sets.
- All migrated legacy modules compile at Java 25 source level with tests skipped.
- Python synthetic training and invariance/masking checks run on CPU.
- Every Java policy matches JAX's actions on 500 fixtures, with small float32
  numerical differences. Parity checks include absent-slot junk and empty scenes.
- GPU kernels, game-driven training, and integrated
  world-tick timing remain separate work; package success alone does not validate them.
