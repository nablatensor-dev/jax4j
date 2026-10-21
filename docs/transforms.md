# Tensor transformations

The tensor transformation implementation targets CPU, Vulkan and wgpu-native F32 execution.
See [wgpu setup and hardware validation](wgpu.md) for the additional backend.
It provides JAX-style function transformations; it does not execute Python source
or use XLA. Legacy `AutoDiff`/`Var` and scalar replay remain available. Use the
`Transforms` API below for composable tensor higher derivatives.

## APIs and contracts

| Roadmap feature | Implementation |
| --- | --- |
| Cached JIT | `Jit.compile` and `Jit.compileTree`; backend/shape/dtype/device specialization; tensor values remain runtime inputs |
| Gradient and value/gradient | `Transforms.grad`, `valueAndGrad`, `gradTree`, `valueAndGradTree`; scalar loss; structured parameter gradients |
| Batching | `Vmap.apply` with multiple arguments and arbitrary positive/negative axes; null input axes mean shared arguments; `applyTree` preserves structured outputs with per-leaf output axes |
| Tree and shape contracts | Existing tree utilities; structure and shape validation for transformations, control-flow carries, cotangents, and optimizer state; unsupported dtype conversions fail |
| Functional randomness | Immutable `PrngKey.split`, `foldIn`, and counter semantics; `uniformLike`/`normalLike` use an existing tensor's tracing context |
| Structured control flow | `ControlFlow.cond`, `whileLoop`, `scan`; `condTree`, `whileTree`, `scanTree` preserve structured state |
| Derivative transformations | `Transforms.jvp`, `jvpTree`, explicit-cotangent `vjp`, `jacfwd`, `jacrev`, `hessian`; nested tensor gradients |
| Custom derivatives | `CustomDerivative.jvp` and `CustomDerivative.vjp`; callbacks use tensor operations and compose with capture/batching |
| Rematerialization | `CustomDerivative.checkpoint` retains region inputs and recomputes its pure body for the pullback |
| Explicit async completion | `Tensor.blockUntilReady`; Vulkan deferred frees; concurrent cached calls and early input release are validated |
| Compiled training | `FunctionalAdam` with explicit moment trees and a Tensor step counter, composing with compiled structured gradients |

For example:

```java
var gradient = Jit.compile(Transforms.grad(x -> x.mul(x).sum()));
try (Tensor result = gradient.apply(input)) {
    result.blockUntilReady();
}

Tensor mapped = Vmap.apply(
    args -> args.get(0).add(args.get(1)),
    List.of(matrix, sharedBias), Arrays.asList(1, null), 1);
```

`Vmap.applyTree` accepts input and output axes in deterministic pytree leaf order.
A null output axis requires an unmapped value. Mapped axis sizes must agree.
Primitive rules cover elementwise operations, broadcasts, reshape, transpose,
matmul/batched matmul, reductions and reduction gradients, slicing, stacking,
static gathers/scatters, convolution and its input/weight derivatives, custom
primitives, and structured regions. Nested maps preserve separate tracing contexts.
Random generation with a shared static key produces shared samples, as a pure
function of the key; it does not silently advance or split that key per example.

## Control-flow and differentiation restrictions

Predicates are single-element F32 tensors: zero means false, nonzero means true.
Both conditional branches must produce matching shape/structure. Carries retain
fixed shape, dtype, and structure. Captured tensor operands are device inputs;
no predicate or loop value is downloaded by Vulkan lowering. Nested regions work.

Ordinary scans and their first forward/reverse derivatives retain shader loops;
the Java body is captured once. Reverse-mode scans store the carry trajectory.
Dynamic while loops support forward differentiation and explicitly reject reverse
mode, matching the static-memory restriction in JAX. A custom VJP without a JVP
explicitly rejects forward mode. User-supplied JVP rules must be linear in the
tangent and VJP rules linear in the cotangent.

Higher derivatives through static scans currently expand those scans during
transformation. Jacobians use explicit basis sweeps. These paths are correct for
the validated cases but can produce large graphs; they are not a throughput claim.
Nondifferentiable primitive conventions follow the existing tensor backend:
axis-max gradients select the first maximum, ReLU has derivative zero at zero,
and sign/argmax have derivative zero. Exact JAX parity at nondifferentiable ties
is not claimed.

The device-region lowering currently executes each region serially in one shader
invocation, including regions under batching. It is device-resident and avoids
per-iteration Java dispatch, but needs parallel lowering before it is suitable for
large matrix/convolution workloads inside loops. Static gather maps are embedded
in shaders; large maps increase compilation cost. Ordinary tensor kernels retain
the existing parallel Vulkan implementation.

## Functional optimizer state

`FunctionalAdam.initialize(parameters)` returns first/second moments and a scalar
Tensor step. `update(parameters, gradients, state)` returns new parameters and new
state without mutating the inputs. All leaves can be inputs/outputs of one
`Jit.compileTree` training function; bias correction changes on cached calls.
The caller owns returned tensors and can use `TreeUtil.closeAll` on old states.

The original mutable `Adam` remains usable outside transformations. It rejects
traced parameters so cached Java side effects cannot silently freeze its counter.

## Numerical and lifecycle boundaries

The validated tensor dtype is F32. Other enum values do not imply implemented
storage/arithmetic. No implicit CPU fallback or host read is allowed inside a
compiled/differentiated/mapped device computation. Host-based linalg helpers are
not differentiable or batchable primitives. RNG keys are immutable Java values
and therefore static to capture; runtime Tensor key arrays and JAX Threefry streams
are not implemented. Closed-over tensors must remain alive while a plan uses them.
Functions must be pure; Java scalar values and side effects are evaluated at capture.

Checkpointing validates gradient equivalence and recomputation, but peak-memory
benefit depends on the region and queued-resource lifetime. Allocation pooling,
parallel region scheduling, selectable per-device numerical policies, and broad
performance budgets remain compiler/runtime work beyond the feature API coverage.

## Reproduction

```sh
.venv/bin/python python/jax_parity.py  # Requires JAX 0.11.2; forces CPU
mvn clean test -Djax4j.vulkan.tests=true
```

The checked-in fixture includes gradients, JVPs, Jacobians, Hessians, scan input
and carry gradients, and three functional Adam steps. Both CPU and explicitly
enabled Vulkan tests compare against it. Additional Vulkan checks cover zero
host transfers in cached scans and reverse scans, dispatch counts independent of
scan length, native allocation cleanup, and concurrent calls/early release.
