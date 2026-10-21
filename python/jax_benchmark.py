"""Compare JAX JIT and eager execution of the JAX4J Vulkan benchmark chain."""
import json
from time import perf_counter

import jax
import jax.numpy as jnp
import numpy as np

SIZE = 262_144
ITERATIONS = 20


def chain(x):
    for _ in range(12):
        x = jnp.tanh(x * 1.0001 + 0.0001)
    return x


def main():
    device = jax.devices()[0]
    compiled = jax.jit(chain)
    x = jax.device_put(np.full(SIZE, 0.5, dtype=np.float32), device)
    x.block_until_ready()

    start = perf_counter()
    compiled(x).block_until_ready()
    cold_ms = (perf_counter() - start) * 1000

    for _ in range(5):
        compiled(x).block_until_ready()
        chain(x).block_until_ready()

    def measure(fn):
        start = perf_counter()
        for _ in range(ITERATIONS):
            fn(x).block_until_ready()
        return (perf_counter() - start) * 1000 / ITERATIONS

    fused_ms = measure(compiled)
    eager_ms = measure(chain)

    ref = np.float32(0.5)
    for _ in range(12):
        ref = np.float32(np.tanh(float(ref) * 1.0001 + 0.0001))
    actual = np.asarray(compiled(x))
    if not np.all(np.abs(actual - ref) <= 3e-5):
        raise AssertionError("numerical mismatch")

    print(json.dumps({
        "jaxVersion": jax.__version__,
        "device": str(device),
        "deviceKind": device.device_kind,
        "platform": device.platform,
        "dtype": str(x.dtype),
        "elements": SIZE,
        "iterations": ITERATIONS,
        "chainOperations": 36,
        "coldCaptureAndCompileMs": cold_ms,
        "fusedMs": fused_ms,
        "eagerMs": eager_ms,
        "speedup": eager_ms / fused_ms,
        "numericalTolerance": 0.00003,
    }, indent=2))


if __name__ == "__main__":
    main()
