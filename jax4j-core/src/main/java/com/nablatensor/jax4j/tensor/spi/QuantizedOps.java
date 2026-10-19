package com.nablatensor.jax4j.tensor.spi;

import com.nablatensor.jax4j.tensor.Device;
import com.nablatensor.jax4j.tensor.QuantType;
import com.nablatensor.jax4j.tensor.Shape;

import java.lang.foreign.MemorySegment;
import java.util.Set;

/**
 * Optional capability: hold quantized weights and multiply them by an F32
 * vector without ever materialising the dequantized matrix.
 *
 * <p>Discover with {@code instanceof}; never assume a {@link ComputeBackend}
 * implements this. Added so tensorjit-llm can run fused dequant-GEMV kernels
 * without adding quant-specific methods to the stable {@link ComputeBackend}
 * SPI (see JAX4J_LLM_PLAN.md §4).
 */
public interface QuantizedOps {

  Set<QuantType> supportedQuantTypes();

  /** Uploads an opaque quantized blob. {@code logicalShape} is the F32 shape it decodes to. */
  DeviceBuffer uploadQuantized(MemorySegment blocks, QuantType type, Shape logicalShape, Device device);

  /** {@code y(rows) = W(rows, cols) . x(cols)}, dequantizing on the fly. */
  DeviceBuffer quantMatvec(DeviceBuffer weights, DeviceBuffer x);

  /** {@code y(rows) = W(rows, cols) . x(cols) + add(rows)} -- the fused residual. */
  DeviceBuffer quantMatvecAdd(DeviceBuffer weights, DeviceBuffer x, DeviceBuffer add);

  /** One logical row of a quantized matrix, as F32 -- the embedding lookup. */
  DeviceBuffer quantGatherRow(DeviceBuffer weights, int row);
}
