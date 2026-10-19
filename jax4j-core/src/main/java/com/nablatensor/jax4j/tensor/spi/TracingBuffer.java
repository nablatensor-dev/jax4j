package com.nablatensor.jax4j.tensor.spi;

/** Associates a symbolic value with the transformation that owns it. */
public interface TracingBuffer extends DeviceBuffer {
  ComputeBackend tracingBackend();
}
