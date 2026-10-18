package com.jax4j.tensor.spi;

/** A transformation layer; tensor operations choose the innermost compatible tracing context. */
public interface TracingBackend extends ComputeBackend {
  ComputeBackend parentBackend();
}
