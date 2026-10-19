package com.nablatensor.jax4j.backend.cpu;

import com.nablatensor.jax4j.tensor.DType;
import com.nablatensor.jax4j.tensor.Device;
import com.nablatensor.jax4j.tensor.Shape;
import com.nablatensor.jax4j.tensor.spi.DeviceBuffer;

/** CPU-resident tensor storage backed by a flat {@code float[]}. */
final class CpuBuffer implements DeviceBuffer {

  final float[] data;
  private final Shape shape;
  private final DType dtype;
  private final Device device;

  CpuBuffer(float[] data, Shape shape, DType dtype, Device device) {
    this.data = data;
    this.shape = shape;
    this.dtype = dtype;
    this.device = device;
  }

  @Override
  public Shape shape() {
    return shape;
  }

  @Override
  public DType dtype() {
    return dtype;
  }

  @Override
  public Device device() {
    return device;
  }
}
