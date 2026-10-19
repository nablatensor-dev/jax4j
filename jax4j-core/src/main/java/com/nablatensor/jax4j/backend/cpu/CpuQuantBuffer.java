package com.nablatensor.jax4j.backend.cpu;

import com.nablatensor.jax4j.tensor.DType;
import com.nablatensor.jax4j.tensor.Device;
import com.nablatensor.jax4j.tensor.QuantType;
import com.nablatensor.jax4j.tensor.Shape;
import com.nablatensor.jax4j.tensor.spi.DeviceBuffer;

/** CPU-resident opaque quantized blob: raw ggml blocks, never dequantized in bulk. */
final class CpuQuantBuffer implements DeviceBuffer {

  final byte[] blocks;
  final QuantType type;
  private final Shape logicalShape;
  private final Device device;

  CpuQuantBuffer(byte[] blocks, QuantType type, Shape logicalShape, Device device) {
    this.blocks = blocks;
    this.type = type;
    this.logicalShape = logicalShape;
    this.device = device;
  }

  @Override
  public Shape shape() {
    return logicalShape;
  }

  @Override
  public DType dtype() {
    return DType.F32; // logical (decoded) element type; storage itself is quantized
  }

  @Override
  public Device device() {
    return device;
  }
}
