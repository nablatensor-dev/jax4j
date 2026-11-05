/*
 * Copyright 2026 The NablaTensor Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.nablatensor.jax4j.tensor;

import com.nablatensor.jax4j.tensor.spi.ComputeBackend;
import com.nablatensor.jax4j.tensor.spi.DeviceBuffer;

import java.util.List;
import java.util.Random;

/** Fluent entry point for creating tensors and querying devices/backends. */
public final class Tensors {

  private Tensors() {
  }

  // ---- device / backend introspection ------------------------------------

  public static List<ComputeBackend> backends() {
    return BackendRegistry.available();
  }

  /** Devices available on this machine, one per available backend. */
  public static List<Device> devices() {
    return BackendRegistry.available().stream()
        .map(b -> new Device(b.deviceType(), 0))
        .toList();
  }

  public static Device defaultDevice() {
    ComputeBackend backend = BackendRegistry.defaultBackend();
    return new Device(backend.deviceType(), 0);
  }

  /** Creates an immutable reproducible counter-based random key. */
  public static PrngKey key(long seed) {
    return PrngKey.of(seed);
  }

  // ---- tensor construction ------------------------------------------------

  public static Tensor array(float[] data, int... dims) {
    Shape shape = Shape.of(dims.length == 0 ? new int[] {data.length} : dims);
    return arrayOn(data, shape, defaultDevice());
  }

  public static Tensor array(float[][] data) {
    int rows = data.length;
    int cols = data[0].length;
    float[] flat = new float[rows * cols];
    for (int r = 0; r < rows; r++) {
      System.arraycopy(data[r], 0, flat, r * cols, cols);
    }
    return arrayOn(flat, Shape.of(rows, cols), defaultDevice());
  }

  public static Tensor arrayOn(float[] data, Shape shape, Device device) {
    ComputeBackend backend = BackendRegistry.forDevice(device);
    DeviceBuffer buffer = backend.upload(data, shape, DType.F32, device);
    return new Tensor(backend, buffer);
  }

  public static Tensor zeros(int... dims) {
    return full(0, dims);
  }

  /** Creates an F32 constant on the default device. */
  public static Tensor full(float value, int... dims) {
    return full(defaultDevice(), value, dims);
  }

  /** Creates an F32 constant using the device backend's fill implementation. */
  public static Tensor full(Device device, float value, int... dims) {
    ComputeBackend backend = BackendRegistry.forDevice(device);
    return new Tensor(backend, backend.full(value, Shape.of(dims), DType.F32, device));
  }

  public static Tensor ones(int... dims) { return full(1, dims); }
  public static Tensor ones(Device device, int... dims) { return full(device, 1, dims); }
  public static Tensor zeros(Device device, int... dims) { return full(device, 0, dims); }

  /** F32 sequence, including start and excluding stop. Values are generated on the host
   * and uploaded once to the requested device. Length is computed in double precision.
   * This is an eager constructor; create inputs outside compiled functions. */
  public static Tensor arange(double stop) { return arange(0, stop, 1); }
  public static Tensor arange(double start, double stop) { return arange(start, stop, 1); }
  public static Tensor arange(double start, double stop, double step) {
    return arange(defaultDevice(), start, stop, step);
  }
  public static Tensor arange(Device device, double stop) { return arange(device, 0, stop, 1); }
  public static Tensor arange(Device device, double start, double stop) {
    return arange(device, start, stop, 1);
  }
  public static Tensor arange(Device device, double start, double stop, double step) {
    if (!Double.isFinite(start) || !Double.isFinite(stop) || !Double.isFinite(step) || step == 0)
      throw new IllegalArgumentException("arange requires finite bounds and a finite nonzero step");
    double count = Math.max(0, Math.ceil((stop - start) / step));
    if (!Double.isFinite(count) || count > Integer.MAX_VALUE)
      throw new IllegalArgumentException("arange result is too large");
    float[] values = new float[(int) count];
    for (int i = 0; i < values.length; i++) values[i] = (float) (start + i * step);
    return arrayOn(values, Shape.of(values.length), device);
  }

  /** F32 evenly spaced samples including both endpoints. */
  public static Tensor linspace(double start, double stop, int count) {
    return linspace(defaultDevice(), start, stop, count, true);
  }
  public static Tensor linspace(double start, double stop, int count, boolean endpoint) {
    return linspace(defaultDevice(), start, stop, count, endpoint);
  }
  public static Tensor linspace(Device device, double start, double stop, int count) {
    return linspace(device, start, stop, count, true);
  }
  /** Eager host-generated samples, uploaded once; count zero is empty and count one is start. */
  public static Tensor linspace(Device device, double start, double stop, int count, boolean endpoint) {
    if (!Double.isFinite(start) || !Double.isFinite(stop) || count < 0)
      throw new IllegalArgumentException("linspace requires finite bounds and nonnegative count");
    float[] values = new float[count];
    double denominator = endpoint ? count - 1.0 : count;
    for (int i = 0; i < count; i++) {
      double t = count == 1 ? 0 : i / denominator;
      values[i] = (float) ((1 - t) * start + t * stop);
    }
    if (endpoint && count > 1) values[count - 1] = (float) stop;
    return arrayOn(values, Shape.of(count), device);
  }

  public static Tensor randn(int... dims) {
    return randn(new Random(), dims);
  }

  public static Tensor randn(Random random, int... dims) {
    Shape shape = Shape.of(dims);
    float[] data = new float[(int) shape.size()];
    for (int i = 0; i < data.length; i++) {
      data[i] = (float) random.nextGaussian();
    }
    return arrayOn(data, shape, defaultDevice());
  }
}
