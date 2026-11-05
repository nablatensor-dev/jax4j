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

  public static Tensor where(Tensor mask, Tensor yes, Tensor no) {
    return Tensor.where(mask, yes, no);
  }

  public static Tensor oneHot(Tensor indices, int classes) {
    return indices.oneHot(classes);
  }

  public static Tensor concatenate(List<Tensor> parts, int axis) {
    return Tensor.concatenate(parts, axis);
  }

  public static Tensor stack(List<Tensor> parts, int axis) {
    return Tensor.stack(parts, axis);
  }

  public static Tensor logspace(double start, double stop, int count) {
    return logspace(defaultDevice(), start, stop, count, 10);
  }

  public static Tensor logspace(Device device, double start, double stop, int count) {
    return logspace(device, start, stop, count, 10);
  }

  public static Tensor logspace(double start, double stop, int count, double base) {
    return logspace(defaultDevice(), start, stop, count, base);
  }

  public static Tensor logspace(Device device, double start, double stop, int count, double base) {
    if (!Double.isFinite(start)
        || !Double.isFinite(stop)
        || !Double.isFinite(base)
        || base <= 0
        || count < 0) throw new IllegalArgumentException("logspace bounds/base/count");
    float[] values = new float[count];
    for (int i = 0; i < count; i++) {
      double u = count <= 1 ? 0 : i / (count - 1.0);
      values[i] = (float) Math.pow(base, (1 - u) * start + u * stop);
    }
    return arrayOn(values, Shape.of(count), device);
  }

  public static Tensor geomspace(double start, double stop, int count) {
    return geomspace(defaultDevice(), start, stop, count);
  }

  public static Tensor geomspace(Device device, double start, double stop, int count) {
    if (!Double.isFinite(start)
        || !Double.isFinite(stop)
        || start == 0
        || stop == 0
        || Math.signum(start) != Math.signum(stop)
        || count < 0)
      throw new IllegalArgumentException("geomspace same-sign finite nonzero bounds/count");
    float[] values = new float[count];
    for (int i = 0; i < count; i++) {
      double u = count <= 1 ? 0 : i / (count - 1.0);
      values[i] =
          (float)
              (Math.signum(start)
                  * Math.exp((1 - u) * Math.log(Math.abs(start)) + u * Math.log(Math.abs(stop))));
    }
    if (count > 0) values[0] = (float) start;
    if (count > 1) values[count - 1] = (float) stop;
    return arrayOn(values, Shape.of(count), device);
  }

  public static Tensor eye(int rows) {
    return eye(defaultDevice(), rows, rows, 0);
  }

  public static Tensor eye(int rows, int cols) {
    return eye(defaultDevice(), rows, cols, 0);
  }

  public static Tensor eye(int rows, int cols, int offset) {
    return eye(defaultDevice(), rows, cols, offset);
  }

  public static Tensor eye(Device device, int rows) {
    return eye(device, rows, rows, 0);
  }

  public static Tensor eye(Device device, int rows, int cols) {
    return eye(device, rows, cols, 0);
  }

  public static Tensor eye(Device device, int rows, int cols, int offset) {
    Shape shape = Shape.of(rows, cols);
    float[] values = new float[Math.toIntExact(shape.size())];
    for (int i = 0; i < rows; i++) {
      long j = (long) i + offset;
      if (j >= 0 && j < cols) values[i * cols + (int) j] = 1;
    }
    return arrayOn(values, shape, device);
  }

  /** Matrix-indexed dense grids by default; xy swaps the first two output axes. */
  public static List<Tensor> meshgrid(Tensor... axes) {
    return meshgrid(List.of(axes));
  }

  public static List<Tensor> meshgrid(List<Tensor> axes) {
    return meshgrid(axes, "ij", false);
  }

  public static List<Tensor> meshgrid(List<Tensor> axes, String indexing, boolean sparse) {
    if (!indexing.equals("ij") && !indexing.equals("xy"))
      throw new IllegalArgumentException("grid indexing must be ij or xy");
    int rank = axes.size();
    int[] dims = new int[rank];
    for (int d = 0; d < rank; d++) {
      if (axes.get(d).shape().rank() != 1)
        throw new IllegalArgumentException("grid axes must be vectors");
      if (!axes.get(d).device().equals(axes.getFirst().device()))
        throw new IllegalArgumentException("grid device mismatch");
      dims[d] = axes.get(d).shape().dim(0);
    }
    if (indexing.equals("xy") && rank >= 2) {
      int v = dims[0];
      dims[0] = dims[1];
      dims[1] = v;
    }
    java.util.ArrayList<Tensor> grids = new java.util.ArrayList<>();
    try {
      for (int d = 0; d < rank; d++) {
        int axis = indexing.equals("xy") && rank >= 2 && d < 2 ? 1 - d : d;
        int[] shape = new int[rank];
        java.util.Arrays.fill(shape, 1);
        shape[axis] = axes.get(d).shape().dim(0);
        Tensor reshaped = axes.get(d).reshape(shape);
        Shape target = Shape.of(sparse ? shape : dims);
        if (reshaped.shape().equals(target)) grids.add(reshaped);
        else {
          try {
            grids.add(reshaped.broadcastTo(target));
          } finally {
            reshaped.close();
          }
        }
      }
      return List.copyOf(grids);
    } catch (RuntimeException e) {
      for (Tensor grid : grids) grid.close();
      throw e;
    }
  }

  public static List<Tensor> indices(int... dims) {
    return indices(defaultDevice(), dims);
  }

  public static List<Tensor> indices(Device device, int... dims) {
    Shape.of(dims);
    java.util.ArrayList<Tensor> axes = new java.util.ArrayList<>();
    try {
      for (int dim : dims) {
        if (dim > 16777216) throw new IllegalArgumentException("F32 coordinate range");
        axes.add(arange(device, dim));
      }
      return meshgrid(axes);
    } finally {
      for (Tensor axis : axes) axis.close();
    }
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
