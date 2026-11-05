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

/**
 * An immutable, fluent tensor handle. Every operation returns a new {@code Tensor}; materialization
 * happens on the backend as operations are issued, and {@link #eval()} / {@link #blockUntilReady()}
 * wait for completion.
 *
 * <p>Backends normally reclaim device memory only once a wrapper becomes unreachable and the
 * GC/Cleaner get around to it, which is heuristic and can lag under sustained allocation pressure.
 * Tight loops that discard a result every iteration (benchmarks, training steps) should call {@link
 * #close()} (or use try-with-resources) to free it deterministically instead.
 */
public final class Tensor implements AutoCloseable {

  private final ComputeBackend backend;
  private final DeviceBuffer buffer;

  Tensor(ComputeBackend backend, DeviceBuffer buffer) {
    this.backend = backend;
    this.buffer = buffer;
  }

  // ---- metadata -----------------------------------------------------------

  public Shape shape() {
    return buffer.shape();
  }

  public DType dtype() {
    return buffer.dtype();
  }

  public Device device() {
    return buffer.device();
  }

  DeviceBuffer buffer() {
    return buffer;
  }

  ComputeBackend backend() {
    return backend;
  }

  ComputeBackend common(Tensor other) {
    if (backend == other.backend) return backend;
    ComputeBackend cursor = other.backend;
    while (cursor instanceof com.nablatensor.jax4j.tensor.spi.TracingBackend trace) {
      cursor = trace.parentBackend();
      if (cursor == backend) return other.backend;
    }
    cursor = backend;
    while (cursor instanceof com.nablatensor.jax4j.tensor.spi.TracingBackend trace) {
      cursor = trace.parentBackend();
      if (cursor == other.backend) return backend;
    }
    ComputeBackend left = backend, right = other.backend;
    while (left instanceof com.nablatensor.jax4j.tensor.spi.TracingBackend trace)
      left = trace.parentBackend();
    while (right instanceof com.nablatensor.jax4j.tensor.spi.TracingBackend trace)
      right = trace.parentBackend();
    if (left.getClass() == right.getClass()) {
      if (!(other.backend instanceof com.nablatensor.jax4j.tensor.spi.TracingBackend)) return backend;
      if (!(backend instanceof com.nablatensor.jax4j.tensor.spi.TracingBackend)) return other.backend;
    }
    throw new IllegalArgumentException("Operands belong to incompatible backend contexts");
  }

  /** Whether this handle belongs to an active transformation context. */
  public boolean isTracing() {
    return backend instanceof com.nablatensor.jax4j.tensor.spi.TracingBackend;
  }

  // ---- device / placement -------------------------------------------------

  /** Move to another device, re-uploading through the target backend. */
  public Tensor on(Device device) {
    if (device.equals(device())) {
      return this;
    }
    ComputeBackend target = BackendRegistry.forDevice(device);
    DeviceBuffer moved = target.upload(backend.download(buffer), shape(), dtype(), device);
    return new Tensor(target, moved);
  }

  /** Creates a constant with this tensor's shape, placement and tracing context. */
  public Tensor fullLike(float value) {
    return wrap(backend.full(value, shape(), dtype(), device()));
  }

  /** Constant zero preserving shape, dtype, device and tracing context. */
  public Tensor zerosLike() { return fullLike(0); }

  /** Constant one preserving shape, dtype, device and tracing context. */
  public Tensor onesLike() { return fullLike(1); }

  public Tensor f32() {
    return dtype(DType.F32);
  }

  public Tensor dtype(DType dtype) {
    if (dtype != dtype()) {
      throw new UnsupportedOperationException(
          "Tensor dtype conversion is unavailable: " + dtype() + " to " + dtype);
    }
    return this;
  }

  // ---- binary elementwise -------------------------------------------------

  public Tensor add(Tensor other) {
    return binaryBroadcast(Op.ADD, other);
  }

  public Tensor sub(Tensor other) {
    return binaryBroadcast(Op.SUB, other);
  }

  public Tensor mul(Tensor other) {
    return binaryBroadcast(Op.MUL, other);
  }

  public Tensor div(Tensor other) {
    return binaryBroadcast(Op.DIV, other);
  }

  public Tensor maximum(Tensor other) {
    return binaryBroadcast(Op.MAX, other);
  }

  public Tensor minimum(Tensor other) {
    return binaryBroadcast(Op.MIN, other);
  }

  /**
   * Binary elementwise op with NumPy-style broadcasting: operands with differing shapes are each
   * expanded to their common {@link Shape#broadcast} shape first. Any tensor materialized just for
   * this broadcast is closed immediately after the op runs, so no extra buffers linger.
   */
  private Tensor binaryBroadcast(Op op, Tensor other) {
    if (dtype() != other.dtype() || !device().equals(other.device()))
      throw new IllegalArgumentException("Elementwise operands must share dtype and device");
    ComputeBackend selected = common(other);
    Shape ls = shape();
    Shape rs = other.shape();
    if (ls.equals(rs)) {
      return new Tensor(selected, selected.binary(op, buffer, other.buffer));
    }
    Shape target = Shape.broadcast(ls, rs);
    Tensor lb =
        ls.equals(target) ? null : new Tensor(selected, selected.broadcastTo(buffer, target));
    Tensor rb =
        rs.equals(target) ? null : new Tensor(selected, selected.broadcastTo(other.buffer, target));
    try {
      DeviceBuffer lBuf = lb != null ? lb.buffer : buffer;
      DeviceBuffer rBuf = rb != null ? rb.buffer : other.buffer;
      return new Tensor(selected, selected.binary(op, lBuf, rBuf));
    } finally {
      if (lb != null) {
        lb.close();
      }
      if (rb != null) {
        rb.close();
      }
    }
  }

  // ---- scalar elementwise -------------------------------------------------

  public Tensor add(double value) {
    return wrap(backend.scalar(Op.ADD, buffer, value));
  }

  public Tensor sub(double value) {
    return wrap(backend.scalar(Op.SUB, buffer, value));
  }

  public Tensor mul(double value) {
    return wrap(backend.scalar(Op.MUL, buffer, value));
  }

  public Tensor div(double value) {
    return wrap(backend.scalar(Op.DIV, buffer, value));
  }

  public Tensor maximum(double value) {
    return wrap(backend.scalar(Op.MAX, buffer, value));
  }

  public Tensor minimum(double value) {
    return wrap(backend.scalar(Op.MIN, buffer, value));
  }

  // ---- unary elementwise --------------------------------------------------

  public Tensor neg() {
    return wrap(backend.unary(Op.NEG, buffer));
  }

  /** Elementwise square. */
  public Tensor square() { return mul(this); }

  /** Accurate log(1+x), including values of x close to zero. */
  public Tensor log1p() { return wrap(backend.unary(Op.LOG1P, buffer)); }

  /** Accurate exp(x)-1, including values of x close to zero. */
  public Tensor expm1() { return wrap(backend.unary(Op.EXPM1, buffer)); }

  /** Stable log(1+exp(x)). */
  public Tensor softplus() { return wrap(backend.unary(Op.SOFTPLUS, buffer)); }

  public Tensor exp() {
    return wrap(backend.unary(Op.EXP, buffer));
  }

  public Tensor log() {
    return wrap(backend.unary(Op.LOG, buffer));
  }

  public Tensor sqrt() {
    return wrap(backend.unary(Op.SQRT, buffer));
  }

  public Tensor rsqrt() {
    return wrap(backend.unary(Op.RSQRT, buffer));
  }

  public Tensor tanh() {
    return wrap(backend.unary(Op.TANH, buffer));
  }

  public Tensor sigmoid() {
    return wrap(backend.unary(Op.SIGMOID, buffer));
  }

  public Tensor relu() {
    return wrap(backend.unary(Op.RELU, buffer));
  }

  public Tensor abs() {
    return wrap(backend.unary(Op.ABS, buffer));
  }

  public Tensor sign() {
    return wrap(backend.unary(Op.SIGN, buffer));
  }

  // ---- movement / contraction --------------------------------------------

  public Tensor transpose() {
    return wrap(backend.transpose(buffer));
  }

  public Tensor matmul(Tensor other) {
    return new Tensor(common(other), common(other).matmul(buffer, other.buffer));
  }

  // ---- factorizations ---------------------------------------------------

  /**
   * Cholesky factor {@code L} of this SPD matrix ({@code this = L·Lᵀ}). See {@link
   * Linalg#cholesky}.
   */
  public Tensor cholesky() {
    return Linalg.cholesky(this);
  }

  /** LU factorization with partial pivoting. See {@link Linalg#lu}. */
  public Linalg.Lu lu() {
    return Linalg.lu(this);
  }

  /** Householder QR factorization. See {@link Linalg#qr}. */
  public Linalg.Qr qr() {
    return Linalg.qr(this);
  }

  /** Symmetric eigendecomposition (cyclic Jacobi). See {@link Linalg#eigh}. */
  public Linalg.Eigh eigh() {
    return Linalg.eigh(this);
  }

  /** Thin singular value decomposition (one-sided Jacobi). See {@link Linalg#svd}. */
  public Linalg.Svd svd() {
    return Linalg.svd(this);
  }

  /** Solves {@code this · X = b} for {@code X}. See {@link Linalg#solve}. */
  public Tensor solve(Tensor b) {
    return Linalg.solve(this, b);
  }

  /** The inverse of this square matrix. See {@link Linalg#inv}. */
  public Tensor inv() {
    return Linalg.inv(this);
  }

  /** The determinant of this square matrix. See {@link Linalg#det}. */
  public double det() {
    return Linalg.det(this);
  }

  /** Batched matrix multiplication: {@code (B,M,K) x (B,K,N) -> (B,M,N)}. */
  public Tensor batchedMatmul(Tensor other) {
    if (!device().equals(other.device()) || dtype() != other.dtype()) {
      throw new IllegalArgumentException(
          "batchedMatmul operands must use the same backend, device, and dtype");
    }
    return new Tensor(common(other), common(other).batchedMatmul(buffer, other.buffer));
  }

  /** Copies one leading-axis slice and removes the leading dimension. */
  public Tensor sliceAxis0(int index) {
    return wrap(backend.sliceAxis0(buffer, index));
  }

  /** Stacks equally-shaped tensors along a new leading axis. */
  public static Tensor stackAxis0(List<Tensor> tensors) {
    if (tensors.isEmpty()) {
      throw new IllegalArgumentException("stackAxis0 requires at least one tensor");
    }
    Tensor first = tensors.getFirst();
    ComputeBackend selected = first.backend;
    for (Tensor tensor : tensors) {
      if (!tensor.shape().equals(first.shape())
          || tensor.dtype() != first.dtype()
          || !tensor.device().equals(first.device()))
        throw new IllegalArgumentException("stack placement or shape mismatch");
      selected = new Tensor(selected, first.buffer).common(tensor);
    }
    DeviceBuffer[] buffers = new DeviceBuffer[tensors.size()];
    for (int i = 0; i < tensors.size(); i++) {
      Tensor tensor = tensors.get(i);
      buffers[i] = tensor.buffer;
    }
    return new Tensor(selected, selected.stackAxis0(buffers));
  }

  /** Returns an independent tensor with the same elements and a different shape. */
  public Tensor reshape(int... dims) {
    Shape target = shape().reshape(dims);
    return wrap(backend.reshape(buffer, target));
  }

  /** Permutes axes with a device-local gather. */
  public Tensor permute(int... axes) {
    int rank = shape().rank();
    if (axes.length != rank) throw new IllegalArgumentException("permutation rank mismatch");
    boolean[] used = new boolean[rank];
    int[] dims = new int[rank];
    for (int i = 0; i < rank; i++) {
      int axis = axes[i];
      if (axis < 0 || axis >= rank || used[axis])
        throw new IllegalArgumentException("invalid permutation");
      used[axis] = true;
      dims[i] = shape().dim(axis);
    }
    Shape target = Shape.of(dims);
    int[] map = new int[Math.toIntExact(target.size())];
    for (int i = 0; i < map.length; i++) {
      int rest = i, index = 0;
      int[] coordinates = new int[rank];
      for (int j = rank - 1; j >= 0; j--) {
        coordinates[axes[j]] = rest % dims[j];
        rest /= dims[j];
      }
      for (int j = 0; j < rank; j++) index = index * shape().dim(j) + coordinates[j];
      map[i] = index;
    }
    return reindex(target, map);
  }

  public Tensor moveAxis(int from, int to) {
    int rank = shape().rank();
    from = from < 0 ? from + rank : from;
    to = to < 0 ? to + rank : to;
    if (from < 0 || from >= rank || to < 0 || to >= rank)
      throw new IllegalArgumentException("axis out of bounds");
    int[] axes = new int[rank];
    int j = 0;
    for (int i = 0; i < rank; i++)
      if (i != to) {
        while (j == from) j++;
        axes[i] = j++;
      }
    axes[to] = from;
    return permute(axes);
  }

  /** Static device-local gather, with -1 entries producing zero. */
  public Tensor reindex(Shape target, int[] indices) {
    return wrap(backend.reindex(buffer, target, indices.clone()));
  }

  public Tensor scatterAdd(Shape target, int[] indices) {
    return wrap(backend.scatterAdd(buffer, target, indices.clone()));
  }

  public Tensor lt(Tensor other) {
    return binaryBroadcast(Op.LT, other);
  }

  public Tensor lt(double value) {
    return wrap(backend.scalar(Op.LT, buffer, value));
  }

  public Tensor le(Tensor other) {
    return binaryBroadcast(Op.LE, other);
  }

  public Tensor le(double value) {
    return wrap(backend.scalar(Op.LE, buffer, value));
  }

  public Tensor gt(Tensor other) {
    return binaryBroadcast(Op.GT, other);
  }

  public Tensor gt(double value) {
    return wrap(backend.scalar(Op.GT, buffer, value));
  }

  public Tensor ge(Tensor other) {
    return binaryBroadcast(Op.GE, other);
  }

  public Tensor ge(double value) {
    return wrap(backend.scalar(Op.GE, buffer, value));
  }

  public Tensor eq(Tensor other) {
    return binaryBroadcast(Op.EQ, other);
  }

  public Tensor eq(double value) {
    return wrap(backend.scalar(Op.EQ, buffer, value));
  }

  public Tensor ne(Tensor other) {
    return binaryBroadcast(Op.NE, other);
  }

  public Tensor ne(double value) {
    return wrap(backend.scalar(Op.NE, buffer, value));
  }

  public Tensor and(Tensor other) {
    return binaryBroadcast(Op.AND, other);
  }

  public Tensor and(double value) {
    return wrap(backend.scalar(Op.AND, buffer, value));
  }

  public Tensor or(Tensor other) {
    return binaryBroadcast(Op.OR, other);
  }

  public Tensor or(double value) {
    return wrap(backend.scalar(Op.OR, buffer, value));
  }

  public Tensor xor(Tensor other) {
    return binaryBroadcast(Op.XOR, other);
  }

  public Tensor xor(double value) {
    return wrap(backend.scalar(Op.XOR, buffer, value));
  }

  public Tensor pow(Tensor other) {
    return binaryBroadcast(Op.POW, other);
  }

  public Tensor pow(double value) {
    return wrap(backend.scalar(Op.POW, buffer, value));
  }

  public Tensor hypot(Tensor other) {
    return binaryBroadcast(Op.HYPOT, other);
  }

  public Tensor hypot(double value) {
    return wrap(backend.scalar(Op.HYPOT, buffer, value));
  }

  public Tensor sin() {
    return wrap(backend.unary(Op.SIN, buffer));
  }

  public Tensor cos() {
    return wrap(backend.unary(Op.COS, buffer));
  }

  public Tensor not() {
    return wrap(backend.unary(Op.NOT, buffer));
  }

  public Tensor isfinite() {
    return wrap(backend.unary(Op.ISFINITE, buffer));
  }

  public Tensor isnan() {
    return wrap(backend.unary(Op.ISNAN, buffer));
  }

  /** Execute a shape-specialized array operation in the innermost input context. */
  static Tensor arrayOp(ArrayKernel kernel, Tensor... tensors) {
    if (tensors.length == 0) throw new IllegalArgumentException("inputs required");
    ComputeBackend context = tensors[0].backend;
    DeviceBuffer[] buffers = new DeviceBuffer[tensors.length];
    for (int i = 0; i < tensors.length; i++) {
      if (!tensors[i].device().equals(tensors[0].device()) || tensors[i].dtype() != DType.F32)
        throw new IllegalArgumentException("array operations require same-device F32 inputs");
      context = new Tensor(context, tensors[0].buffer).common(tensors[i]);
      buffers[i] = tensors[i].buffer;
    }
    return new Tensor(context, context.arrayKernel(kernel, buffers));
  }

  static Tensor arrayOp(
      ArrayKernel.Kind kind, Shape output, int axis, int[] params, Tensor... inputs) {
    Shape[] shapes = new Shape[inputs.length];
    for (int i = 0; i < inputs.length; i++) shapes[i] = inputs[i].shape();
    return arrayOp(new ArrayKernel(kind, output, shapes, axis, params), inputs);
  }

  public static Tensor where(Tensor mask, Tensor yes, Tensor no) {
    Shape target = Shape.broadcast(mask.shape(), Shape.broadcast(yes.shape(), no.shape()));
    java.util.ArrayList<Tensor> expanded = new java.util.ArrayList<>();
    Tensor[] inputs = {mask, yes, no};
    try {
      for (int i = 0; i < inputs.length; i++)
        if (!inputs[i].shape().equals(target)) {
          inputs[i] = inputs[i].broadcastTo(target);
          expanded.add(inputs[i]);
        }
      return arrayOp(ArrayKernel.Kind.WHERE, target, 0, new int[0], inputs);
    } finally {
      for (Tensor tensor : expanded) tensor.close();
    }
  }

  public Tensor slice(int axis, int start, int stop) {
    return slice(axis, start, stop, 1);
  }

  /** Python-style explicit integer bounds; stop is exclusive, step cannot be zero. */
  public Tensor slice(int axis, int start, int stop, int step) {
    axis = normalizeAxis(axis);
    int length = shape().dim(axis);
    if (step == 0) throw new IllegalArgumentException("zero slice step");
    long a = start < 0 ? (long) start + length : start, b = stop < 0 ? (long) stop + length : stop;
    a = step > 0 ? Math.max(0, Math.min(length, a)) : Math.max(-1, Math.min(length - 1, a));
    b = step > 0 ? Math.max(0, Math.min(length, b)) : Math.max(-1, Math.min(length - 1, b));
    int count =
        (int)
            Math.max(
                0, step > 0 ? (b - a + step - 1) / step : (a - b - (long) step - 1) / -(long) step);
    int[] dims = shape().dims();
    dims[axis] = count;
    Shape target = Shape.of(dims);
    int[] map = new int[Math.toIntExact(target.size())];
    for (int i = 0; i < map.length; i++) {
      int source = 0, stride = 1, n = i;
      for (int d = dims.length - 1; d >= 0; d--) {
        int c = n / axisStride(target, d) % Math.max(1, dims[d]);
        source += (d == axis ? (int) (a + (long) c * step) : c) * stride;
        stride *= shape().dim(d);
      }
      map[i] = source;
    }
    return reindex(target, map);
  }

  private static int axisStride(Shape shape, int axis) {
    int n = 1;
    for (int d = axis + 1; d < shape.rank(); d++) n = Math.multiplyExact(n, shape.dim(d));
    return Math.max(1, n);
  }

  public Tensor take(Tensor indices, int axis) {
    axis = normalizeAxis(axis);
    if (shape().dim(axis) > 16777216) throw new IllegalArgumentException("F32 index range");
    int[] dims = new int[shape().rank() - 1 + indices.shape().rank()];
    int j = 0;
    for (int d = 0; d < axis; d++) dims[j++] = shape().dim(d);
    for (int v : indices.shape().dims()) dims[j++] = v;
    for (int d = axis + 1; d < shape().rank(); d++) dims[j++] = shape().dim(d);
    return arrayOp(ArrayKernel.Kind.TAKE, Shape.of(dims), axis, new int[0], this, indices);
  }

  private void checkRegion(Shape region, Tensor starts) {
    if (region.rank() != shape().rank()
        || starts.shape().rank() != 1
        || starts.shape().dim(0) != shape().rank())
      throw new IllegalArgumentException("region/start rank");
    for (int d = 0; d < region.rank(); d++)
      if (region.dim(d) > shape().dim(d) || shape().dim(d) > 16777216)
        throw new IllegalArgumentException("region extent");
  }

  public Tensor dynamicSlice(Tensor starts, int... sizes) {
    Shape region = Shape.of(sizes);
    checkRegion(region, starts);
    return arrayOp(ArrayKernel.Kind.SLICE, region, 0, new int[0], this, starts);
  }

  public Tensor dynamicUpdateSlice(Tensor update, Tensor starts) {
    checkRegion(update.shape(), starts);
    return arrayOp(ArrayKernel.Kind.UPDATE, shape(), 0, new int[0], this, update, starts);
  }

  public static Tensor concatenate(List<Tensor> parts, int axis) {
    if (parts.isEmpty()) throw new IllegalArgumentException("empty concatenate");
    Tensor first = parts.getFirst();
    axis = first.normalizeAxis(axis);
    int[] dims = first.shape().dims();
    dims[axis] = 0;
    for (Tensor part : parts) {
      if (part.shape().rank() != dims.length)
        throw new IllegalArgumentException("concatenate rank");
      for (int d = 0; d < dims.length; d++)
        if (d != axis && part.shape().dim(d) != dims[d])
          throw new IllegalArgumentException("concatenate shape");
      dims[axis] = Math.addExact(dims[axis], part.shape().dim(axis));
    }
    return arrayOp(
        ArrayKernel.Kind.CONCAT, Shape.of(dims), axis, new int[0], parts.toArray(Tensor[]::new));
  }

  public static Tensor stack(List<Tensor> parts, int axis) {
    if (parts.isEmpty()) throw new IllegalArgumentException("empty stack");
    int rank = parts.getFirst().shape().rank() + 1;
    axis = axis < 0 ? axis + rank : axis;
    if (axis < 0 || axis >= rank) throw new IllegalArgumentException("stack axis");
    java.util.ArrayList<Tensor> expanded = new java.util.ArrayList<>();
    try {
      for (Tensor part : parts) {
        if (!part.shape().equals(parts.getFirst().shape()))
          throw new IllegalArgumentException("stack shape");
        int[] dims = new int[rank];
        for (int d = 0, j = 0; d < rank; d++) dims[d] = d == axis ? 1 : part.shape().dim(j++);
        expanded.add(part.reshape(dims));
      }
      return concatenate(expanded, axis);
    } finally {
      for (Tensor tensor : expanded) tensor.close();
    }
  }

  public Tensor prod() {
    try (Tensor flat = reshape(Math.toIntExact(shape().size()))) {
      return arrayOp(ArrayKernel.Kind.PROD, Shape.of(1), 0, new int[0], flat);
    }
  }

  public Tensor prod(int axis) {
    return prod(axis, false);
  }

  public Tensor prod(int axis, boolean keepDims) {
    axis = normalizeAxis(axis);
    int[] dims = new int[shape().rank() - (keepDims ? 0 : 1)];
    for (int d = 0, j = 0; d < shape().rank(); d++)
      if (d != axis) dims[j++] = shape().dim(d);
      else if (keepDims) dims[j++] = 1;
    return arrayOp(ArrayKernel.Kind.PROD, Shape.of(dims), axis, new int[0], this);
  }

  public Tensor argmin(int axis) {
    try (Tensor negative = neg()) {
      return negative.argmax(axis);
    }
  }

  public record TopK(Tensor values, Tensor indices) {}

  /** Descending along the last axis; NaNs first, ties prefer smaller indices. */
  public TopK topK(int k) {
    if (shape().rank() == 0) throw new IllegalArgumentException("topK rank");
    int[] dims = shape().dims();
    int width = dims[dims.length - 1];
    if (k < 0 || k > width || width > 16777216)
      throw new IllegalArgumentException("topK count/index range");
    dims[dims.length - 1] = k;
    Tensor values = arrayOp(ArrayKernel.Kind.TOPK, Shape.of(dims), 0, new int[] {k, 0}, this);
    try {
      return new TopK(
          values, arrayOp(ArrayKernel.Kind.TOPK, Shape.of(dims), 0, new int[] {k, 1}, this));
    } catch (RuntimeException e) {
      values.close();
      throw e;
    }
  }

  public Tensor segmentSum(Tensor ids, int count) {
    if (shape().rank() < 1
        || ids.shape().rank() != 1
        || ids.shape().dim(0) != shape().dim(0)
        || count < 0
        || count > 16777216) throw new IllegalArgumentException("segment shapes/count");
    int[] dims = shape().dims();
    dims[0] = count;
    return arrayOp(ArrayKernel.Kind.SEGMENT, Shape.of(dims), 0, new int[0], this, ids);
  }

  public Tensor oneHot(int classes) {
    if (classes < 0 || classes > 16777216) throw new IllegalArgumentException("class count");
    int[] dims = java.util.Arrays.copyOf(shape().dims(), shape().rank() + 1);
    dims[dims.length - 1] = classes;
    return arrayOp(ArrayKernel.Kind.ONEHOT, Shape.of(dims), 0, new int[0], this);
  }

  public Tensor diff(int axis) {
    return diff(1, axis);
  }

  public Tensor diff(int order, int axis) {
    axis = normalizeAxis(axis);
    if (order < 0) throw new IllegalArgumentException("negative diff order");
    Tensor current = reshape(shape().dims());
    try {
      for (int n = 0; n < order; n++) {
        int count = current.shape().dim(axis);
        if (count == 0) break;
        try (Tensor left = current.slice(axis, 0, count - 1);
            Tensor right = current.slice(axis, 1, count)) {
          Tensor next = right.sub(left);
          current.close();
          current = next;
        }
      }
      Tensor result = current;
      current = null;
      return result;
    } finally {
      if (current != null) current.close();
    }
  }

  public Tensor roll(int shift, int axis) {
    axis = normalizeAxis(axis);
    int width = shape().dim(axis);
    if (width == 0) return reshape(shape().dims());
    int amount = Math.floorMod(shift, width), stride = axisStride(shape(), axis);
    int[] map = new int[Math.toIntExact(shape().size())];
    for (int i = 0; i < map.length; i++) {
      int c = i / stride % width;
      map[i] = i + (Math.floorMod(c - amount, width) - c) * stride;
    }
    return reindex(shape(), map);
  }

  public Tensor pad(int[][] widths) {
    return pad(widths, 0);
  }

  public Tensor pad(int[][] widths, float value) {
    if (widths.length != shape().rank()) throw new IllegalArgumentException("pad rank");
    int[] dims = shape().dims();
    for (int d = 0; d < dims.length; d++) {
      if (widths[d].length != 2 || widths[d][0] < 0 || widths[d][1] < 0)
        throw new IllegalArgumentException("pad width");
      dims[d] = Math.addExact(dims[d], Math.addExact(widths[d][0], widths[d][1]));
    }
    Shape target = Shape.of(dims);
    int[] map = new int[Math.toIntExact(target.size())];
    for (int i = 0; i < map.length; i++) {
      int source = 0;
      boolean inside = true;
      for (int d = 0; d < dims.length; d++) {
        int c = i / axisStride(target, d) % Math.max(1, dims[d]) - widths[d][0];
        inside &= c >= 0 && c < shape().dim(d);
        source += c * axisStride(shape(), d);
      }
      map[i] = inside ? source : -1;
    }
    if (value == 0) return reindex(target, map);
    try (Tensor result = reindex(target, map);
        Tensor ones = onesLike();
        Tensor mask = ones.reindex(target, map);
        Tensor fill = result.fullLike(value)) {
      return where(mask, result, fill);
    }
  }

  // ---- convolution --------------------------------------------------------

  /**
   * 2-D convolution of batched images (one image per row) against {@code weights} of shape {@code
   * (outChannels, inChannels * k * k)}.
   */
  public Tensor conv2d(Tensor weights, ConvSpec spec) {
    return new Tensor(common(weights), common(weights).conv2d(buffer, weights.buffer, spec));
  }

  /** Gradient of {@link #conv2d} with respect to the input it was given. */
  public Tensor conv2dGradInput(Tensor weights, ConvSpec spec) {
    return new Tensor(
        common(weights), common(weights).conv2dGradInput(buffer, weights.buffer, spec));
  }

  /** Gradient of {@link #conv2d} with respect to its weights; {@code this} is the input. */
  public Tensor conv2dGradWeight(Tensor upstream, ConvSpec spec) {
    return new Tensor(
        common(upstream), common(upstream).conv2dGradWeight(buffer, upstream.buffer, spec));
  }

  /** Routes {@code this} (an upstream gradient) through a ReLU evaluated at {@code input}. */
  public Tensor reluBackward(Tensor input) {
    return new Tensor(common(input), common(input).reluBackward(buffer, input.buffer));
  }

  // ---- reductions ---------------------------------------------------------

  /** Sums every element, returning a single-element (shape {@code (1)}) tensor. */
  public Tensor sum() {
    return wrap(backend.reduceSum(buffer));
  }

  /** Mean over every element, returning a single-element (shape {@code (1)}) tensor. */
  public Tensor mean() {
    try (Tensor total = sum()) {
      return total.div((double) shape().size());
    }
  }

  /** Axis mean, dropping the selected dimension. */
  public Tensor mean(int axis) { return mean(axis, false); }
  public Tensor mean(int axis, boolean keepDims) {
    int normalized = normalizeAxis(axis);
    try (Tensor total = sum(normalized, keepDims)) {
      return total.div(shape().dim(normalized));
    }
  }

  public Tensor var() { return varianceAll(0); }
  public Tensor std() { try (Tensor v = var()) { return v.sqrt(); } }
  public Tensor var(int axis) { return var(axis, 0, false); }
  public Tensor var(int axis, boolean keepDims) { return var(axis, 0, keepDims); }
  public Tensor var(int axis, int correction, boolean keepDims) {
    if (correction < 0) throw new IllegalArgumentException("Negative variance correction");
    int normalized = normalizeAxis(axis);
    try (Tensor center = mean(normalized, true);
         Tensor differences = sub(center);
         Tensor squares = differences.square();
         Tensor total = squares.sum(normalized, keepDims)) {
      int divisor = shape().dim(normalized) - correction;
      return divisor <= 0 ? total.fullLike(Float.NaN) : total.div(divisor);
    }
  }
  public Tensor std(int axis) { return std(axis, 0, false); }
  public Tensor std(int axis, boolean keepDims) { return std(axis, 0, keepDims); }
  public Tensor std(int axis, int correction, boolean keepDims) {
    try (Tensor variance = var(axis, correction, keepDims)) { return variance.sqrt(); }
  }
  private Tensor varianceAll(int correction) {
    try (Tensor center = mean(); Tensor differences = sub(center);
         Tensor squares = differences.square(); Tensor total = squares.sum()) {
      long divisor = shape().size() - correction;
      return divisor <= 0 ? total.fullLike(Float.NaN) : total.div(divisor);
    }
  }

  public Tensor logsumexp() {
    try (Tensor maximum = max(); Tensor lower = maximum.maximum(-Float.MAX_VALUE);
         Tensor shift = lower.minimum(Float.MAX_VALUE); Tensor shifted = sub(shift);
         Tensor e = shifted.exp(); Tensor total = e.sum(); Tensor logged = total.log()) {
      return logged.add(shift);
    }
  }
  public Tensor logsumexp(int axis) { return logsumexp(axis, false); }
  public Tensor logsumexp(int axis, boolean keepDims) {
    int normalized = normalizeAxis(axis);
    try (Tensor maximum = max(normalized, true);
         Tensor lower = maximum.maximum(-Float.MAX_VALUE);
         Tensor shift = lower.minimum(Float.MAX_VALUE); Tensor shifted = sub(shift);
         Tensor e = shifted.exp(); Tensor total = e.sum(normalized, true);
         Tensor logged = total.log(); Tensor result = logged.add(shift)) {
      if (keepDims) return result.reshape(result.shape().dims());
      int[] dims = new int[shape().rank() - 1];
      for (int i = 0, j = 0; i < shape().rank(); i++) if (i != normalized) dims[j++] = shape().dim(i);
      return result.reshape(dims);
    }
  }
  public Tensor softmax(int axis) {
    try (Tensor maximum = max(axis, true); Tensor shifted = sub(maximum);
         Tensor e = shifted.exp(); Tensor denominator = e.sum(axis, true)) {
      return e.div(denominator);
    }
  }
  public Tensor logSoftmax(int axis) {
    try (Tensor maximum = max(axis, true); Tensor shifted = sub(maximum);
         Tensor e = shifted.exp(); Tensor total = e.sum(axis, true); Tensor logged = total.log()) {
      return shifted.sub(logged);
    }
  }

  /** Maximum over every element, returning shape {@code (1)}. */
  public Tensor max() {
    return wrap(backend.reduceMax(buffer));
  }

  /** Sums a rank-2 tensor over axis 0 (the batch axis), returning shape {@code (cols)}. */
  public Tensor sumAxis0() {
    if (shape().rank() != 2) {
      throw new IllegalStateException("sumAxis0 requires a rank-2 tensor, got " + shape());
    }
    return wrap(backend.sumAxis0(buffer));
  }

  public Tensor sum(int axis) {
    return sum(axis, false);
  }

  public Tensor sum(int axis, boolean keepDims) {
    return wrap(backend.sumAxis(buffer, normalizeAxis(axis), keepDims));
  }

  public Tensor max(int axis) {
    return max(axis, false);
  }

  public Tensor max(int axis, boolean keepDims) {
    return wrap(backend.maxAxis(buffer, normalizeAxis(axis), keepDims));
  }

  /**
   * Returns maximum-value indices along {@code axis} as F32 values. Indices up to 2^24 are exact.
   */
  public Tensor argmax(int axis) {
    return wrap(backend.argMaxAxis(buffer, normalizeAxis(axis)));
  }

  /** Routes reduced gradients to the first maximum position along {@code axis}. */
  public Tensor maxAxisBackward(Tensor upstream, int axis) {
    return new Tensor(
        common(upstream),
        common(upstream).maxAxisBackward(upstream.buffer, buffer, normalizeAxis(axis)));
  }

  /** Broadcasts to {@code target}, following {@link Shape#broadcast}'s rules. */
  public Tensor broadcastTo(Shape target) {
    return shape().equals(target) ? this : wrap(backend.broadcastTo(buffer, target));
  }

  private int normalizeAxis(int axis) {
    int rank = shape().rank();
    int normalized = axis < 0 ? axis + rank : axis;
    if (normalized < 0 || normalized >= rank) {
      throw new IllegalArgumentException("axis " + axis + " is out of bounds for shape " + shape());
    }
    return normalized;
  }

  // ---- terminals ----------------------------------------------------------

  /** Force completion of all pending kernels and return this handle. */
  public Tensor eval() {
    backend.sync();
    return this;
  }

  public Tensor blockUntilReady() {
    return eval();
  }

  /**
   * Deterministically frees this tensor's device buffer now, instead of waiting on GC/Cleaner
   * timing. Safe to call even if the GC later also reclaims the same (by-then-unreachable) buffer -
   * backends guarantee the underlying free only ever runs once. Do not use this tensor after
   * closing it.
   */
  @Override
  public void close() {
    backend.release(buffer);
  }

  public float[] toFloatArray() {
    return backend.download(buffer);
  }

  public float[][] toFloat2D() {
    Shape shape = shape();
    if (shape.rank() != 2) {
      throw new IllegalStateException("toFloat2D requires a rank-2 tensor, got " + shape);
    }
    int rows = shape.dim(0);
    int cols = shape.dim(1);
    float[] flat = toFloatArray();
    float[][] out = new float[rows][cols];
    for (int r = 0; r < rows; r++) {
      System.arraycopy(flat, r * cols, out[r], 0, cols);
    }
    return out;
  }

  public float item() {
    float[] data = toFloatArray();
    if (data.length != 1) {
      throw new IllegalStateException(
          "item() requires a single-element tensor, got " + data.length);
    }
    return data[0];
  }

  private Tensor wrap(DeviceBuffer result) {
    return new Tensor(backend, result);
  }

  @Override
  public String toString() {
    return "Tensor(shape=" + shape() + ", dtype=" + dtype() + ", device=" + device() + ")";
  }
}
