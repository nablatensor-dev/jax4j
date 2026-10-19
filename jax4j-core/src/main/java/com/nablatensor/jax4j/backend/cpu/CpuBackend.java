package com.nablatensor.jax4j.backend.cpu;

import com.nablatensor.jax4j.tensor.ConvSpec;
import com.nablatensor.jax4j.tensor.DType;
import com.nablatensor.jax4j.tensor.Device;
import com.nablatensor.jax4j.tensor.DeviceType;
import com.nablatensor.jax4j.tensor.Op;
import com.nablatensor.jax4j.tensor.QuantType;
import com.nablatensor.jax4j.tensor.Shape;
import com.nablatensor.jax4j.tensor.expr.Expr;
import com.nablatensor.jax4j.tensor.spi.ComputeBackend;
import com.nablatensor.jax4j.tensor.spi.DeviceBuffer;
import com.nablatensor.jax4j.tensor.spi.QuantizedOps;
import java.lang.foreign.MemorySegment;
import java.util.EnumSet;
import java.util.Set;

/**
 * Always-available CPU backend. Kernels are simple fused Java loops today; later phases replace
 * them with {@code jdk.incubator.vector} SIMD and source-generated fused kernels for parity with
 * the GPU fusion path.
 */
public final class CpuBackend implements ComputeBackend, QuantizedOps {

  @Override
  public String name() {
    return "cpu";
  }

  @Override
  public DeviceType deviceType() {
    return DeviceType.CPU;
  }

  @Override
  public boolean isAvailable() {
    return true;
  }

  @Override
  public int priority() {
    return 0;
  }

  @Override
  public DeviceBuffer upload(float[] data, Shape shape, DType dtype, Device device) {
    if (data.length != shape.size())
      throw new IllegalArgumentException("Upload length does not match shape");
    if (dtype != DType.F32)
      throw new UnsupportedOperationException("CPU tensor upload supports F32 only, got " + dtype);
    if (device.type() != DeviceType.CPU || device.index() != 0)
      throw new IllegalArgumentException("CPU backend requires CPU device 0");
    return new CpuBuffer(data.clone(), shape, dtype, device);
  }

  @Override
  public float[] download(DeviceBuffer buffer) {
    return cpu(buffer).data.clone();
  }

  @Override
  public DeviceBuffer binary(Op op, DeviceBuffer a, DeviceBuffer b) {
    CpuBuffer left = cpu(a);
    CpuBuffer right = cpu(b);
    if (left.data.length != right.data.length) {
      throw new IllegalArgumentException(
          "shape mismatch: " + left.shape() + " vs " + right.shape());
    }
    float[] out = new float[left.data.length];
    float[] x = left.data;
    float[] y = right.data;
    switch (op) {
      case ADD -> {
        for (int i = 0; i < out.length; i++) out[i] = x[i] + y[i];
      }
      case SUB -> {
        for (int i = 0; i < out.length; i++) out[i] = x[i] - y[i];
      }
      case MUL -> {
        for (int i = 0; i < out.length; i++) out[i] = x[i] * y[i];
      }
      case DIV -> {
        for (int i = 0; i < out.length; i++) out[i] = x[i] / y[i];
      }
      case MAX -> {
        for (int i = 0; i < out.length; i++) out[i] = Math.max(x[i], y[i]);
      }
      case MIN -> {
        for (int i = 0; i < out.length; i++) out[i] = Math.min(x[i], y[i]);
      }
      default -> throw new IllegalArgumentException("not a binary op: " + op);
    }
    return like(left, out);
  }

  @Override
  public DeviceBuffer scalar(Op op, DeviceBuffer a, double value) {
    CpuBuffer left = cpu(a);
    float[] out = new float[left.data.length];
    float[] x = left.data;
    float s = (float) value;
    switch (op) {
      case ADD -> {
        for (int i = 0; i < out.length; i++) out[i] = x[i] + s;
      }
      case SUB -> {
        for (int i = 0; i < out.length; i++) out[i] = x[i] - s;
      }
      case MUL -> {
        for (int i = 0; i < out.length; i++) out[i] = x[i] * s;
      }
      case DIV -> {
        for (int i = 0; i < out.length; i++) out[i] = x[i] / s;
      }
      case MAX -> {
        for (int i = 0; i < out.length; i++) out[i] = Math.max(x[i], s);
      }
      case MIN -> {
        for (int i = 0; i < out.length; i++) out[i] = Math.min(x[i], s);
      }
      default -> throw new IllegalArgumentException("not a scalar op: " + op);
    }
    return like(left, out);
  }

  @Override
  public DeviceBuffer unary(Op op, DeviceBuffer a) {
    CpuBuffer left = cpu(a);
    float[] out = new float[left.data.length];
    float[] x = left.data;
    switch (op) {
      case NEG -> {
        for (int i = 0; i < out.length; i++) out[i] = -x[i];
      }
      case EXP -> {
        for (int i = 0; i < out.length; i++) out[i] = (float) Math.exp(x[i]);
      }
      case LOG -> {
        for (int i = 0; i < out.length; i++) out[i] = (float) Math.log(x[i]);
      }
      case SQRT -> {
        for (int i = 0; i < out.length; i++) out[i] = (float) Math.sqrt(x[i]);
      }
      case RSQRT -> {
        for (int i = 0; i < out.length; i++) out[i] = (float) (1.0 / Math.sqrt(x[i]));
      }
      case TANH -> {
        for (int i = 0; i < out.length; i++) out[i] = (float) Math.tanh(x[i]);
      }
      case SIGMOID -> {
        for (int i = 0; i < out.length; i++) out[i] = (float) (1.0 / (1.0 + Math.exp(-x[i])));
      }
      case RELU -> {
        for (int i = 0; i < out.length; i++) out[i] = Math.max(0.0f, x[i]);
      }
      case ABS -> {
        for (int i = 0; i < out.length; i++) out[i] = Math.abs(x[i]);
      }
      case SIGN -> {
        for (int i = 0; i < out.length; i++) {
          out[i] = x[i] > 0 ? 1.0f : x[i] < 0 ? -1.0f : 0.0f;
        }
      }
      default -> throw new IllegalArgumentException("not a unary op: " + op);
    }
    return like(left, out);
  }

  @Override
  public DeviceBuffer transpose(DeviceBuffer a) {
    CpuBuffer left = cpu(a);
    Shape shape = left.shape();
    if (shape.rank() != 2) {
      throw new IllegalStateException("transpose requires a rank-2 tensor, got " + shape);
    }
    int rows = shape.dim(0);
    int cols = shape.dim(1);
    float[] in = left.data;
    float[] out = new float[in.length];
    for (int r = 0; r < rows; r++) {
      for (int c = 0; c < cols; c++) {
        out[c * rows + r] = in[r * cols + c];
      }
    }
    return new CpuBuffer(out, shape.transposed(), left.dtype(), left.device());
  }

  @Override
  public DeviceBuffer matmul(DeviceBuffer a, DeviceBuffer b) {
    CpuBuffer left = cpu(a);
    CpuBuffer right = cpu(b);
    Shape ls = left.shape();
    Shape rs = right.shape();
    requireMatchingPlacement(left, right);
    if (ls.rank() != 2 || rs.rank() != 2 || ls.dim(1) != rs.dim(0)) {
      throw new IllegalArgumentException("incompatible matmul shapes: " + ls + " x " + rs);
    }
    int m = ls.dim(0);
    int k = ls.dim(1);
    int n = rs.dim(1);
    requirePositiveMatmulDimensions(m, k, n);
    float[] x = left.data;
    float[] y = right.data;
    float[] out = new float[m * n];
    for (int i = 0; i < m; i++) {
      for (int p = 0; p < k; p++) {
        float xip = x[i * k + p];
        int yRow = p * n;
        int oRow = i * n;
        for (int j = 0; j < n; j++) {
          out[oRow + j] += xip * y[yRow + j];
        }
      }
    }
    return new CpuBuffer(out, Shape.of(m, n), left.dtype(), left.device());
  }

  @Override
  public DeviceBuffer batchedMatmul(DeviceBuffer a, DeviceBuffer b) {
    CpuBuffer left = cpu(a);
    CpuBuffer right = cpu(b);
    Shape ls = left.shape();
    Shape rs = right.shape();
    requireMatchingPlacement(left, right);
    if (ls.rank() != 3 || rs.rank() != 3 || ls.dim(0) != rs.dim(0) || ls.dim(2) != rs.dim(1)) {
      throw new IllegalArgumentException("incompatible batched matmul shapes: " + ls + " x " + rs);
    }
    int batch = ls.dim(0);
    int m = ls.dim(1);
    int k = ls.dim(2);
    int n = rs.dim(2);
    requirePositiveMatmulDimensions(batch, m, k, n);
    float[] out = new float[batch * m * n];
    for (int bIndex = 0; bIndex < batch; bIndex++) {
      int leftBatch = bIndex * m * k;
      int rightBatch = bIndex * k * n;
      int outBatch = bIndex * m * n;
      for (int row = 0; row < m; row++) {
        for (int p = 0; p < k; p++) {
          float value = left.data[leftBatch + row * k + p];
          for (int col = 0; col < n; col++) {
            out[outBatch + row * n + col] += value * right.data[rightBatch + p * n + col];
          }
        }
      }
    }
    return new CpuBuffer(out, Shape.of(batch, m, n), left.dtype(), left.device());
  }

  private static void requirePositiveMatmulDimensions(int... dimensions) {
    for (int dimension : dimensions) {
      if (dimension < 1) {
        throw new IllegalArgumentException("matmul dimensions must be positive");
      }
    }
  }

  private static void requireMatchingPlacement(CpuBuffer left, CpuBuffer right) {
    if (left.dtype() != right.dtype() || !left.device().equals(right.device())) {
      throw new IllegalArgumentException("matmul operands must use the same device and dtype");
    }
  }

  @Override
  public DeviceBuffer reduceSum(DeviceBuffer a) {
    CpuBuffer left = cpu(a);
    double acc = 0;
    for (float v : left.data) {
      acc += v;
    }
    return new CpuBuffer(new float[] {(float) acc}, Shape.of(1), left.dtype(), left.device());
  }

  @Override
  public DeviceBuffer reduceMax(DeviceBuffer a) {
    CpuBuffer left = cpu(a);
    float max = Float.NEGATIVE_INFINITY;
    for (float v : left.data) {
      if (v > max) {
        max = v;
      }
    }
    return new CpuBuffer(new float[] {max}, Shape.of(1), left.dtype(), left.device());
  }

  @Override
  public DeviceBuffer sumAxis0(DeviceBuffer a) {
    Shape shape = a.shape();
    if (shape.rank() != 2) {
      throw new IllegalStateException("sumAxis0 requires a rank-2 tensor, got " + shape);
    }
    return sumAxis(a, 0, false);
  }

  @Override
  public DeviceBuffer sumAxis(DeviceBuffer a, int axis, boolean keepDims) {
    CpuBuffer left = cpu(a);
    AxisReduction reduction = AxisReduction.of(left.shape(), axis, keepDims);
    float[] out = new float[reduction.outputSize()];
    for (int outer = 0; outer < reduction.outer(); outer++) {
      for (int inner = 0; inner < reduction.inner(); inner++) {
        double total = 0;
        int base = outer * reduction.axisSize() * reduction.inner() + inner;
        for (int i = 0; i < reduction.axisSize(); i++) {
          total += left.data[base + i * reduction.inner()];
        }
        out[outer * reduction.inner() + inner] = (float) total;
      }
    }
    return new CpuBuffer(out, reduction.outputShape(), left.dtype(), left.device());
  }

  @Override
  public DeviceBuffer maxAxis(DeviceBuffer a, int axis, boolean keepDims) {
    CpuBuffer left = cpu(a);
    AxisReduction reduction = AxisReduction.of(left.shape(), axis, keepDims);
    float[] out = new float[reduction.outputSize()];
    for (int outer = 0; outer < reduction.outer(); outer++) {
      for (int inner = 0; inner < reduction.inner(); inner++) {
        float best = Float.NEGATIVE_INFINITY;
        int base = outer * reduction.axisSize() * reduction.inner() + inner;
        for (int i = 0; i < reduction.axisSize(); i++) {
          best = Math.max(best, left.data[base + i * reduction.inner()]);
        }
        out[outer * reduction.inner() + inner] = best;
      }
    }
    return new CpuBuffer(out, reduction.outputShape(), left.dtype(), left.device());
  }

  @Override
  public DeviceBuffer argMaxAxis(DeviceBuffer a, int axis) {
    CpuBuffer left = cpu(a);
    AxisReduction reduction = AxisReduction.of(left.shape(), axis, false);
    float[] out = new float[reduction.outputSize()];
    for (int outer = 0; outer < reduction.outer(); outer++) {
      for (int inner = 0; inner < reduction.inner(); inner++) {
        float best = Float.NEGATIVE_INFINITY;
        int bestIndex = 0;
        int base = outer * reduction.axisSize() * reduction.inner() + inner;
        for (int i = 0; i < reduction.axisSize(); i++) {
          float value = left.data[base + i * reduction.inner()];
          if (Float.isNaN(value)) {
            bestIndex = i;
            break;
          } else if (value > best) {
            best = value;
            bestIndex = i;
          }
        }
        out[outer * reduction.inner() + inner] = bestIndex;
      }
    }
    return new CpuBuffer(out, reduction.outputShape(), left.dtype(), left.device());
  }

  @Override
  public DeviceBuffer maxAxisBackward(DeviceBuffer upstream, DeviceBuffer input, int axis) {
    CpuBuffer grad = cpu(upstream);
    CpuBuffer in = cpu(input);
    AxisReduction reduction = AxisReduction.of(in.shape(), axis, false);
    if (grad.data.length != reduction.outputSize()) {
      throw new IllegalArgumentException(
          "maxAxisBackward gradient shape "
              + grad.shape()
              + " is incompatible with input "
              + in.shape()
              + " and axis "
              + axis);
    }
    float[] out = new float[in.data.length];
    for (int outer = 0; outer < reduction.outer(); outer++) {
      for (int inner = 0; inner < reduction.inner(); inner++) {
        int base = outer * reduction.axisSize() * reduction.inner() + inner;
        int bestIndex = 0;
        float best = Float.NEGATIVE_INFINITY;
        for (int i = 0; i < reduction.axisSize(); i++) {
          float value = in.data[base + i * reduction.inner()];
          if (Float.isNaN(value) || value > best) {
            best = value;
            bestIndex = i;
            if (Float.isNaN(value)) {
              break;
            }
          }
        }
        out[base + bestIndex * reduction.inner()] = grad.data[outer * reduction.inner() + inner];
      }
    }
    return new CpuBuffer(out, in.shape(), in.dtype(), in.device());
  }

  @Override
  public DeviceBuffer reshape(DeviceBuffer a, Shape target) {
    CpuBuffer left = cpu(a);
    // CPU storage is immutable, uploads/downloads copy, and close does not free heap arrays.
    // A shape-only view can share storage safely, avoiding copies for [B,K,F] -> [B*K,F].
    return new CpuBuffer(left.data, target, left.dtype(), left.device());
  }

  @Override
  public DeviceBuffer broadcastTo(DeviceBuffer a, Shape target) {
    CpuBuffer left = cpu(a);
    Shape src = left.shape();
    if (src.equals(target)) {
      return left;
    }
    if (target.rank() > 4) {
      throw new UnsupportedOperationException("broadcastTo supports up to rank 4, got " + target);
    }
    if (!Shape.broadcast(src, target).equals(target)) {
      throw new IllegalArgumentException("cannot broadcast shape " + src + " to " + target);
    }
    int rank = target.rank();
    int pad = rank - src.rank();
    int[] srcDims = new int[rank];
    for (int i = 0; i < rank; i++) {
      srcDims[i] = i < pad ? 1 : src.dim(i - pad);
    }
    long[] srcStrides = new long[rank];
    long stride = 1;
    for (int i = rank - 1; i >= 0; i--) {
      srcStrides[i] = srcDims[i] == 1 ? 0 : stride;
      stride *= srcDims[i];
    }
    int[] targetDims = target.dims();
    long[] targetStrides = new long[rank];
    long s = 1;
    for (int i = rank - 1; i >= 0; i--) {
      targetStrides[i] = s;
      s *= targetDims[i];
    }
    int total = (int) target.size();
    float[] out = new float[total];
    float[] in = left.data;
    for (int linear = 0; linear < total; linear++) {
      long rem = linear;
      long srcOffset = 0;
      for (int d = 0; d < rank; d++) {
        long coord = rem / targetStrides[d];
        rem = rem % targetStrides[d];
        srcOffset += coord * srcStrides[d];
      }
      out[linear] = in[(int) srcOffset];
    }
    return new CpuBuffer(out, target, left.dtype(), left.device());
  }

  private record AxisReduction(
      int outer, int axisSize, int inner, int outputSize, Shape outputShape) {

    private static AxisReduction of(Shape shape, int axis, boolean keepDims) {
      if (axis < 0 || axis >= shape.rank()) {
        throw new IllegalArgumentException("axis " + axis + " is out of bounds for shape " + shape);
      }
      int outer = 1;
      for (int i = 0; i < axis; i++) {
        outer = Math.multiplyExact(outer, shape.dim(i));
      }
      int inner = 1;
      for (int i = axis + 1; i < shape.rank(); i++) {
        inner = Math.multiplyExact(inner, shape.dim(i));
      }
      int[] outputDims;
      if (keepDims) {
        outputDims = shape.dims();
        outputDims[axis] = 1;
      } else {
        outputDims = new int[shape.rank() - 1];
        System.arraycopy(shape.dims(), 0, outputDims, 0, axis);
        System.arraycopy(shape.dims(), axis + 1, outputDims, axis, shape.rank() - axis - 1);
      }
      Shape outputShape = Shape.of(outputDims);
      return new AxisReduction(outer, shape.dim(axis), inner, outer * inner, outputShape);
    }
  }

  @Override
  public DeviceBuffer conv2d(DeviceBuffer x, DeviceBuffer w, ConvSpec spec) {
    CpuBuffer in = cpu(x);
    CpuBuffer filters = cpu(w);
    int batch = in.shape().dim(0);
    int outH = spec.outHeight();
    int outW = spec.outWidth();
    float[] src = in.data;
    float[] weights = filters.data;
    float[] out = new float[batch * spec.outputSize()];
    for (int image = 0; image < batch; image++) {
      int imageBase = image * spec.inputSize();
      for (int oc = 0; oc < spec.outChannels(); oc++) {
        int filterBase = oc * spec.weightSize();
        for (int oh = 0; oh < outH; oh++) {
          for (int ow = 0; ow < outW; ow++) {
            float acc = 0f;
            for (int ic = 0; ic < spec.inChannels(); ic++) {
              int channelBase = imageBase + ic * spec.inHeight() * spec.inWidth();
              int kernelBase = filterBase + ic * spec.kernel() * spec.kernel();
              for (int kh = 0; kh < spec.kernel(); kh++) {
                int ih = oh * spec.stride() - spec.pad() + kh;
                if (ih < 0 || ih >= spec.inHeight()) {
                  continue;
                }
                for (int kw = 0; kw < spec.kernel(); kw++) {
                  int iw = ow * spec.stride() - spec.pad() + kw;
                  if (iw < 0 || iw >= spec.inWidth()) {
                    continue;
                  }
                  acc +=
                      src[channelBase + ih * spec.inWidth() + iw]
                          * weights[kernelBase + kh * spec.kernel() + kw];
                }
              }
            }
            out[image * spec.outputSize() + (oc * outH + oh) * outW + ow] = acc;
          }
        }
      }
    }
    return new CpuBuffer(out, spec.outputShape(batch), in.dtype(), in.device());
  }

  @Override
  public DeviceBuffer conv2dGradInput(DeviceBuffer upstream, DeviceBuffer w, ConvSpec spec) {
    CpuBuffer grad = cpu(upstream);
    CpuBuffer filters = cpu(w);
    int batch = grad.shape().dim(0);
    int outH = spec.outHeight();
    int outW = spec.outWidth();
    float[] dOut = grad.data;
    float[] weights = filters.data;
    float[] dx = new float[batch * spec.inputSize()];
    for (int image = 0; image < batch; image++) {
      for (int oc = 0; oc < spec.outChannels(); oc++) {
        for (int oh = 0; oh < outH; oh++) {
          for (int ow = 0; ow < outW; ow++) {
            float g = dOut[image * spec.outputSize() + (oc * outH + oh) * outW + ow];
            if (g == 0f) {
              continue;
            }
            for (int ic = 0; ic < spec.inChannels(); ic++) {
              int channelBase = image * spec.inputSize() + ic * spec.inHeight() * spec.inWidth();
              int kernelBase = oc * spec.weightSize() + ic * spec.kernel() * spec.kernel();
              for (int kh = 0; kh < spec.kernel(); kh++) {
                int ih = oh * spec.stride() - spec.pad() + kh;
                if (ih < 0 || ih >= spec.inHeight()) {
                  continue;
                }
                for (int kw = 0; kw < spec.kernel(); kw++) {
                  int iw = ow * spec.stride() - spec.pad() + kw;
                  if (iw < 0 || iw >= spec.inWidth()) {
                    continue;
                  }
                  dx[channelBase + ih * spec.inWidth() + iw] +=
                      g * weights[kernelBase + kh * spec.kernel() + kw];
                }
              }
            }
          }
        }
      }
    }
    return new CpuBuffer(dx, spec.inputShape(batch), grad.dtype(), grad.device());
  }

  @Override
  public DeviceBuffer conv2dGradWeight(DeviceBuffer x, DeviceBuffer upstream, ConvSpec spec) {
    CpuBuffer in = cpu(x);
    CpuBuffer grad = cpu(upstream);
    int batch = in.shape().dim(0);
    int outH = spec.outHeight();
    int outW = spec.outWidth();
    float[] src = in.data;
    float[] dOut = grad.data;
    float[] dw = new float[spec.outChannels() * spec.weightSize()];
    for (int image = 0; image < batch; image++) {
      for (int oc = 0; oc < spec.outChannels(); oc++) {
        for (int oh = 0; oh < outH; oh++) {
          for (int ow = 0; ow < outW; ow++) {
            float g = dOut[image * spec.outputSize() + (oc * outH + oh) * outW + ow];
            if (g == 0f) {
              continue;
            }
            for (int ic = 0; ic < spec.inChannels(); ic++) {
              int channelBase = image * spec.inputSize() + ic * spec.inHeight() * spec.inWidth();
              int kernelBase = oc * spec.weightSize() + ic * spec.kernel() * spec.kernel();
              for (int kh = 0; kh < spec.kernel(); kh++) {
                int ih = oh * spec.stride() - spec.pad() + kh;
                if (ih < 0 || ih >= spec.inHeight()) {
                  continue;
                }
                for (int kw = 0; kw < spec.kernel(); kw++) {
                  int iw = ow * spec.stride() - spec.pad() + kw;
                  if (iw < 0 || iw >= spec.inWidth()) {
                    continue;
                  }
                  dw[kernelBase + kh * spec.kernel() + kw] +=
                      g * src[channelBase + ih * spec.inWidth() + iw];
                }
              }
            }
          }
        }
      }
    }
    return new CpuBuffer(dw, spec.weightShape(), in.dtype(), in.device());
  }

  @Override
  public DeviceBuffer reluBackward(DeviceBuffer upstream, DeviceBuffer input) {
    CpuBuffer grad = cpu(upstream);
    CpuBuffer in = cpu(input);
    float[] g = grad.data;
    float[] x = in.data;
    float[] out = new float[g.length];
    for (int i = 0; i < out.length; i++) {
      out[i] = x[i] > 0f ? g[i] : 0f;
    }
    return like(grad, out);
  }

  private static CpuBuffer cpu(DeviceBuffer buffer) {
    if (buffer instanceof CpuBuffer cpu) {
      return cpu;
    }
    throw new IllegalArgumentException("expected a CPU buffer, got " + buffer.getClass());
  }

  @Override
  public DeviceBuffer fused(Expr expr, DeviceBuffer[] inputs) {
    CpuBuffer[] buffers = new CpuBuffer[inputs.length];
    int n = -1;
    for (int i = 0; i < inputs.length; i++) {
      buffers[i] = cpu(inputs[i]);
      if (i == 0) {
        n = buffers[i].data.length;
      } else if (buffers[i].data.length != n) {
        throw new IllegalArgumentException("fused inputs must share element count");
      }
    }
    float[] out = new float[n];
    for (int i = 0; i < n; i++) {
      out[i] = evalFused(expr, buffers, i);
    }
    return like(buffers[0], out);
  }

  private static float evalFused(Expr expr, CpuBuffer[] inputs, int i) {
    return switch (expr) {
      case Expr.Input in -> inputs[in.index()].data[i];
      case Expr.Unary u -> applyUnary(u.op(), evalFused(u.in(), inputs, i));
      case Expr.Binary b ->
          applyBinary(b.op(), evalFused(b.left(), inputs, i), evalFused(b.right(), inputs, i));
      case Expr.Scalar s -> applyBinary(s.op(), evalFused(s.in(), inputs, i), (float) s.value());
    };
  }

  private static float applyBinary(Op op, float x, float y) {
    return switch (op) {
      case ADD -> x + y;
      case SUB -> x - y;
      case MUL -> x * y;
      case DIV -> x / y;
      case MAX -> Math.max(x, y);
      case MIN -> Math.min(x, y);
      default -> throw new IllegalArgumentException("not a binary/scalar op: " + op);
    };
  }

  private static float applyUnary(Op op, float x) {
    return switch (op) {
      case NEG -> -x;
      case EXP -> (float) Math.exp(x);
      case LOG -> (float) Math.log(x);
      case SQRT -> (float) Math.sqrt(x);
      case RSQRT -> (float) (1.0 / Math.sqrt(x));
      case TANH -> (float) Math.tanh(x);
      case SIGMOID -> (float) (1.0 / (1.0 + Math.exp(-x)));
      case RELU -> Math.max(0.0f, x);
      case ABS -> Math.abs(x);
      default -> throw new IllegalArgumentException("not a unary op: " + op);
    };
  }

  private static CpuBuffer like(CpuBuffer template, float[] data) {
    return new CpuBuffer(data, template.shape(), template.dtype(), template.device());
  }

  // ---- QuantizedOps (optional capability; discover with instanceof) -------

  @Override
  public Set<QuantType> supportedQuantTypes() {
    return EnumSet.of(QuantType.Q2_0, QuantType.Q4_0);
  }

  @Override
  public DeviceBuffer uploadQuantized(
      MemorySegment blocks, QuantType type, Shape logicalShape, Device device) {
    if (type != QuantType.Q2_0 && type != QuantType.Q4_0) {
      throw new IllegalArgumentException(
          "CpuBackend.uploadQuantized only supports Q2_0/Q4_0, got " + type);
    }
    byte[] bytes = new byte[Math.toIntExact(blocks.byteSize())];
    MemorySegment.copy(blocks, 0, MemorySegment.ofArray(bytes), 0, bytes.length);
    return new CpuQuantBuffer(bytes, type, logicalShape, device);
  }

  @Override
  public DeviceBuffer quantMatvec(DeviceBuffer weights, DeviceBuffer x) {
    CpuQuantBuffer w = quant(weights);
    CpuBuffer input = cpu(x);
    Shape shape = w.shape();
    if (shape.rank() != 2 || shape.dim(1) != input.data.length) {
      throw new IllegalArgumentException(
          "quantMatvec shape mismatch: " + shape + " x " + input.shape());
    }
    int rows = shape.dim(0);
    int columns = shape.dim(1);
    float[] out = new float[rows];
    switch (w.type) {
      case Q2_0 -> CpuQ2Gemm.multiply(out, input.data, w.blocks, rows, columns);
      case Q4_0 -> CpuQ4Gemm.multiply(out, input.data, w.blocks, rows, columns);
      default -> throw new IllegalArgumentException("unsupported quant type: " + w.type);
    }
    return new CpuBuffer(out, Shape.of(rows), DType.F32, w.device());
  }

  @Override
  public DeviceBuffer quantMatvecAdd(DeviceBuffer weights, DeviceBuffer x, DeviceBuffer add) {
    CpuBuffer result = cpu(quantMatvec(weights, x));
    CpuBuffer bias = cpu(add);
    float[] out = new float[result.data.length];
    for (int i = 0; i < out.length; i++) {
      out[i] = result.data[i] + bias.data[i];
    }
    return like(result, out);
  }

  @Override
  public DeviceBuffer quantGatherRow(DeviceBuffer weights, int row) {
    CpuQuantBuffer w = quant(weights);
    Shape shape = w.shape();
    if (shape.rank() != 2) {
      throw new IllegalArgumentException(
          "quantGatherRow requires a rank-2 quantized matrix, got " + shape);
    }
    int rows = shape.dim(0);
    int columns = shape.dim(1);
    float[] out = new float[columns];
    switch (w.type) {
      case Q2_0 -> CpuQ2Gemm.gather(out, w.blocks, row, rows, columns);
      case Q4_0 -> CpuQ4Gemm.gather(out, w.blocks, row, rows, columns);
      default -> throw new IllegalArgumentException("unsupported quant type: " + w.type);
    }
    return new CpuBuffer(out, Shape.of(columns), DType.F32, w.device());
  }

  private static CpuQuantBuffer quant(DeviceBuffer buffer) {
    if (buffer instanceof CpuQuantBuffer quant) {
      return quant;
    }
    throw new IllegalArgumentException("expected a CPU quantized buffer, got " + buffer.getClass());
  }
}
