package com.jax4j.tensor;

import com.jax4j.tensor.spi.*;
import com.jax4j.tensor.tree.TreeUtil;
import java.lang.reflect.*;
import java.util.*;
import java.util.function.*;

/**
 * Primitive batching with arbitrary mapped axes. A body is evaluated once, never once per example.
 */
public final class Vmap {
  private Vmap() {}

  public static Tensor apply(UnaryOperator<Tensor> body, Tensor input) {
    return apply(xs -> body.apply(xs.getFirst()), List.of(input), Arrays.asList(0), 0);
  }

  public static UnaryOperator<Tensor> of(UnaryOperator<Tensor> body) {
    return x -> apply(body, x);
  }

  /** Null input axes mean shared arguments; negative axes count from the end. */
  public static Tensor apply(
      Function<List<Tensor>, Tensor> body, List<Tensor> inputs, List<Integer> inAxes, int outAxis) {
    return applyTree(body, inputs, inAxes, List.of(outAxis));
  }

  /**
   * Input/output axes are in deterministic pytree leaf order; null output axes require an unmapped
   * result.
   */
  @SuppressWarnings("unchecked")
  public static <P, R> R applyTree(
      Function<P, R> body, P inputs, List<Integer> inAxes, List<Integer> outAxes) {
    var flat = TreeUtil.flatten(inputs);
    List<Tensor> leaves = Jit.tensorLeaves(flat.leaves());
    if (leaves.isEmpty() || inAxes.size() != leaves.size())
      throw new IllegalArgumentException("one in_axis per input leaf required");
    Tensor context = leaves.getFirst();
    for (Tensor leaf : leaves) context = new Tensor(context.common(leaf), context.buffer());
    ComputeBackend real = context.backend();
    int batch = -1;
    for (int i = 0; i < leaves.size(); i++) {
      Tensor t = leaves.get(i);
      if (!t.device().equals(leaves.getFirst().device()))
        throw new IllegalArgumentException("vmap placement mismatch");
      if (inAxes.get(i) != null) {
        int size = t.shape().dim(axis(inAxes.get(i), t.shape().rank()));
        if (batch != -1 && batch != size)
          throw new IllegalArgumentException("mapped axis sizes differ");
        batch = size;
      }
    }
    if (batch < 0) throw new IllegalArgumentException("at least one mapped argument is required");
    Batching batching = new Batching(real, batch);
    leaves.forEach(t -> batching.borrowed.add(t.buffer()));
    List<Tensor> wrapped = new ArrayList<>();
    try {
      for (int i = 0; i < leaves.size(); i++) {
        Tensor t = leaves.get(i);
        Integer a = inAxes.get(i);
        if (a == null) wrapped.add(new Tensor(batching.proxy, t.buffer()));
        else {
          int normalized = axis(a, t.shape().rank());
          Tensor physical = normalized == 0 ? t : t.moveAxis(normalized, 0);
          if (physical != t) batching.owned.add(physical.buffer());
          wrapped.add(
              new Tensor(
                  batching.proxy,
                  new Batched(
                      batching,
                      physical.buffer(),
                      Shape.of(
                          Arrays.copyOfRange(
                              physical.shape().dims(), 1, physical.shape().rank())))));
        }
      }
      var outputs = TreeUtil.flatten(body.apply((P) TreeUtil.unflatten(flat.def(), wrapped)));
      if (outAxes.size() != outputs.leaves().size())
        throw new IllegalArgumentException("one out_axis per output leaf required");
      List<Tensor> results = new ArrayList<>();
      for (int i = 0; i < outAxes.size(); i++) {
        Tensor output = (Tensor) outputs.leaves().get(i);
        Integer a = outAxes.get(i);
        DeviceBuffer b = output.buffer();
        if (a == null) {
          if (b instanceof Batched x && x.owner == batching)
            throw new IllegalArgumentException("mapped output requires an out_axis");
          results.add(new Tensor(real, b));
        } else {
          DeviceBuffer physical = batching.physical(b);
          Tensor t = new Tensor(real, physical);
          int normalized = axis(a, t.shape().rank());
          results.add(normalized == 0 ? t : t.moveAxis(0, normalized));
        }
      }
      Set<DeviceBuffer> keep = Collections.newSetFromMap(new IdentityHashMap<>());
      results.forEach(t -> keep.add(t.buffer()));
      batching.cleanup(keep);
      return (R) TreeUtil.unflatten(outputs.def(), results);
    } catch (RuntimeException | Error e) {
      batching.cleanup(Set.of());
      throw e;
    }
  }

  private static int axis(int a, int rank) {
    int n = a < 0 ? a + rank : a;
    if (n < 0 || n >= rank) throw new IllegalArgumentException("axis out of bounds");
    return n;
  }

  private record Batched(Batching owner, DeviceBuffer physical, Shape shape)
      implements TracingBuffer {
    public ComputeBackend tracingBackend() {
      return owner.proxy;
    }

    public DType dtype() {
      return physical.dtype();
    }

    public Device device() {
      return physical.device();
    }
  }

  private static final class Batching implements InvocationHandler {
    final ComputeBackend real, proxy;
    final int batch;
    final List<DeviceBuffer> owned = new ArrayList<>();
    final Set<DeviceBuffer> borrowed = Collections.newSetFromMap(new IdentityHashMap<>());

    Batching(ComputeBackend real, int batch) {
      this.real = real;
      this.batch = batch;
      proxy =
          (ComputeBackend)
              Proxy.newProxyInstance(
                  ComputeBackend.class.getClassLoader(),
                  new Class<?>[] {TracingBackend.class},
                  this);
    }

    Shape batchShape(Shape sample) {
      int[] dims = new int[sample.rank() + 1];
      dims[0] = batch;
      System.arraycopy(sample.dims(), 0, dims, 1, sample.rank());
      return Shape.of(dims);
    }

    DeviceBuffer own(DeviceBuffer b) {
      if (!borrowed.contains(b)) owned.add(b);
      return b;
    }

    DeviceBuffer physical(DeviceBuffer b) {
      if (b instanceof Batched x && x.owner == this) return x.physical();
      if (owned.stream().noneMatch(value -> value == b)) borrowed.add(b);
      int[] dims = new int[b.shape().rank() + 1];
      dims[0] = 1;
      System.arraycopy(b.shape().dims(), 0, dims, 1, b.shape().rank());
      DeviceBuffer expanded = own(real.reshape(b, Shape.of(dims)));
      return own(real.broadcastTo(expanded, batchShape(b.shape())));
    }

    Tensor t(DeviceBuffer b) {
      return new Tensor(real, b);
    }

    DeviceBuffer result(Tensor t) {
      return own(t.buffer());
    }

    public Object invoke(Object p, Method method, Object[] raw) throws Throwable {
      String name = method.getName();
      Object[] args = raw == null ? new Object[0] : raw.clone();
      if (name.equals("parentBackend")) return real;
      if (name.equals("download"))
        throw new IllegalStateException("Host reads inside vmap are unsupported");
      if (name.equals("release")) return null;
      boolean mapped = false;
      for (Object arg : args) {
        if (arg instanceof Batched x && x.owner == this) mapped = true;
        if (arg instanceof DeviceBuffer[] bs)
          for (var b : bs) if (b instanceof Batched x && x.owner == this) mapped = true;
      }
      if (!mapped) {
        for (Object arg : args) {
          if (arg instanceof DeviceBuffer b && owned.stream().noneMatch(value -> value == b))
            borrowed.add(b);
          else if (arg instanceof DeviceBuffer[] bs)
            for (DeviceBuffer b : bs)
              if (owned.stream().noneMatch(value -> value == b)) borrowed.add(b);
        }
        Object r = call(method, args);
        return r instanceof DeviceBuffer b ? own(b) : r;
      }
      if (name.equals("custom")) {
        CustomDerivative custom = (CustomDerivative) args[1];
        DeviceBuffer input = (DeviceBuffer) args[0];
        CustomDerivative mappedRule =
            new CustomDerivative(
                x -> Vmap.apply(custom.forward, x),
                custom.jvp == null
                    ? null
                    : (x, t) ->
                        Vmap.apply(
                            ts -> custom.jvp.apply(ts.get(0), ts.get(1)),
                            List.of(x, t),
                            List.of(0, 0),
                            0),
                custom.vjp == null
                    ? null
                    : (x, g) ->
                        Vmap.apply(
                            ts -> custom.vjp.apply(ts.get(0), ts.get(1)),
                            List.of(x, g),
                            List.of(0, 0),
                            0));
        return new Batched(this, own(real.custom(physical(input), mappedRule)), input.shape());
      }
      if (name.equals("program")) {
        DeviceProgram program = (DeviceProgram) args[0];
        DeviceBuffer[] originals = (DeviceBuffer[]) args[1],
            physical = new DeviceBuffer[originals.length];
        for (int i = 0; i < originals.length; i++) physical[i] = physical(originals[i]);
        return new Batched(
            this, own(real.program(program.batched(batch), physical)), program.outputShape());
      }
      if (name.equals("fused")) return InvocationHandler.invokeDefault(p, method, args);
      if (name.equals("binary") || name.equals("unary") || name.equals("scalar")) {
        DeviceBuffer first = (DeviceBuffer) args[1];
        args[1] = physical(first);
        if (name.equals("binary")) args[2] = physical((DeviceBuffer) args[2]);
        return new Batched(this, own((DeviceBuffer) call(method, args)), first.shape());
      }
      if (name.equals("stackAxis0")) {
        DeviceBuffer[] bs = (DeviceBuffer[]) args[0];
        DeviceBuffer[] ps = new DeviceBuffer[bs.length];
        for (int i = 0; i < bs.length; i++) ps[i] = physical(bs[i]);
        Tensor stacked = t(own(real.stackAxis0(ps)));
        Tensor output = stacked.moveAxis(0, 1);
        int[] dims = new int[bs[0].shape().rank() + 1];
        dims[0] = bs.length;
        System.arraycopy(bs[0].shape().dims(), 0, dims, 1, dims.length - 1);
        return new Batched(this, own(output.buffer()), Shape.of(dims));
      }
      DeviceBuffer a = (DeviceBuffer) args[0];
      Shape sample = a.shape();
      DeviceBuffer pa = physical(a), r;
      switch (name) {
        case "reshape" -> {
          sample = (Shape) args[1];
          r = real.reshape(pa, batchShape(sample));
        }
        case "broadcastTo" -> {
          sample = (Shape) args[1];
          int[] dims = new int[sample.rank() + 1];
          Arrays.fill(dims, 1);
          dims[0] = batch;
          System.arraycopy(
              a.shape().dims(), 0, dims, dims.length - a.shape().rank(), a.shape().rank());
          r = real.broadcastTo(own(real.reshape(pa, Shape.of(dims))), batchShape(sample));
        }
        case "transpose" -> {
          if (sample.rank() != 2)
            throw new IllegalArgumentException("transpose expects matrix samples");
          sample = Shape.of(sample.dim(1), sample.dim(0));
          r = result(t(pa).permute(0, 2, 1));
        }
        case "matmul" -> {
          DeviceBuffer b = (DeviceBuffer) args[1];
          sample = Shape.of(a.shape().dim(0), b.shape().dim(1));
          r = real.batchedMatmul(pa, physical(b));
        }
        case "batchedMatmul" -> {
          DeviceBuffer b = (DeviceBuffer) args[1];
          int n = Math.max(a.shape().dim(0), b.shape().dim(0));
          Tensor x = t(pa).broadcastTo(Shape.of(batch, n, a.shape().dim(1), a.shape().dim(2)));
          Tensor y =
              t(physical(b)).broadcastTo(Shape.of(batch, n, b.shape().dim(1), b.shape().dim(2)));
          own(x.buffer());
          own(y.buffer());
          DeviceBuffer xx =
              own(
                  real.reshape(
                      x.buffer(), Shape.of(batch * n, a.shape().dim(1), a.shape().dim(2))));
          DeviceBuffer yy =
              own(
                  real.reshape(
                      y.buffer(), Shape.of(batch * n, b.shape().dim(1), b.shape().dim(2))));
          sample = Shape.of(n, a.shape().dim(1), b.shape().dim(2));
          r = real.reshape(own(real.batchedMatmul(xx, yy)), batchShape(sample));
        }
        case "reduceSum", "reduceMax" -> {
          DeviceBuffer flat =
              own(real.reshape(pa, Shape.of(batch, Math.toIntExact(sample.size()))));
          sample = Shape.of(1);
          r = name.equals("reduceSum") ? real.sumAxis(flat, 1, true) : real.maxAxis(flat, 1, true);
        }
        case "sumAxis0", "sumAxis", "maxAxis", "argMaxAxis" -> {
          int ax = name.equals("sumAxis0") ? 0 : (int) args[1];
          boolean keep = args.length > 2 && (boolean) args[2];
          sample = reduced(sample, ax, keep);
          r =
              switch (name) {
                case "sumAxis0", "sumAxis" -> real.sumAxis(pa, ax + 1, keep);
                case "maxAxis" -> real.maxAxis(pa, ax + 1, keep);
                default -> real.argMaxAxis(pa, ax + 1);
              };
        }
        case "reluBackward" -> r = real.reluBackward(pa, physical((DeviceBuffer) args[1]));
        case "maxAxisBackward" -> {
          DeviceBuffer b = (DeviceBuffer) args[1];
          sample = b.shape();
          r = real.maxAxisBackward(pa, physical(b), (int) args[2] + 1);
        }
        case "sliceAxis0" -> {
          int ix = (int) args[1];
          if (ix < 0 || ix >= sample.dim(0)) throw new IllegalArgumentException("slice index");
          sample = Shape.of(Arrays.copyOfRange(sample.dims(), 1, sample.rank()));
          int size = Math.toIntExact(sample.size());
          int[] indices = new int[batch * size];
          for (int i = 0; i < indices.length; i++)
            indices[i] = (i / size) * Math.toIntExact(a.shape().size()) + ix * size + i % size;
          r = real.reindex(pa, batchShape(sample), indices);
        }
        case "reindex", "scatterAdd" -> {
          sample = (Shape) args[1];
          int[] original = (int[]) args[2], indices = new int[batch * original.length];
          int stride = Math.toIntExact(name.equals("reindex") ? a.shape().size() : sample.size());
          for (int i = 0; i < indices.length; i++)
            indices[i] =
                original[i % original.length] < 0
                    ? -1
                    : original[i % original.length] + (i / original.length) * stride;
          r =
              name.equals("reindex")
                  ? real.reindex(pa, batchShape(sample), indices)
                  : real.scatterAdd(pa, batchShape(sample), indices);
        }
        case "conv2d", "conv2dGradInput", "conv2dGradWeight" -> {
          ConvSpec spec = (ConvSpec) args[2];
          DeviceBuffer b = (DeviceBuffer) args[1];
          Tensor output = convolution(name, t(pa), t(physical(b)), spec);
          sample = Shape.of(Arrays.copyOfRange(output.shape().dims(), 1, output.shape().rank()));
          r = output.buffer();
        }
        default -> throw new UnsupportedOperationException("No vmap batching rule for " + name);
      }
      return new Batched(this, own(r), sample);
    }

    Object call(Method method, Object[] args) throws Throwable {
      try {
        return method.invoke(real, args);
      } catch (InvocationTargetException e) {
        throw e.getCause();
      }
    }

    Shape reduced(Shape s, int axis, boolean keep) {
      int[] dims = s.dims();
      if (keep) {
        dims[axis] = 1;
        return Shape.of(dims);
      }
      int[] out = new int[dims.length - 1];
      System.arraycopy(dims, 0, out, 0, axis);
      System.arraycopy(dims, axis + 1, out, axis, dims.length - axis - 1);
      return Shape.of(out);
    }

    Tensor track(Tensor value) {
      own(value.buffer());
      return value;
    }

    Tensor convolution(String op, Tensor a, Tensor b, ConvSpec s) {
      int n = a.shape().dim(1),
          sp = s.outHeight() * s.outWidth(),
          k = s.weightSize(),
          o = s.outChannels();
      if (op.equals("conv2dGradInput")) {
        Tensor g = track(track(a.reshape(batch, n, o, sp)).permute(0, 1, 3, 2));
        Tensor patches = track(track(g.reshape(batch, n * sp, o)).batchedMatmul(b));
        return track(patches.scatterAdd(Shape.of(batch, n, s.inputSize()), patchMap(s, n)));
      }
      Tensor patches = track(a.reindex(Shape.of(batch, n * sp, k), patchMap(s, n)));
      if (op.equals("conv2dGradWeight")) {
        Tensor g =
            track(
                track(track(b.reshape(batch, n, o, sp)).permute(0, 2, 1, 3))
                    .reshape(batch, o, n * sp));
        return track(g.batchedMatmul(patches));
      }
      Tensor wt = track(b.permute(0, 2, 1));
      Tensor out = track(patches.batchedMatmul(wt));
      return track(
          track(track(out.reshape(batch, n, sp, o)).permute(0, 1, 3, 2))
              .reshape(batch, n, s.outputSize()));
    }

    int[] patchMap(ConvSpec s, int n) {
      int sp = s.outHeight() * s.outWidth(), k = s.weightSize();
      int[] map = new int[batch * n * sp * k];
      int i = 0;
      for (int z = 0; z < batch; z++)
        for (int image = 0; image < n; image++)
          for (int y = 0; y < s.outHeight(); y++)
            for (int x = 0; x < s.outWidth(); x++)
              for (int c = 0; c < s.inChannels(); c++)
                for (int ky = 0; ky < s.kernel(); ky++)
                  for (int kx = 0; kx < s.kernel(); kx++) {
                    int iy = y * s.stride() - s.pad() + ky, ix = x * s.stride() - s.pad() + kx;
                    map[i++] =
                        iy < 0 || ix < 0 || iy >= s.inHeight() || ix >= s.inWidth()
                            ? -1
                            : (z * n + image) * s.inputSize()
                                + (c * s.inHeight() + iy) * s.inWidth()
                                + ix;
                  }
      return map;
    }

    void cleanup(Set<DeviceBuffer> keep) {
      Set<DeviceBuffer> freed = Collections.newSetFromMap(new IdentityHashMap<>());
      for (var b : owned) if (!keep.contains(b) && freed.add(b)) real.release(b);
    }
  }
}
