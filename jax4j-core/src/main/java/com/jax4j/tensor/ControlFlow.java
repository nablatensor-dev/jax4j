package com.jax4j.tensor;

import com.jax4j.tensor.spi.*;
import java.util.*;
import java.util.function.*;

/** Fixed-shape structured regions. Predicates are scalar F32 tensors: zero is false. */
public final class ControlFlow {
  private ControlFlow() {}

  public record ScanStep(Tensor carry, Tensor output) {}

  public record ScanResult(Tensor carry, Tensor outputs) {}

  public static Tensor cond(
      Tensor predicate,
      UnaryOperator<Tensor> onTrue,
      UnaryOperator<Tensor> onFalse,
      Tensor operand) {
    if (predicate.shape().size() != 1)
      throw new IllegalArgumentException("cond predicate must be scalar");
    var yes =
        DeviceProgram.Graph.capture(
            operand.backend(), List.of(operand), xs -> List.of(onTrue.apply(xs.getFirst())));
    var no =
        DeviceProgram.Graph.capture(
            operand.backend(), List.of(operand), xs -> List.of(onFalse.apply(xs.getFirst())));
    if (!yes.outputs.getFirst().shape().equals(no.outputs.getFirst().shape()))
      throw new IllegalArgumentException("cond branch shapes differ");
    var program =
        new DeviceProgram(
            DeviceProgram.Kind.COND, yes, no, yes.outputs.getFirst().shape(), null, 0);
    return execute(program, List.of(predicate, operand), operand.backend());
  }

  public static Tensor whileLoop(
      UnaryOperator<Tensor> predicate, UnaryOperator<Tensor> body, Tensor initial) {
    var test =
        DeviceProgram.Graph.capture(
            initial.backend(), List.of(initial), xs -> List.of(predicate.apply(xs.getFirst())));
    if (test.outputs.getFirst().shape().size() != 1)
      throw new IllegalArgumentException("while predicate must be scalar");
    var step =
        DeviceProgram.Graph.capture(
            initial.backend(), List.of(initial), xs -> List.of(body.apply(xs.getFirst())));
    if (!step.outputs.getFirst().shape().equals(initial.shape()))
      throw new IllegalArgumentException("while carry shape changed");
    return execute(
        new DeviceProgram(DeviceProgram.Kind.WHILE, test, step, initial.shape(), null, 0),
        List.of(initial),
        initial.backend());
  }

  /** Scan's body is captured once; the iteration dimension remains a device loop. */
  public static ScanResult scan(
      BiFunction<Tensor, Tensor, ScanStep> body, Tensor initial, Tensor xs) {
    if (xs.shape().rank() < 1) throw new IllegalArgumentException("scan requires a leading axis");
    Shape sampleShape = Shape.of(Arrays.copyOfRange(xs.shape().dims(), 1, xs.shape().rank()));
    DeviceBuffer template =
        new DeviceBuffer() {
          public Shape shape() {
            return sampleShape;
          }

          public DType dtype() {
            return xs.dtype();
          }

          public Device device() {
            return xs.device();
          }
        };
    var graph =
        DeviceProgram.Graph.capture(
            initial.backend(),
            List.of(initial, new Tensor(initial.backend(), template)),
            args -> {
              ScanStep result = body.apply(args.get(0), args.get(1));
              return List.of(result.carry(), result.output());
            });
    if (!graph.outputs.getFirst().shape().equals(initial.shape()))
      throw new IllegalArgumentException("scan carry shape changed");
    int count = xs.shape().dim(0);
    var program =
        new DeviceProgram(
            DeviceProgram.Kind.SCAN, graph, null, initial.shape(), sampleShape, count);
    Shape outSample = graph.outputs.get(1).shape();
    int[] dims = new int[outSample.rank() + 1];
    dims[0] = count;
    System.arraycopy(outSample.dims(), 0, dims, 1, outSample.rank());
    try (Tensor packed = execute(program, List.of(initial, xs), initial.backend())) {
      int[] carry = new int[Math.toIntExact(initial.shape().size())],
          ys = new int[Math.toIntExact(count * outSample.size())];
      for (int i = 0; i < carry.length; i++) carry[i] = i;
      for (int i = 0; i < ys.length; i++) ys[i] = carry.length + i;
      return new ScanResult(
          packed.reindex(initial.shape(), carry), packed.reindex(Shape.of(dims), ys));
    }
  }

  public record TreeScanStep<C, Y>(C carry, Y output) {}

  public record TreeScanResult<C, Y>(C carry, Y outputs) {}

  private record TreeShape(com.jax4j.tensor.tree.TreeDef def, List<Shape> shapes) {
    static TreeShape of(Object tree) {
      var flat = com.jax4j.tensor.tree.TreeUtil.flatten(tree);
      return new TreeShape(
          flat.def(), Jit.tensorLeaves(flat.leaves()).stream().map(Tensor::shape).toList());
    }

    void check(Object tree) {
      if (!equals(of(tree)))
        throw new IllegalArgumentException("control-flow tree structure or shapes changed");
    }

    @SuppressWarnings("unchecked")
    <T> T unpack(Tensor packed) {
      List<Tensor> leaves = new ArrayList<>();
      int offset = 0;
      for (Shape shape : shapes) {
        int[] indices = new int[Math.toIntExact(shape.size())];
        for (int i = 0; i < indices.length; i++) indices[i] = offset + i;
        leaves.add(packed.reindex(shape, indices));
        offset += indices.length;
      }
      return (T) com.jax4j.tensor.tree.TreeUtil.unflatten(def, leaves);
    }

    @SuppressWarnings("unchecked")
    <T> T unpackBatch(Tensor packed, int batch) {
      List<Tensor> leaves = new ArrayList<>();
      int total = shapes.stream().mapToInt(x -> Math.toIntExact(x.size())).sum(), offset = 0;
      for (Shape shape : shapes) {
        int size = Math.toIntExact(shape.size());
        int[] indices = new int[batch * size], dims = new int[shape.rank() + 1];
        dims[0] = batch;
        System.arraycopy(shape.dims(), 0, dims, 1, shape.rank());
        for (int i = 0; i < indices.length; i++)
          indices[i] = (i / size) * total + offset + i % size;
        leaves.add(packed.reindex(Shape.of(dims), indices));
        offset += size;
      }
      return (T) com.jax4j.tensor.tree.TreeUtil.unflatten(def, leaves);
    }
  }

  static Tensor pack(Object tree) {
    List<Tensor> leaves =
        Jit.tensorLeaves(com.jax4j.tensor.tree.TreeUtil.flatten(tree).leaves());
    if (leaves.isEmpty()) throw new IllegalArgumentException("control-flow tree requires tensors");
    int total = leaves.stream().mapToInt(t -> Math.toIntExact(t.shape().size())).sum(), offset = 0;
    Tensor packed = null;
    for (Tensor leaf : leaves) {
      int[] indices = new int[total];
      Arrays.fill(indices, -1);
      int size = Math.toIntExact(leaf.shape().size());
      for (int i = 0; i < size; i++) indices[offset + i] = i;
      Tensor part = leaf.reindex(Shape.of(total), indices);
      offset += size;
      if (packed == null) packed = part;
      else {
        Tensor next = packed.add(part);
        packed.close();
        part.close();
        packed = next;
      }
    }
    return packed;
  }

  public static <P> P condTree(
      Tensor predicate, UnaryOperator<P> yes, UnaryOperator<P> no, P operand) {
    TreeShape input = TreeShape.of(operand);
    TreeShape[] output = {null};
    UnaryOperator<Tensor> yesPacked =
        x -> {
          P value = yes.apply(input.unpack(x));
          output[0] = TreeShape.of(value);
          return pack(value);
        };
    UnaryOperator<Tensor> noPacked =
        x -> {
          P value = no.apply(input.unpack(x));
          output[0].check(value);
          return pack(value);
        };
    try (Tensor initial = pack(operand);
        Tensor result = cond(predicate, yesPacked, noPacked, initial)) {
      return output[0].unpack(result);
    }
  }

  public static <P> P whileTree(Function<P, Tensor> predicate, UnaryOperator<P> body, P initial) {
    TreeShape spec = TreeShape.of(initial);
    try (Tensor packed = pack(initial);
        Tensor result =
            whileLoop(
                x -> predicate.apply(spec.unpack(x)),
                x -> {
                  P next = body.apply(spec.unpack(x));
                  spec.check(next);
                  return pack(next);
                },
                packed)) {
      return spec.unpack(result);
    }
  }

  public static <C, X, Y> TreeScanResult<C, Y> scanTree(
      BiFunction<C, X, TreeScanStep<C, Y>> body, C initial, X xs) {
    var flat = com.jax4j.tensor.tree.TreeUtil.flatten(xs);
    List<Tensor> leaves = Jit.tensorLeaves(flat.leaves());
    if (leaves.isEmpty()) throw new IllegalArgumentException("scan requires input tensors");
    int length = leaves.getFirst().shape().dim(0);
    List<Shape> samples = new ArrayList<>();
    int total = 0;
    for (Tensor t : leaves) {
      if (t.shape().rank() < 1 || t.shape().dim(0) != length)
        throw new IllegalArgumentException("scan axis mismatch");
      Shape sample = Shape.of(Arrays.copyOfRange(t.shape().dims(), 1, t.shape().rank()));
      samples.add(sample);
      total += Math.toIntExact(sample.size());
    }
    TreeShape inputs = new TreeShape(flat.def(), samples), carry = TreeShape.of(initial);
    TreeShape[] output = {null};
    Tensor stacked = null;
    int offset = 0;
    for (int k = 0; k < leaves.size(); k++) {
      int size = Math.toIntExact(samples.get(k).size());
      int[] indices = new int[length * total];
      Arrays.fill(indices, -1);
      for (int b = 0; b < length; b++)
        for (int j = 0; j < size; j++) indices[b * total + offset + j] = b * size + j;
      Tensor part = leaves.get(k).reindex(Shape.of(length, total), indices);
      offset += size;
      if (stacked == null) stacked = part;
      else {
        Tensor next = stacked.add(part);
        stacked.close();
        part.close();
        stacked = next;
      }
    }
    try (Tensor scanInputs = stacked;
        Tensor packed = pack(initial)) {
      ScanResult result =
          scan(
              (c, x) -> {
                TreeScanStep<C, Y> next = body.apply(carry.unpack(c), inputs.unpack(x));
                carry.check(next.carry());
                output[0] = TreeShape.of(next.output());
                return new ScanStep(pack(next.carry()), pack(next.output()));
              },
              packed,
              scanInputs);
      try (Tensor finalCarry = result.carry();
          Tensor ys = result.outputs()) {
        return new TreeScanResult<>(carry.unpack(finalCarry), output[0].unpackBatch(ys, length));
      }
    }
  }

  private static Tensor execute(
      DeviceProgram program, List<Tensor> explicit, ComputeBackend backend) {
    List<DeviceBuffer> inputs = new ArrayList<>();
    for (Tensor t : explicit) {
      if (!t.device().equals(explicit.getFirst().device()))
        throw new IllegalArgumentException("control-flow placement mismatch");
      inputs.add(t.buffer());
    }
    inputs.addAll(program.captures);
    Tensor context = explicit.getFirst();
    for (Tensor t : explicit) context = new Tensor(context.common(t), context.buffer());
    for (DeviceBuffer b : program.captures)
      if (b instanceof TracingBuffer trace)
        context =
            new Tensor(context.common(new Tensor(trace.tracingBackend(), b)), context.buffer());
    ComputeBackend selected = context.backend();
    return new Tensor(selected, selected.program(program, inputs.toArray(DeviceBuffer[]::new)));
  }
}
