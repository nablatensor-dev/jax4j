package com.nablatensor.jax4j.tensor;

import com.nablatensor.jax4j.tensor.spi.*;
import java.lang.reflect.*;
import java.util.*;
import java.util.function.*;

/** Typed, fixed-shape device regions used by structured control flow. */
public final class DeviceProgram {
  enum Kind {
    GRAPH,
    COND,
    WHILE,
    SCAN
  }

  final Kind kind;
  final Graph first, second;
  final Shape stateShape, sampleShape, resultShape;
  final int length;
  final List<DeviceBuffer> captures = new ArrayList<>();
  final DeviceProgram mappedBase;
  final int mappedBatch;
  final DeviceProgram original;
  final boolean forwardDerivative;
  final Graph dFirst, dSecond;
  final List<Shape> primalShapes;

  DeviceProgram(
      Kind kind, Graph first, Graph second, Shape stateShape, Shape sampleShape, int length) {
    mappedBase = null;
    mappedBatch = 0;
    original = null;
    forwardDerivative = false;
    dFirst = null;
    dSecond = null;
    primalShapes = List.of();
    this.kind = kind;
    this.first = first;
    this.second = second;
    this.stateShape = stateShape;
    this.sampleShape = sampleShape;
    this.length = length;
    for (var b : first.captures) addCapture(b);
    if (second != null) for (var b : second.captures) addCapture(b);
    resultShape =
        kind == Kind.SCAN
            ? Shape.of(
                Math.toIntExact(stateShape.size() + length * first.outputs.get(1).shape().size()))
            : stateShape;
  }

  private DeviceProgram(DeviceProgram base, int batch) {
    mappedBase = base;
    mappedBatch = batch;
    original = null;
    forwardDerivative = false;
    dFirst = null;
    dSecond = null;
    primalShapes = List.of();
    kind = base.kind;
    first = base.first;
    second = base.second;
    stateShape = base.stateShape;
    sampleShape = base.sampleShape;
    length = base.length;
    int[] dims = new int[base.outputShape().rank() + 1];
    dims[0] = batch;
    System.arraycopy(base.outputShape().dims(), 0, dims, 1, dims.length - 1);
    resultShape = Shape.of(dims);
  }

  DeviceProgram batched(int batch) {
    return new DeviceProgram(this, batch);
  }

  private List<Tensor> sampleTemplates(List<Tensor> inputs) {
    return inputs.stream()
        .map(
            t ->
                new Tensor(
                    t.backend(),
                    new DeviceBuffer() {
                      public Shape shape() {
                        return Shape.of(Arrays.copyOfRange(t.shape().dims(), 1, t.shape().rank()));
                      }

                      public DType dtype() {
                        return t.dtype();
                      }

                      public Device device() {
                        return t.device();
                      }
                    }))
        .toList();
  }

  private DeviceProgram(DeviceProgram base, boolean forward, List<Tensor> inputs) {

    if (base.kind == Kind.WHILE && !forward)
      throw new UnsupportedOperationException(
          "Dynamic while loops support forward-mode differentiation only");
    mappedBase = null;
    mappedBatch = 0;
    original = base;
    forwardDerivative = forward;
    kind = base.kind;
    first = base.first;
    second = base.second;
    stateShape = base.stateShape;
    sampleShape = base.sampleShape;
    length = base.length;
    captures.addAll(base.captures);
    ComputeBackend backend = inputs.getFirst().backend();
    dFirst = kind == Kind.WHILE ? null : first.derivative(forward, backend);
    dSecond = second == null ? null : second.derivative(forward, backend);
    primalShapes = inputs.stream().map(Tensor::shape).toList();
    resultShape =
        forward
            ? base.resultShape
            : Shape.of(Math.toIntExact(primalShapes.stream().mapToLong(Shape::size).sum()));
  }

  private static DeviceProgram derive(DeviceProgram base, boolean forward, List<Tensor> inputs) {
    if (base.original != null) {
      DeviceProgram previous = base;
      Graph graph =
          Graph.capture(
              inputs.getFirst().backend(),
              inputs,
              args -> List.of(previous.expandDerivative(args)));
      base = new DeviceProgram(Kind.GRAPH, graph, null, graph.outputs.getFirst().shape(), null, 0);
    }
    return new DeviceProgram(base, forward, inputs);
  }

  private List<Tensor> regionValues(Graph graph, List<Tensor> roots, List<Tensor> explicit) {
    List<Tensor> args = new ArrayList<>(explicit);
    for (var c : graph.captures) args.add(roots.get(explicitInputs() + captureIndex(c)));
    return args;
  }

  private List<Tensor> expandedCall(
      Graph graph,
      Graph derivative,
      List<Tensor> roots,
      List<Tensor> explicit,
      List<Tensor> dualOrSeeds) {
    List<Tensor> args = regionValues(graph, roots, explicit);
    args.addAll(dualOrSeeds);
    if (forwardDerivative)
      for (var c : graph.captures)
        args.add(roots.get(primalShapes.size() + explicitInputs() + captureIndex(c)));
    return derivative.apply(args);
  }

  /**
   * Higher-order differentiation expands static scans; ordinary execution and first derivatives
   * retain loops.
   */
  private Tensor expandDerivative(List<Tensor> roots) {
    int n = primalShapes.size();
    if (kind == Kind.GRAPH) {
      List<Tensor> args = regionValues(first, roots, roots.subList(0, explicitInputs()));
      if (forwardDerivative) {
        for (int i = 0; i < explicitInputs(); i++) args.add(roots.get(n + i));
        for (var c : first.captures) args.add(roots.get(n + explicitInputs() + captureIndex(c)));
        return dFirst.apply(args).get(1);
      }
      args.add(roots.get(n));
      return ControlFlow.pack(dFirst.apply(args));
    }
    if (kind == Kind.COND) {
      return ControlFlow.cond(
          roots.get(0),
          x -> expandedBranch(first, dFirst, roots, x),
          x -> expandedBranch(second, dSecond, roots, x),
          roots.get(1));
    }
    if (kind == Kind.WHILE) {
      List<Tensor> result =
          ControlFlow.whileTree(
              (List<Tensor> state) ->
                  first.apply(regionValues(first, roots, List.of(state.get(0)))).getFirst(),
              state ->
                  expandedCall(
                      second, dSecond, roots, List.of(state.get(0)), List.of(state.get(1))),
              List.of(roots.get(0), roots.get(n)));
      return result.get(1);
    }
    List<Tensor> states = new ArrayList<>();
    Tensor carry = roots.get(0), dc = forwardDerivative ? roots.get(n) : null;
    List<Tensor> ys = new ArrayList<>();
    for (int step = 0; step < length; step++) {
      states.add(carry);
      Tensor x = roots.get(1).sliceAxis0(step);
      if (forwardDerivative) {
        var result =
            expandedCall(
                first,
                dFirst,
                roots,
                List.of(carry, x),
                List.of(dc, roots.get(n + 1).sliceAxis0(step)));
        carry = result.get(0);
        dc = result.get(2);
        ys.add(result.get(3));
      } else carry = first.apply(regionValues(first, roots, List.of(carry, x))).getFirst();
    }
    if (forwardDerivative) {
      List<Tensor> output = new ArrayList<>(List.of(dc));
      output.addAll(ys);
      return ControlFlow.pack(output);
    }
    List<Tensor> gradients = new ArrayList<>();
    for (int i = 0; i < n; i++) gradients.add(roots.get(i).fullLike(0));
    Tensor seed = roots.get(n);
    int[] ci = new int[Math.toIntExact(stateShape.size())];
    for (int i = 0; i < ci.length; i++) ci[i] = i;
    Tensor gc = seed.reindex(stateShape, ci);
    List<Tensor> dxs = new ArrayList<>(Collections.nCopies(length, null));
    Shape y = first.outputs.get(1).shape();
    for (int step = length - 1; step >= 0; step--) {
      int[] yi = new int[Math.toIntExact(y.size())];
      for (int i = 0; i < yi.length; i++)
        yi[i] = Math.toIntExact(stateShape.size() + step * y.size() + i);
      Tensor gy = seed.reindex(y, yi);
      var result =
          expandedCall(
              first,
              dFirst,
              roots,
              List.of(states.get(step), roots.get(1).sliceAxis0(step)),
              List.of(gc, gy));
      gc = result.get(0);
      dxs.set(step, result.get(1));
      for (int i = 0; i < first.captures.size(); i++) {
        int index = explicitInputs() + captureIndex(first.captures.get(i));
        gradients.set(index, gradients.get(index).add(result.get(2 + i)));
      }
    }
    gradients.set(0, gc);
    gradients.set(1, dxs.isEmpty() ? roots.get(1).fullLike(0) : Tensor.stackAxis0(dxs));
    return ControlFlow.pack(gradients);
  }

  private Tensor expandedBranch(Graph graph, Graph derivative, List<Tensor> roots, Tensor operand) {
    int n = primalShapes.size();
    List<Tensor> result =
        expandedCall(
            graph,
            derivative,
            roots,
            List.of(operand),
            List.of(roots.get(forwardDerivative ? n + 1 : n)));
    if (forwardDerivative) return result.get(1);
    List<Tensor> gradients = new ArrayList<>();
    for (int i = 0; i < n; i++) gradients.add(roots.get(i).fullLike(0));
    List<Integer> indices = regionIndices(graph, 1);
    for (int i = 0; i < indices.size(); i++) {
      int index = indices.get(i);
      gradients.set(index, gradients.get(index).add(result.get(i)));
    }
    return ControlFlow.pack(gradients);
  }

  Tensor push(List<Tensor> inputs, List<Tensor> tangents) {
    DeviceProgram derived =
        mappedBase == null
            ? derive(this, true, inputs)
            : derive(mappedBase, true, sampleTemplates(inputs)).batched(mappedBatch);
    List<Tensor> all = new ArrayList<>(inputs);
    all.addAll(tangents);
    return new Tensor(
        inputs.getFirst().backend(),
        inputs
            .getFirst()
            .backend()
            .program(derived, all.stream().map(Tensor::buffer).toArray(DeviceBuffer[]::new)));
  }

  List<Tensor> pull(List<Tensor> inputs, Tensor cotangent) {
    DeviceProgram derived =
        mappedBase == null
            ? derive(this, false, inputs)
            : derive(mappedBase, false, sampleTemplates(inputs)).batched(mappedBatch);
    List<Tensor> all = new ArrayList<>(inputs);
    all.add(cotangent);
    Tensor packed =
        new Tensor(
            inputs.getFirst().backend(),
            inputs
                .getFirst()
                .backend()
                .program(derived, all.stream().map(Tensor::buffer).toArray(DeviceBuffer[]::new)));
    List<Tensor> gradients = new ArrayList<>();
    int offset = 0;
    for (Tensor input : inputs) {
      int[] ix = new int[Math.toIntExact(input.shape().size())];
      int sample = mappedBase == null ? ix.length : ix.length / mappedBatch;
      int total =
          mappedBase == null ? 0 : Math.toIntExact(derived.outputShape().size() / mappedBatch);
      for (int i = 0; i < ix.length; i++)
        ix[i] = mappedBase == null ? offset + i : (i / sample) * total + offset + i % sample;
      gradients.add(packed.reindex(input.shape(), ix));
      offset += sample;
    }
    packed.close();
    return gradients;
  }

  void addCapture(DeviceBuffer b) {
    for (var old : captures) if (old == b) return;
    captures.add(b);
  }

  int captureIndex(DeviceBuffer b) {
    for (int i = 0; i < captures.size(); i++) if (captures.get(i) == b) return i;
    throw new IllegalArgumentException("capture");
  }

  public Shape outputShape() {
    return resultShape;
  }

  private int explicitInputs() {
    return kind == Kind.GRAPH
        ? first.inputs.size() - first.captures.size()
        : kind == Kind.WHILE ? 1 : 2;
  }

  /** CPU reference interpreter. Accelerator backends must supply device lowering. */
  public DeviceBuffer evaluate(ComputeBackend backend, DeviceBuffer[] inputs) {
    if (mappedBase != null) return evaluateMapped(backend, inputs);
    if (original != null) return evaluateDerivative(backend, inputs);
    if (backend.deviceType() != DeviceType.CPU)
      throw new UnsupportedOperationException("device control-flow lowering required");
    DeviceBuffer[] constants = Arrays.copyOfRange(inputs, explicitInputs(), inputs.length);
    if (kind == Kind.GRAPH)
      return first
          .run(backend, Arrays.copyOf(inputs, explicitInputs()), this, constants)
          .getFirst();
    if (kind == Kind.COND) {
      Graph selected = backend.download(inputs[0])[0] != 0 ? first : second;
      return selected.run(backend, new DeviceBuffer[] {inputs[1]}, this, constants).getFirst();
    }
    DeviceBuffer carry = inputs[0];
    List<DeviceBuffer> allocated = new ArrayList<>();
    try {
      if (kind == Kind.WHILE) {
        while (true) {
          DeviceBuffer predicate =
              first.run(backend, new DeviceBuffer[] {carry}, this, constants).getFirst();
          boolean run = backend.download(predicate)[0] != 0;
          backend.release(predicate);
          if (!run) break;
          DeviceBuffer next =
              second.run(backend, new DeviceBuffer[] {carry}, this, constants).getFirst();
          if (next != carry) allocated.add(next);
          carry = next;
        }
        return carry;
      }
      List<DeviceBuffer> ys = new ArrayList<>();
      for (int i = 0; i < length; i++) {
        DeviceBuffer sample = backend.sliceAxis0(inputs[1], i);
        allocated.add(sample);
        List<DeviceBuffer> step =
            first.run(backend, new DeviceBuffer[] {carry, sample}, this, constants);
        carry = step.get(0);
        allocated.add(carry);
        ys.add(step.get(1));
        allocated.add(step.get(1));
      }
      float[] result = new float[Math.toIntExact(resultShape.size())];
      float[] c = backend.download(carry);
      System.arraycopy(c, 0, result, 0, c.length);
      int offset = c.length;
      for (var y : ys) {
        float[] data = backend.download(y);
        System.arraycopy(data, 0, result, offset, data.length);
        offset += data.length;
      }
      return backend.upload(result, resultShape, inputs[0].dtype(), inputs[0].device());
    } finally {
      Set<DeviceBuffer> freed = Collections.newSetFromMap(new IdentityHashMap<>());
      for (var b : allocated)
        if ((kind != Kind.WHILE || b != carry) && b != inputs[0] && freed.add(b))
          backend.release(b);
    }
  }

  /** One invocation executes a region; loops and predicates remain entirely on the device. */
  public String glsl(DeviceBuffer[] inputs) {
    if (mappedBase != null) return mappedGlsl(inputs);
    if (original != null) return derivativeGlsl(inputs);
    StringBuilder source =
        new StringBuilder(
            "#version 450\n"
                + "layout(local_size_x=1) in;\n"
                + "layout(std430,binding=0) writeonly buffer O { float outData[]; };\n");
    for (int i = 0; i < inputs.length; i++)
      source
          .append("layout(std430,binding=")
          .append(i + 1)
          .append(") readonly buffer I")
          .append(i)
          .append("{float in")
          .append(i)
          .append("[];};\n");
    source.append(first.glsl("first"));
    if (second != null) source.append(second.glsl("second"));
    source.append("void main(){");
    for (int i = 0; i < inputs.length; i++) {
      declare(source, "p" + i, inputs[i].shape());
      copy(source, "p" + i, "in" + i, inputs[i].shape());
    }
    if (kind == Kind.GRAPH) {
      declare(source, "r", resultShape);
      List<String> args = new ArrayList<>();
      for (int i = 0; i < explicitInputs(); i++) args.add("p" + i);
      invoke(source, "first", first, args, List.of("r"));
      copy(source, "outData", "r", resultShape);
    } else if (kind == Kind.COND) {
      declare(source, "r", stateShape);
      source.append("if(p0[0]!=0.){");
      invoke(source, "first", first, List.of("p1"), List.of("r"));
      source.append("}else{");
      invoke(source, "second", second, List.of("p1"), List.of("r"));
      source.append('}');
      copy(source, "outData", "r", stateShape);
    } else if (kind == Kind.WHILE) {
      declare(source, "c", stateShape);
      copy(source, "c", "p0", stateShape);
      declare(source, "r", stateShape);
      declare(source, "pred", Shape.of(1));
      source.append("while(true){");
      invoke(source, "first", first, List.of("c"), List.of("pred"));
      source.append("if(pred[0]==0.)break;");
      invoke(source, "second", second, List.of("c"), List.of("r"));
      copy(source, "c", "r", stateShape);
      source.append('}');
      copy(source, "outData", "c", stateShape);
    } else {
      Shape y = first.outputs.get(1).shape();
      declare(source, "c", stateShape);
      copy(source, "c", "p0", stateShape);
      declare(source, "r", stateShape);
      declare(source, "x", sampleShape);
      declare(source, "y", y);
      source.append("for(int step=0;step<").append(length).append(";step++){");
      source
          .append("for(int j=0;j<")
          .append(sampleShape.size())
          .append(";j++)x[j]=p1[step*")
          .append(sampleShape.size())
          .append("+j];");
      invoke(source, "first", first, List.of("c", "x"), List.of("r", "y"));
      copy(source, "c", "r", stateShape);
      source
          .append("for(int j=0;j<")
          .append(y.size())
          .append(";j++)outData[")
          .append(stateShape.size())
          .append("+step*")
          .append(y.size())
          .append("+j]=y[j];}");
      copy(source, "outData", "c", stateShape);
    }
    return source.append('}').toString();
  }

  private DeviceBuffer evaluateMapped(ComputeBackend backend, DeviceBuffer[] inputs) {
    if (backend.deviceType() != DeviceType.CPU)
      throw new UnsupportedOperationException("device map lowering required");
    List<DeviceBuffer> outputs = new ArrayList<>(), owned = new ArrayList<>();
    try {
      for (int i = 0; i < mappedBatch; i++) {
        DeviceBuffer[] samples = new DeviceBuffer[inputs.length];
        for (int j = 0; j < inputs.length; j++) {
          samples[j] = backend.sliceAxis0(inputs[j], i);
          owned.add(samples[j]);
        }
        DeviceBuffer out = mappedBase.evaluate(backend, samples);
        outputs.add(out);
        owned.add(out);
      }
      return backend.stackAxis0(outputs.toArray(DeviceBuffer[]::new));
    } finally {
      for (DeviceBuffer b : owned) backend.release(b);
    }
  }

  private String asFunction(String name, DeviceBuffer[] inputs) {
    String source = glsl(inputs);
    java.util.regex.Matcher functions =
        java.util.regex.Pattern.compile("(?:void|uvec2|uint)\\s+(\\w+)\\(").matcher(source);
    List<String> names = new ArrayList<>();
    int first = -1;
    while (functions.find()) {
      if (first < 0) first = functions.start();
      String function = functions.group(1);
      if (!function.equals("main")) names.add(function);
    }
    source = source.substring(first);
    for (String function : names)
      source = source.replaceAll("\\b" + function + "(?=\\()", name + "_" + function);
    int main = source.indexOf("void main(){");
    String prefix = source.substring(0, main),
        body = source.substring(main + "void main(){".length(), source.length() - 1);
    List<String> params = new ArrayList<>();
    for (int i = 0; i < inputs.length; i++)
      params.add("in float in" + i + "[" + Math.max(1, inputs[i].shape().size()) + "]");
    params.add("out float outData[" + Math.max(1, resultShape.size()) + "]");
    return prefix + "void " + name + "(" + String.join(",", params) + "){" + body + "}\n";
  }

  private String mappedGlsl(DeviceBuffer[] inputs) {
    DeviceBuffer[] samples = new DeviceBuffer[inputs.length];
    for (int i = 0; i < inputs.length; i++) {
      DeviceBuffer in = inputs[i];
      samples[i] =
          new DeviceBuffer() {
            public Shape shape() {
              return Shape.of(Arrays.copyOfRange(in.shape().dims(), 1, in.shape().rank()));
            }

            public DType dtype() {
              return in.dtype();
            }

            public Device device() {
              return in.device();
            }
          };
    }
    StringBuilder source =
        new StringBuilder(
            "#version 450\n"
                + "layout(local_size_x=1) in;\n"
                + "layout(std430,binding=0) writeonly buffer O {float outData[];};\n");
    for (int i = 0; i < inputs.length; i++)
      source
          .append("layout(std430,binding=")
          .append(i + 1)
          .append(") readonly buffer I")
          .append(i)
          .append("{float in")
          .append(i)
          .append("[];};\n");
    source.append(mappedBase.asFunction("region", samples)).append("void main(){");
    for (int i = 0; i < samples.length; i++) declare(source, "s" + i, samples[i].shape());
    declare(source, "r", mappedBase.resultShape);
    source.append("for(int b=0;b<").append(mappedBatch).append(";b++){");
    List<String> args = new ArrayList<>();
    for (int i = 0; i < samples.length; i++) {
      source
          .append("for(int j=0;j<")
          .append(samples[i].shape().size())
          .append(";j++)s")
          .append(i)
          .append("[j]=in")
          .append(i)
          .append("[b*")
          .append(samples[i].shape().size())
          .append("+j];");
      args.add("s" + i);
    }
    args.add("r");
    source
        .append("region(")
        .append(String.join(",", args))
        .append(");for(int j=0;j<")
        .append(mappedBase.resultShape.size())
        .append(";j++)outData[b*")
        .append(mappedBase.resultShape.size())
        .append("+j]=r[j];}}");
    return source.toString();
  }

  private List<String> regionInputs(Graph graph, List<String> explicit) {
    List<String> args = new ArrayList<>(explicit);
    for (DeviceBuffer c : graph.captures) args.add("p" + (explicitInputs() + captureIndex(c)));
    return args;
  }

  private List<Integer> regionIndices(Graph graph, int... explicit) {
    List<Integer> indices = new ArrayList<>();
    for (int i : explicit) indices.add(i);
    for (DeviceBuffer c : graph.captures) indices.add(explicitInputs() + captureIndex(c));
    return indices;
  }

  private List<String> callGraph(
      StringBuilder s, String function, Graph graph, List<String> inputs, String prefix) {
    List<String> outs = new ArrayList<>();
    for (int i = 0; i < graph.outputs.size(); i++) {
      String name = prefix + i;
      declare(s, name, graph.outputs.get(i).shape());
      outs.add(name);
    }
    List<String> all = new ArrayList<>(inputs);
    all.addAll(outs);
    s.append(function).append('(').append(String.join(",", all)).append(");");
    return outs;
  }

  private List<String> dualArgs(Graph graph, List<String> values, List<String> tangents) {
    List<String> args = regionInputs(graph, values);
    args.addAll(tangents);
    for (DeviceBuffer c : graph.captures)
      args.add("p" + (primalShapes.size() + explicitInputs() + captureIndex(c)));
    return args;
  }

  private List<String> pullArgs(Graph graph, List<String> values, List<String> seeds) {
    List<String> args = regionInputs(graph, values);
    args.addAll(seeds);
    return args;
  }

  private void accumulate(StringBuilder s, String target, String source, Shape shape) {
    s.append("for(int j=0;j<")
        .append(shape.size())
        .append(";j++)")
        .append(target)
        .append("[j]+=")
        .append(source)
        .append("[j];");
  }

  private String derivativeGlsl(DeviceBuffer[] inputs) {
    int n = primalShapes.size();
    StringBuilder s =
        new StringBuilder(
            "#version 450\n"
                + "layout(local_size_x=1) in;\n"
                + "layout(std430,binding=0) writeonly buffer O { float outData[]; };\n");
    for (int i = 0; i < inputs.length; i++)
      s.append("layout(std430,binding=")
          .append(i + 1)
          .append(") readonly buffer I")
          .append(i)
          .append("{float in")
          .append(i)
          .append("[];};\n");
    s.append(first.glsl("first"));
    if (second != null) s.append(second.glsl("second"));
    if (dFirst != null) s.append(dFirst.glsl("dfirst"));
    if (dSecond != null) s.append(dSecond.glsl("dsecond"));
    s.append("void main(){");
    for (int i = 0; i < inputs.length; i++) {
      declare(s, "p" + i, inputs[i].shape());
      copy(s, "p" + i, "in" + i, inputs[i].shape());
    }
    if (!forwardDerivative)
      for (int i = 0; i < n; i++) {
        declare(s, "g" + i, primalShapes.get(i));
        s.append("for(int j=0;j<")
            .append(primalShapes.get(i).size())
            .append(";j++)g")
            .append(i)
            .append("[j]=0.;");
      }
    if (kind == Kind.GRAPH) {
      List<String> args = new ArrayList<>();
      for (int i = 0; i < explicitInputs(); i++) args.add("p" + i);
      if (forwardDerivative) {
        List<String> ts = new ArrayList<>();
        for (int i = 0; i < explicitInputs(); i++) ts.add("p" + (n + i));
        var outputs = callGraph(s, "dfirst", dFirst, dualArgs(first, args, ts), "result");
        copy(s, "outData", outputs.get(1), resultShape);
      } else {
        var outputs =
            callGraph(s, "dfirst", dFirst, pullArgs(first, args, List.of("p" + n)), "result");
        List<Integer> indices = new ArrayList<>();
        for (int i = 0; i < explicitInputs(); i++) indices.add(i);
        for (var c : first.captures) indices.add(explicitInputs() + captureIndex(c));
        for (int i = 0; i < indices.size(); i++)
          accumulate(s, "g" + indices.get(i), outputs.get(i), primalShapes.get(indices.get(i)));
      }
    } else if (kind == Kind.COND) {
      s.append("if(p0[0]!=0.){");
      emitCondDerivative(s, first, dFirst, "dfirst", n);
      s.append("}else{");
      emitCondDerivative(s, second, dSecond, "dsecond", n);
      s.append('}');
    } else if (kind == Kind.WHILE) {
      declare(s, "c", stateShape);
      declare(s, "dc", stateShape);
      copy(s, "c", "p0", stateShape);
      copy(s, "dc", "p" + n, stateShape);
      s.append("while(true){");
      var pred = callGraph(s, "first", first, regionInputs(first, List.of("c")), "pred");
      s.append("if(").append(pred.getFirst()).append("[0]==0.)break;");
      var outputs =
          callGraph(s, "dsecond", dSecond, dualArgs(second, List.of("c"), List.of("dc")), "step");
      copy(s, "c", outputs.get(0), stateShape);
      copy(s, "dc", outputs.get(1), stateShape);
      s.append('}');
      copy(s, "outData", "dc", stateShape);
    } else if (forwardDerivative) {
      Shape y = first.outputs.get(1).shape();
      declare(s, "c", stateShape);
      declare(s, "dc", stateShape);
      declare(s, "x", sampleShape);
      declare(s, "dx", sampleShape);
      copy(s, "c", "p0", stateShape);
      copy(s, "dc", "p" + n, stateShape);
      s.append("for(int step=0;step<").append(length).append(";step++){");
      sample(s, "x", "p1", sampleShape);
      sample(s, "dx", "p" + (n + 1), sampleShape);
      var output =
          callGraph(
              s, "dfirst", dFirst, dualArgs(first, List.of("c", "x"), List.of("dc", "dx")), "step");
      copy(s, "c", output.get(0), stateShape);
      copy(s, "dc", output.get(2), stateShape);
      s.append("for(int j=0;j<")
          .append(y.size())
          .append(";j++)outData[")
          .append(stateShape.size())
          .append("+step*")
          .append(y.size())
          .append("+j]=")
          .append(output.get(3))
          .append("[j];}");
      copy(s, "outData", "dc", stateShape);
    } else {
      Shape y = first.outputs.get(1).shape();
      declare(s, "states", Shape.of(Math.toIntExact(length * stateShape.size())));
      declare(s, "c", stateShape);
      declare(s, "x", sampleShape);
      declare(s, "gy", y);
      declare(s, "gc", stateShape);
      copy(s, "c", "p0", stateShape);
      s.append("for(int j=0;j<")
          .append(stateShape.size())
          .append(";j++)gc[j]=p")
          .append(n)
          .append("[j];");
      s.append("for(int step=0;step<")
          .append(length)
          .append(";step++){for(int j=0;j<")
          .append(stateShape.size())
          .append(";j++)states[step*")
          .append(stateShape.size())
          .append("+j]=c[j];");
      sample(s, "x", "p1", sampleShape);
      var step = callGraph(s, "first", first, regionInputs(first, List.of("c", "x")), "forward");
      copy(s, "c", step.getFirst(), stateShape);
      s.append('}');
      s.append("for(int step=")
          .append(length - 1)
          .append(";step>=0;step--){for(int j=0;j<")
          .append(stateShape.size())
          .append(";j++)c[j]=states[step*")
          .append(stateShape.size())
          .append("+j];");
      sample(s, "x", "p1", sampleShape);
      s.append("for(int j=0;j<")
          .append(y.size())
          .append(";j++)gy[j]=p")
          .append(n)
          .append('[')
          .append(stateShape.size())
          .append("+step*")
          .append(y.size())
          .append("+j];");
      var grads =
          callGraph(
              s,
              "dfirst",
              dFirst,
              pullArgs(first, List.of("c", "x"), List.of("gc", "gy")),
              "backward");
      copy(s, "gc", grads.getFirst(), stateShape);
      s.append("for(int j=0;j<")
          .append(sampleShape.size())
          .append(";j++)g1[step*")
          .append(sampleShape.size())
          .append("+j]=")
          .append(grads.get(1))
          .append("[j];");
      for (int i = 0; i < first.captures.size(); i++) {
        int ix = explicitInputs() + captureIndex(first.captures.get(i));
        accumulate(s, "g" + ix, grads.get(2 + i), primalShapes.get(ix));
      }
      s.append('}');
      copy(s, "g0", "gc", stateShape);
    }
    if (!forwardDerivative) {
      long offset = 0;
      for (int i = 0; i < n; i++) {
        s.append("for(int j=0;j<")
            .append(primalShapes.get(i).size())
            .append(";j++)outData[")
            .append(offset)
            .append("+j]=g")
            .append(i)
            .append("[j];");
        offset += primalShapes.get(i).size();
      }
    }
    return s.append('}').toString();
  }

  private void emitCondDerivative(
      StringBuilder s, Graph graph, Graph derivative, String name, int n) {
    if (forwardDerivative) {
      var out =
          callGraph(
              s,
              name,
              derivative,
              dualArgs(graph, List.of("p1"), List.of("p" + (n + 1))),
              "branch");
      copy(s, "outData", out.get(1), resultShape);
    } else {
      var out =
          callGraph(
              s, name, derivative, pullArgs(graph, List.of("p1"), List.of("p" + n)), "branch");
      var indices = regionIndices(graph, 1);
      for (int i = 0; i < indices.size(); i++)
        accumulate(s, "g" + indices.get(i), out.get(i), primalShapes.get(indices.get(i)));
    }
  }

  private static void sample(StringBuilder s, String out, String input, Shape shape) {
    s.append("for(int j=0;j<")
        .append(shape.size())
        .append(";j++)")
        .append(out)
        .append("[j]=")
        .append(input)
        .append("[step*")
        .append(shape.size())
        .append("+j];");
  }

  private DeviceBuffer evaluateDerivative(ComputeBackend backend, DeviceBuffer[] buffers) {
    if (backend.deviceType() != DeviceType.CPU)
      throw new UnsupportedOperationException("device derivative lowering required");
    // Reference execution deliberately uses ordinary CPU tensors; Vulkan uses the region shader
    // above.
    int n = primalShapes.size();
    List<Tensor> inputs = new ArrayList<>();
    for (int i = 0; i < n; i++) inputs.add(new Tensor(backend, buffers[i]));
    List<Tensor> all = new ArrayList<>(), gradients = new ArrayList<>();
    for (Tensor input : inputs) {
      Tensor zero = input.fullLike(0);
      gradients.add(zero);
      all.add(zero);
    }
    Function<Graph, List<Tensor>> constants =
        graph -> {
          List<Tensor> values = new ArrayList<>();
          for (var c : graph.captures) values.add(inputs.get(explicitInputs() + captureIndex(c)));
          return values;
        };
    BiFunction<Graph, List<Tensor>, List<Tensor>> run =
        (graph, args) -> {
          List<Tensor> values = new ArrayList<>(args);
          values.addAll(constants.apply(graph));
          var result = graph.apply(values);
          all.addAll(result);
          return result;
        };
    try {
      if (kind == Kind.GRAPH) {
        List<Tensor> args = new ArrayList<>(inputs.subList(0, explicitInputs()));
        args.addAll(constants.apply(first));
        if (forwardDerivative) {
          List<Tensor> ts = new ArrayList<>();
          for (int i = 0; i < explicitInputs(); i++) ts.add(new Tensor(backend, buffers[n + i]));
          for (var c : first.captures)
            ts.add(new Tensor(backend, buffers[n + explicitInputs() + captureIndex(c)]));
          var result = Transforms.jvpTree(first::apply, args, ts);
          all.addAll(result.value());
          all.addAll(result.tangent());
          return result.tangent().getFirst().mul(1).buffer();
        }
        var result = Transforms.vjp(first::apply, args, List.of(new Tensor(backend, buffers[n])));
        all.addAll(result.value());
        all.addAll(result.cotangents());
        for (int i = 0; i < explicitInputs(); i++) gradients.set(i, result.cotangents().get(i));
        for (int i = 0; i < first.captures.size(); i++)
          gradients.set(
              explicitInputs() + captureIndex(first.captures.get(i)),
              result.cotangents().get(explicitInputs() + i));
      } else if (kind == Kind.COND) {
        Graph graph = inputs.get(0).item() != 0 ? first : second;
        List<Tensor> args = new ArrayList<>(List.of(inputs.get(1)));
        args.addAll(constants.apply(graph));
        if (forwardDerivative) {
          List<Tensor> ts = new ArrayList<>(List.of(new Tensor(backend, buffers[n + 1])));
          for (var c : graph.captures)
            ts.add(new Tensor(backend, buffers[n + explicitInputs() + captureIndex(c)]));
          var result = Transforms.jvpTree(graph::apply, args, ts);
          all.addAll(result.value());
          all.addAll(result.tangent());
          return result.tangent().getFirst().mul(1).buffer();
        }
        var result = Transforms.vjp(graph::apply, args, List.of(new Tensor(backend, buffers[n])));
        all.addAll(result.value());
        all.addAll(result.cotangents());
        var indices = regionIndices(graph, 1);
        for (int i = 0; i < indices.size(); i++)
          gradients.set(indices.get(i), result.cotangents().get(i));
      } else if (forwardDerivative) {
        Tensor carry = inputs.get(0), dc = new Tensor(backend, buffers[n]);
        List<Tensor> ys = new ArrayList<>();
        for (int step = 0; kind == Kind.WHILE || step < length; step++) {
          if (kind == Kind.WHILE && run.apply(first, List.of(carry)).getFirst().item() == 0) break;
          Graph graph = kind == Kind.WHILE ? second : first;
          List<Tensor> args = new ArrayList<>(List.of(carry)), ts = new ArrayList<>(List.of(dc));
          if (kind == Kind.SCAN) {
            Tensor x = inputs.get(1).sliceAxis0(step),
                dx = new Tensor(backend, buffers[n + 1]).sliceAxis0(step);
            all.add(x);
            all.add(dx);
            args.add(x);
            ts.add(dx);
          }
          args.addAll(constants.apply(graph));
          for (var c : graph.captures)
            ts.add(new Tensor(backend, buffers[n + explicitInputs() + captureIndex(c)]));
          var result = Transforms.jvpTree(graph::apply, args, ts);
          all.addAll(result.value());
          all.addAll(result.tangent());
          carry = result.value().getFirst();
          dc = result.tangent().getFirst();
          if (kind == Kind.SCAN) ys.add(result.tangent().get(1));
        }
        if (kind == Kind.WHILE) return dc.mul(1).buffer();
        List<Tensor> packed = new ArrayList<>(List.of(dc));
        packed.addAll(ys);
        return packCpu(backend, packed, resultShape, buffers[0].device());
      } else {
        List<Tensor> states = new ArrayList<>();
        Tensor carry = inputs.get(0);
        for (int step = 0; step < length; step++) {
          states.add(carry);
          Tensor x = inputs.get(1).sliceAxis0(step);
          all.add(x);
          carry = run.apply(first, List.of(carry, x)).getFirst();
        }
        Tensor seed = new Tensor(backend, buffers[n]);
        int[] ix = new int[Math.toIntExact(stateShape.size())];
        for (int i = 0; i < ix.length; i++) ix[i] = i;
        Tensor gc = seed.reindex(stateShape, ix);
        all.add(gc);
        List<Tensor> dxs = new ArrayList<>(Collections.nCopies(length, null));
        for (int step = length - 1; step >= 0; step--) {
          Tensor x = inputs.get(1).sliceAxis0(step);
          all.add(x);
          Shape y = first.outputs.get(1).shape();
          int[] yi = new int[Math.toIntExact(y.size())];
          for (int j = 0; j < yi.length; j++)
            yi[j] = Math.toIntExact(stateShape.size() + step * y.size() + j);
          Tensor gy = seed.reindex(y, yi);
          all.add(gy);
          List<Tensor> args = new ArrayList<>(List.of(states.get(step), x));
          args.addAll(constants.apply(first));
          var result = Transforms.vjp(first::apply, args, List.of(gc, gy));
          all.addAll(result.value());
          all.addAll(result.cotangents());
          gc = result.cotangents().getFirst();
          dxs.set(step, result.cotangents().get(1));
          for (int i = 0; i < first.captures.size(); i++) {
            int j = explicitInputs() + captureIndex(first.captures.get(i));
            Tensor sum = gradients.get(j).add(result.cotangents().get(2 + i));
            all.add(sum);
            gradients.set(j, sum);
          }
        }
        gradients.set(0, gc);
        Tensor dx = dxs.isEmpty() ? inputs.get(1).fullLike(0) : Tensor.stackAxis0(dxs);
        all.add(dx);
        gradients.set(1, dx);
      }
      return packCpu(backend, gradients, resultShape, buffers[0].device());
    } finally {
      Set<Tensor> freed = Collections.newSetFromMap(new IdentityHashMap<>());
      for (Tensor t : all) if (!inputs.contains(t) && freed.add(t)) t.close();
    }
  }

  private static DeviceBuffer packCpu(
      ComputeBackend backend, List<Tensor> tensors, Shape shape, Device device) {
    float[] values = new float[Math.toIntExact(shape.size())];
    int offset = 0;
    for (Tensor t : tensors) {
      float[] data = t.toFloatArray();
      System.arraycopy(data, 0, values, offset, data.length);
      offset += data.length;
    }
    return backend.upload(values, shape, DType.F32, device);
  }

  void invoke(
      StringBuilder src, String name, Graph graph, List<String> parameters, List<String> outputs) {
    List<String> args = new ArrayList<>(parameters);
    for (var c : graph.captures) args.add("p" + (explicitInputs() + captureIndex(c)));
    args.addAll(outputs);
    src.append(name).append('(').append(String.join(",", args)).append(");");
  }

  static void declare(StringBuilder s, String name, Shape shape) {
    s.append("float ").append(name).append('[').append(Math.max(1, shape.size())).append("];");
  }

  static void copy(StringBuilder s, String out, String in, Shape shape) {
    s.append("for(int j=0;j<")
        .append(shape.size())
        .append(";j++)")
        .append(out)
        .append("[j]=")
        .append(in)
        .append("[j];");
  }

  record Symbol(Graph owner, int id, Shape shape, DType dtype, Device device)
      implements TracingBuffer {
    public ComputeBackend tracingBackend() {
      return owner.proxy;
    }
  }

  record Instruction(Method method, Object[] args, Symbol output) {}

  static final class Graph implements InvocationHandler {
    final ComputeBackend parent, proxy;
    final List<Symbol> inputs = new ArrayList<>(), outputs = new ArrayList<>();
    final List<DeviceBuffer> captures = new ArrayList<>();
    final List<Instruction> instructions = new ArrayList<>();
    int sequence;

    Graph(ComputeBackend parent) {
      this.parent = parent;
      proxy =
          (ComputeBackend)
              Proxy.newProxyInstance(
                  ComputeBackend.class.getClassLoader(),
                  new Class<?>[] {TracingBackend.class},
                  this);
    }

    static Graph capture(
        ComputeBackend backend, List<Tensor> templates, Function<List<Tensor>, List<Tensor>> body) {
      Graph graph = new Graph(backend);
      List<Tensor> args = new ArrayList<>();
      for (Tensor t : templates) {
        Symbol s = graph.symbol(t.shape(), t.dtype(), t.device());
        graph.inputs.add(s);
        args.add(new Tensor(graph.proxy, s));
      }
      for (Tensor t : body.apply(args)) graph.outputs.add(graph.bind(t.buffer()));
      return graph;
    }

    Symbol symbol(Shape shape, DType dtype, Device device) {
      return new Symbol(this, sequence++, shape, dtype, device);
    }

    Symbol bind(DeviceBuffer b) {
      if (b instanceof Symbol s && s.owner == this) return s;
      for (int i = 0; i < captures.size(); i++)
        if (captures.get(i) == b) return inputs.get(inputs.size() - captures.size() + i);
      Symbol s = symbol(b.shape(), b.dtype(), b.device());
      inputs.add(s);
      captures.add(b);
      return s;
    }

    public Object invoke(Object p, Method m, Object[] raw) throws Throwable {
      String name = m.getName();
      Object[] args = raw == null ? new Object[0] : raw.clone();
      if (name.equals("parentBackend")) return parent;
      if (name.equals("release")) return null;
      if (name.equals("download") || name.equals("sync"))
        throw new IllegalStateException("Host access inside a device region");
      if (name.equals("custom") || name.equals("fused"))
        return InvocationHandler.invokeDefault(p, m, args);
      if (m.getReturnType() != DeviceBuffer.class) return m.invoke(parent, args);
      for (int i = 0; i < args.length; i++)
        if (args[i] instanceof DeviceBuffer b) args[i] = bind(b);
        else if (args[i] instanceof DeviceBuffer[] bs) {
          Symbol[] ss = new Symbol[bs.length];
          for (int j = 0; j < bs.length; j++) ss[j] = bind(bs[j]);
          args[i] = ss;
        }
      Shape shape = infer(name, args);
      DeviceBuffer a = null;
      for (Object arg : args)
        if (arg instanceof DeviceBuffer b) {
          a = b;
          break;
        } else if (arg instanceof DeviceBuffer[] bs && bs.length > 0) {
          a = bs[0];
          break;
        }
      DType dtype =
          a == null ? (name.startsWith("random") ? DType.F32 : (DType) args[2]) : a.dtype();
      Device device = a == null ? (Device) args[3] : a.device();
      if (dtype != DType.F32)
        throw new UnsupportedOperationException("Device regions support F32 only, got " + dtype);
      Symbol out = symbol(shape, dtype, device);
      instructions.add(new Instruction(m, args, out));
      return out;
    }

    Shape infer(String name, Object[] a) {
      DeviceBuffer x = a.length > 0 && a[0] instanceof DeviceBuffer b ? b : null;
      return switch (name) {
        case "program" -> ((DeviceProgram) a[0]).outputShape();
        case "upload", "full" -> (Shape) a[1];
        case "randomUniform", "randomNormal" -> (Shape) a[2];
        case "unary", "scalar", "binary" -> ((DeviceBuffer) a[1]).shape();
        case "reshape", "broadcastTo", "reindex", "scatterAdd" -> {
          Shape target = (Shape) a[1];
          if (name.equals("reshape") && x.shape().size() != target.size())
            throw new IllegalArgumentException("reshape size mismatch");
          if (name.equals("broadcastTo") && !Shape.broadcast(x.shape(), target).equals(target))
            throw new IllegalArgumentException("broadcast shape mismatch");
          if (name.equals("reindex") || name.equals("scatterAdd")) {
            int[] ix = (int[]) a[2];
            boolean scatter = name.equals("scatterAdd");
            if (ix.length != (scatter ? x.shape().size() : target.size()))
              throw new IllegalArgumentException("index map size");
            for (int index : ix)
              if (index < -1 || index >= (scatter ? target.size() : x.shape().size()))
                throw new IllegalArgumentException("index out of bounds");
          }
          yield target;
        }
        case "transpose" -> x.shape().transposed();
        case "matmul" -> {
          Shape left = x.shape(), right = ((DeviceBuffer) a[1]).shape();
          if (left.rank() != 2 || right.rank() != 2 || left.dim(1) != right.dim(0))
            throw new IllegalArgumentException("matmul shape mismatch");
          yield Shape.of(left.dim(0), right.dim(1));
        }
        case "batchedMatmul" -> {
          Shape left = x.shape(), right = ((DeviceBuffer) a[1]).shape();
          if (left.rank() != 3
              || right.rank() != 3
              || left.dim(2) != right.dim(1)
              || (left.dim(0) != right.dim(0) && left.dim(0) != 1 && right.dim(0) != 1))
            throw new IllegalArgumentException("batched matmul shape mismatch");
          yield Shape.of(Math.max(left.dim(0), right.dim(0)), left.dim(1), right.dim(2));
        }
        case "sliceAxis0" -> {
          if (x.shape().rank() < 1 || (int) a[1] < 0 || (int) a[1] >= x.shape().dim(0))
            throw new IllegalArgumentException("slice index out of bounds");
          yield Shape.of(Arrays.copyOfRange(x.shape().dims(), 1, x.shape().rank()));
        }
        case "stackAxis0" -> {
          DeviceBuffer[] bs = (DeviceBuffer[]) a[0];
          int[] dims = new int[bs[0].shape().rank() + 1];
          dims[0] = bs.length;
          System.arraycopy(bs[0].shape().dims(), 0, dims, 1, dims.length - 1);
          yield Shape.of(dims);
        }
        case "reduceSum", "reduceMax" -> Shape.of(1);
        case "sumAxis0", "sumAxis", "maxAxis", "argMaxAxis" -> {
          int ax = name.equals("sumAxis0") ? 0 : (int) a[1];
          boolean keep = a.length > 2 && (boolean) a[2];
          int[] dims = x.shape().dims();
          if (keep) {
            dims[ax] = 1;
            yield Shape.of(dims);
          }
          int[] ds = new int[dims.length - 1];
          System.arraycopy(dims, 0, ds, 0, ax);
          System.arraycopy(dims, ax + 1, ds, ax, dims.length - ax - 1);
          yield Shape.of(ds);
        }
        case "maxAxisBackward", "reluBackward" -> ((DeviceBuffer) a[1]).shape();
        case "conv2d" -> ((ConvSpec) a[2]).outputShape(x.shape().dim(0));
        case "conv2dGradInput" -> ((ConvSpec) a[2]).inputShape(x.shape().dim(0));
        case "conv2dGradWeight" -> ((ConvSpec) a[2]).weightShape();
        default -> throw new UnsupportedOperationException("No device-region rule for " + name);
      };
    }

    List<DeviceBuffer> run(
        ComputeBackend backend,
        DeviceBuffer[] params,
        DeviceProgram program,
        DeviceBuffer[] constants) {
      Map<Symbol, DeviceBuffer> values = new IdentityHashMap<>();
      Set<DeviceBuffer> owned = Collections.newSetFromMap(new IdentityHashMap<>());
      for (int i = 0; i < params.length; i++) values.put(inputs.get(i), params[i]);
      for (int i = 0; i < captures.size(); i++)
        values.put(inputs.get(params.length + i), constants[program.captureIndex(captures.get(i))]);
      try {
        for (Instruction instruction : instructions) {
          Object[] args = instruction.args.clone();
          for (int i = 0; i < args.length; i++)
            if (args[i] instanceof Symbol symbol) args[i] = values.get(symbol);
            else if (args[i] instanceof Symbol[] ss) {
              DeviceBuffer[] actual = new DeviceBuffer[ss.length];
              for (int j = 0; j < ss.length; j++) actual[j] = values.get(ss[j]);
              args[i] = actual;
            }
          DeviceBuffer value;
          try {
            value = (DeviceBuffer) instruction.method.invoke(backend, args);
          } catch (ReflectiveOperationException e) {
            throw new IllegalStateException(e);
          }
          values.put(instruction.output, value);
          owned.add(value);
        }
        List<DeviceBuffer> result =
            outputs.stream().map(values::get).map(b -> backend.scalar(Op.MUL, b, 1)).toList();
        return result;
      } finally {
        for (var b : owned) backend.release(b);
      }
    }

    List<Tensor> apply(List<Tensor> args) {
      ComputeBackend backend = args.getFirst().backend();
      Map<Symbol, DeviceBuffer> values = new IdentityHashMap<>();
      for (int i = 0; i < inputs.size(); i++) values.put(inputs.get(i), args.get(i).buffer());
      for (Instruction instruction : instructions) {
        Object[] operands = instruction.args.clone();
        for (int i = 0; i < operands.length; i++)
          if (operands[i] instanceof Symbol symbol) operands[i] = values.get(symbol);
          else if (operands[i] instanceof Symbol[] ss) {
            DeviceBuffer[] actual = new DeviceBuffer[ss.length];
            for (int j = 0; j < ss.length; j++) actual[j] = values.get(ss[j]);
            operands[i] = actual;
          }
        try {
          values.put(
              instruction.output, (DeviceBuffer) instruction.method.invoke(backend, operands));
        } catch (InvocationTargetException e) {
          if (e.getCause() instanceof RuntimeException r) throw r;
          throw new IllegalStateException(e.getCause());
        } catch (ReflectiveOperationException e) {
          throw new IllegalStateException(e);
        }
      }
      return outputs.stream().map(symbol -> new Tensor(backend, values.get(symbol))).toList();
    }

    Graph derivative(boolean forward, ComputeBackend backend) {
      List<Tensor> templates = new ArrayList<>();
      for (Symbol input : inputs) templates.add(new Tensor(backend, input));
      if (forward) for (Symbol input : inputs) templates.add(new Tensor(backend, input));
      else for (Symbol output : outputs) templates.add(new Tensor(backend, output));
      return capture(
          backend,
          templates,
          args -> {
            List<Tensor> xs = args.subList(0, inputs.size());
            if (forward) {
              var result =
                  Transforms.jvpTree(this::apply, xs, args.subList(inputs.size(), args.size()));
              List<Tensor> all = new ArrayList<>(result.value());
              all.addAll(result.tangent());
              return all;
            }
            return Transforms.vjp(this::apply, xs, args.subList(inputs.size(), args.size()))
                .cotangents();
          });
    }

    String glsl(String name) {
      StringBuilder src = new StringBuilder();
      for (Instruction instruction : instructions)
        if (instruction.method.getName().equals("program"))
          src.append(
              ((DeviceProgram) instruction.args[0])
                  .asFunction(
                      name + "op" + instruction.output.id, (DeviceBuffer[]) instruction.args[1]));
      if (instructions.stream().anyMatch(op -> op.method.getName().startsWith("random")))
        src.append(randomHelpers(name));
      src.append("void ").append(name).append('(');
      List<String> params = new ArrayList<>();
      for (var input : inputs)
        params.add("in float v" + input.id + "[" + Math.max(1, input.shape.size()) + "]");
      for (int i = 0; i < outputs.size(); i++)
        params.add("out float r" + i + "[" + Math.max(1, outputs.get(i).shape.size()) + "]");
      src.append(String.join(",", params)).append("){");
      for (Instruction op : instructions) emit(src, op, name);
      for (int i = 0; i < outputs.size(); i++)
        copy(src, "r" + i, "v" + outputs.get(i).id, outputs.get(i).shape);
      return src.append("}\n").toString();
    }

    static String ref(Object a) {
      return "v" + ((Symbol) a).id;
    }

    static String unary(Op op, String x) {
      return switch (op) {
        case NEG -> "(-" + x + ")";
        case EXP -> "exp(" + x + ")";
        case LOG -> "log(" + x + ")";
        case SQRT -> "sqrt(" + x + ")";
        case RSQRT -> "inversesqrt(" + x + ")";
        case TANH -> "tanh(" + x + ")";
        case SIGMOID -> "(1./(1.+exp(-" + x + ")))";
        case RELU -> "max(" + x + ",0.)";
        case ABS -> "abs(" + x + ")";
        case SIGN -> "sign(" + x + ")";
        default -> throw new IllegalArgumentException("unary");
      };
    }

    static String binary(Op op, String a, String b) {
      return switch (op) {
        case ADD -> "(" + a + "+" + b + ")";
        case SUB -> "(" + a + "-" + b + ")";
        case MUL -> "(" + a + "*" + b + ")";
        case DIV -> "(" + a + "/" + b + ")";
        case MAX -> "max(" + a + "," + b + ")";
        case MIN -> "min(" + a + "," + b + ")";
        default -> throw new IllegalArgumentException("binary");
      };
    }

    static String literal(double x) {
      if (!Double.isFinite(x)) throw new IllegalArgumentException("nonfinite device literal");
      return Float.toString((float) x);
    }

    void emit(StringBuilder s, Instruction instruction, String prefix) {
      Object[] a = instruction.args;
      String name = instruction.method.getName(), out = ref(instruction.output);
      Shape shape = instruction.output.shape;
      declare(s, out, shape);
      if (name.equals("program")) {
        List<String> args = new ArrayList<>();
        for (var input : (DeviceBuffer[]) a[1]) args.add(ref(input));
        args.add(out);
        s.append(prefix)
            .append("op")
            .append(instruction.output.id)
            .append('(')
            .append(String.join(",", args))
            .append(");");
        return;
      }
      if (name.equals("stackAxis0")) {
        Symbol[] inputs = (Symbol[]) a[0];
        int offset = 0;
        for (var input : inputs) {
          s.append("for(int j=0;j<")
              .append(input.shape.size())
              .append(";j++)")
              .append(out)
              .append('[')
              .append(offset)
              .append("+j]=")
              .append(ref(input))
              .append("[j];");
          offset += input.shape.size();
        }
        return;
      }
      if (name.equals("upload")) {
        float[] data = (float[]) a[0];
        for (int i = 0; i < data.length; i++)
          s.append(out).append('[').append(i).append("]=").append(literal(data[i])).append(';');
        return;
      }
      String x = a.length > 0 && a[0] instanceof Symbol ? ref(a[0]) : null;
      s.append("for(int i=0;i<").append(shape.size()).append(";i++){");
      switch (name) {
        case "full" -> s.append(out).append("[i]=").append(literal((float) a[0])).append(';');
        case "unary" ->
            s.append(out).append("[i]=").append(unary((Op) a[0], ref(a[1]) + "[i]")).append(';');
        case "scalar" ->
            s.append(out)
                .append("[i]=")
                .append(binary((Op) a[0], ref(a[1]) + "[i]", literal((double) a[2])))
                .append(';');
        case "binary" ->
            s.append(out)
                .append("[i]=")
                .append(binary((Op) a[0], ref(a[1]) + "[i]", ref(a[2]) + "[i]"))
                .append(';');
        case "reshape" -> s.append(out).append("[i]=").append(x).append("[i];");
        case "sliceAxis0" ->
            s.append(out)
                .append("[i]=")
                .append(x)
                .append('[')
                .append((int) a[1] * shape.size())
                .append("+i];");
        case "transpose" -> {
          Shape original = ((Symbol) a[0]).shape;
          s.append(out)
              .append("[i]=")
              .append(x)
              .append("[(i%")
              .append(original.dim(0))
              .append(")*")
              .append(original.dim(1))
              .append("+i/")
              .append(original.dim(0))
              .append("];");
        }
        case "reindex", "scatterAdd" -> {
          int[] map = (int[]) a[2];
          s.append("const int ix[").append(Math.max(1, map.length)).append("]=int[](");
          if (map.length == 0) s.append("-1");
          for (int j = 0; j < map.length; j++) {
            if (j > 0) s.append(',');
            s.append(map[j]);
          }
          s.append(");");
          if (name.equals("reindex"))
            s.append(out).append("[i]=ix[i]<0?0.:").append(x).append("[ix[i]];");
          else
            s.append("float sum=0.;for(int j=0;j<")
                .append(map.length)
                .append(";j++)if(ix[j]==i)sum+=")
                .append(x)
                .append("[j];")
                .append(out)
                .append("[i]=sum;");
        }
        case "broadcastTo" -> {
          Shape original = ((Symbol) a[0]).shape;
          int[] stride = strides(shape), old = strides(original);
          StringBuilder ix = new StringBuilder("0");
          int shift = shape.rank() - original.rank();
          for (int j = 0; j < original.rank(); j++)
            if (original.dim(j) != 1)
              ix.append("+((i/")
                  .append(stride[j + shift])
                  .append(")%")
                  .append(shape.dim(j + shift))
                  .append(")*")
                  .append(old[j]);
          s.append(out).append("[i]=").append(x).append('[').append(ix).append("];");
        }
        case "matmul", "batchedMatmul" -> {
          Shape left = ((Symbol) a[0]).shape, right = ((Symbol) a[1]).shape;
          boolean batched = name.equals("batchedMatmul");
          int m = left.dim(batched ? 1 : 0),
              k = left.dim(batched ? 2 : 1),
              n = right.dim(batched ? 2 : 1);
          String ax = batched && left.dim(0) > 1 ? "(i/" + (m * n) + ")*" + (m * k) + "+" : "",
              bx = batched && right.dim(0) > 1 ? "(i/" + (m * n) + ")*" + (k * n) + "+" : "";
          s.append("float sum=0.;for(int j=0;j<")
              .append(k)
              .append(";j++)sum+=")
              .append(x)
              .append('[')
              .append(ax)
              .append("((i/")
              .append(n)
              .append(")%")
              .append(m)
              .append(")*")
              .append(k)
              .append("+j]*")
              .append(ref(a[1]))
              .append('[')
              .append(bx)
              .append("j*")
              .append(n)
              .append("+i%")
              .append(n)
              .append("];")
              .append(out)
              .append("[i]=sum;");
        }
        case "reduceSum", "reduceMax", "sumAxis0", "sumAxis", "maxAxis", "argMaxAxis" -> {
          Shape input = ((Symbol) a[0]).shape;
          boolean all = name.startsWith("reduce"),
              sum = name.equals("reduceSum") || name.equals("sumAxis") || name.equals("sumAxis0"),
              arg = name.equals("argMaxAxis");
          int ax = all ? 0 : name.equals("sumAxis0") ? 0 : (int) a[1],
              count = all ? Math.toIntExact(input.size()) : input.dim(ax),
              inner = 1;
          if (!all) for (int j = ax + 1; j < input.rank(); j++) inner *= input.dim(j);
          s.append("float val=")
              .append(sum ? "0." : "-3.402823466e38")
              .append(";int best=0;for(int j=0;j<")
              .append(count)
              .append(";j++){float z=")
              .append(x)
              .append('[')
              .append(
                  all
                      ? "j"
                      : "(i/" + inner + ")*" + (count * inner) + "+j*" + inner + "+i%" + inner)
              .append("];");
          if (sum) s.append("val+=z;");
          else s.append("if(z>val){val=z;best=j;}");
          s.append('}').append(out).append("[i]=").append(arg ? "float(best)" : "val").append(';');
        }
        case "maxAxisBackward" -> {
          Shape input = ((Symbol) a[1]).shape;
          int axis = (int) a[2], inner = 1;
          for (int j = axis + 1; j < input.rank(); j++) inner *= input.dim(j);
          int count = input.dim(axis);
          String in = ref(a[1]);
          s.append("int outer=i/")
              .append(count * inner)
              .append(",inner=i%")
              .append(inner)
              .append(",coordinate=(i/")
              .append(inner)
              .append(")%")
              .append(count)
              .append(";int best=0;float val=-3.402823466e38;for(int j=0;j<")
              .append(count)
              .append(";j++){float z=")
              .append(in)
              .append("[outer*")
              .append(count * inner)
              .append("+j*")
              .append(inner)
              .append("+inner];if(z>val){val=z;best=j;}}")
              .append(out)
              .append("[i]=coordinate==best?")
              .append(x)
              .append("[outer*")
              .append(inner)
              .append("+inner]:0.;");
        }
        case "conv2d", "conv2dGradInput", "conv2dGradWeight" -> {
          ConvSpec spec = (ConvSpec) a[2];
          String other = ref(a[1]);
          int ih = spec.inHeight(),
              iw = spec.inWidth(),
              ic = spec.inChannels(),
              oc = spec.outChannels(),
              k = spec.kernel(),
              oh = spec.outHeight(),
              ow = spec.outWidth(),
              stride = spec.stride(),
              pad = spec.pad();
          if (name.equals("conv2d")) {
            s.append("int batch=i/")
                .append(spec.outputSize())
                .append(",oc=(i/")
                .append(oh * ow)
                .append(")%")
                .append(oc)
                .append(",oy=(i/")
                .append(ow)
                .append(")%")
                .append(oh)
                .append(",ox=i%")
                .append(ow)
                .append(";float sum=0.;");
            s.append("for(int c=0;c<")
                .append(ic)
                .append(";c++)for(int ky=0;ky<")
                .append(k)
                .append(";ky++)for(int kx=0;kx<")
                .append(k)
                .append(";kx++){int iy=oy*")
                .append(stride)
                .append('-')
                .append(pad)
                .append("+ky,ix=ox*")
                .append(stride)
                .append('-')
                .append(pad)
                .append("+kx;if(iy>=0&&iy<")
                .append(ih)
                .append("&&ix>=0&&ix<")
                .append(iw)
                .append(")sum+=")
                .append(x)
                .append("[batch*")
                .append(spec.inputSize())
                .append("+(c*")
                .append(ih)
                .append("+iy)*")
                .append(iw)
                .append("+ix]*")
                .append(other)
                .append("[oc*")
                .append(spec.weightSize())
                .append("+(c*")
                .append(k)
                .append("+ky)*")
                .append(k)
                .append("+kx];}");
          } else if (name.equals("conv2dGradInput")) {
            s.append("int batch=i/")
                .append(spec.inputSize())
                .append(",c=(i/")
                .append(ih * iw)
                .append(")%")
                .append(ic)
                .append(",iy=(i/")
                .append(iw)
                .append(")%")
                .append(ih)
                .append(",ix=i%")
                .append(iw)
                .append(";float sum=0.;for(int o=0;o<")
                .append(oc)
                .append(";o++)for(int ky=0;ky<")
                .append(k)
                .append(";ky++)for(int kx=0;kx<")
                .append(k)
                .append(";kx++){int yy=iy+")
                .append(pad)
                .append("-ky,xx=ix+")
                .append(pad)
                .append("-kx;if(yy>=0&&xx>=0&&yy%")
                .append(stride)
                .append("==0&&xx%")
                .append(stride)
                .append("==0){int oy=yy/")
                .append(stride)
                .append(",ox=xx/")
                .append(stride)
                .append(";if(oy<")
                .append(oh)
                .append("&&ox<")
                .append(ow)
                .append(")sum+=")
                .append(x)
                .append("[batch*")
                .append(spec.outputSize())
                .append("+(o*")
                .append(oh)
                .append("+oy)*")
                .append(ow)
                .append("+ox]*")
                .append(other)
                .append("[o*")
                .append(spec.weightSize())
                .append("+(c*")
                .append(k)
                .append("+ky)*")
                .append(k)
                .append("+kx];}}");
          } else {
            int batch = ((Symbol) a[0]).shape.dim(0);
            s.append("int o=i/")
                .append(spec.weightSize())
                .append(",c=(i/")
                .append(k * k)
                .append(")%")
                .append(ic)
                .append(",ky=(i/")
                .append(k)
                .append(")%")
                .append(k)
                .append(",kx=i%")
                .append(k)
                .append(";float sum=0.;for(int b=0;b<")
                .append(batch)
                .append(";b++)for(int oy=0;oy<")
                .append(oh)
                .append(";oy++)for(int ox=0;ox<")
                .append(ow)
                .append(";ox++){int iy=oy*")
                .append(stride)
                .append('-')
                .append(pad)
                .append("+ky,ix=ox*")
                .append(stride)
                .append('-')
                .append(pad)
                .append("+kx;if(iy>=0&&iy<")
                .append(ih)
                .append("&&ix>=0&&ix<")
                .append(iw)
                .append(")sum+=")
                .append(x)
                .append("[b*")
                .append(spec.inputSize())
                .append("+(c*")
                .append(ih)
                .append("+iy)*")
                .append(iw)
                .append("+ix]*")
                .append(other)
                .append("[b*")
                .append(spec.outputSize())
                .append("+(o*")
                .append(oh)
                .append("+oy)*")
                .append(ow)
                .append("+ox];}");
          }
          s.append(out).append("[i]=sum;");
        }
        case "randomUniform", "randomNormal" -> {
          String rng = prefix + "rng", seed = words((long) a[0]), counter = words((long) a[1]);
          s.append("uvec2 counter=")
              .append(rng)
              .append("add(")
              .append(counter)
              .append(",uvec2(uint(i)")
              .append(name.equals("randomNormal") ? "<<1u,uint(i)>>31u)" : " ,0u)")
              .append(");");
          if (name.equals("randomUniform"))
            s.append(out)
                .append("[i]=float(")
                .append(rng)
                .append("bits(")
                .append(seed)
                .append(",counter))*(1./16777216.);");
          else
            s.append("float u1=float(")
                .append(rng)
                .append("bits(")
                .append(seed)
                .append(",counter)+1u)*(1./16777216.);float u2=float(")
                .append(rng)
                .append("bits(")
                .append(seed)
                .append(',')
                .append(rng)
                .append("add(counter,uvec2(1u,0u))))*(1./16777216.);")
                .append(out)
                .append("[i]=sqrt(-2.*log(u1))*cos(6.283185307179586*u2);");
        }
        case "reluBackward" ->
            s.append(out)
                .append("[i]=")
                .append(ref(a[1]))
                .append("[i]>0.?")
                .append(x)
                .append("[i]:0.;");
        default ->
            throw new UnsupportedOperationException("No GLSL device-region rule for " + name);
      }
      s.append('}');
    }

    static String randomHelpers(String prefix) {
      return ("""
      uvec2 PREFIXadd(uvec2 a,uvec2 b){uint lo=a.x+b.x;return uvec2(lo,a.y+b.y+uint(lo<a.x));}
      uvec2 PREFIXshr(uvec2 a,uint n){return uvec2((a.x>>n)|(a.y<<(32u-n)),a.y>>n);}
      uvec2 PREFIXmul(uvec2 a,uvec2 b){uint hi,lo;umulExtended(a.x,b.x,hi,lo);return uvec2(lo,hi+a.x*b.y+a.y*b.x);}
      uint PREFIXbits(uvec2 seed,uvec2 counter){uvec2 x=PREFIXadd(seed,PREFIXmul(uvec2(0x7f4a7c15u,0x9e3779b9u),counter));x=PREFIXmul(x^PREFIXshr(x,30u),uvec2(0x1ce4e5b9u,0xbf58476du));x=PREFIXmul(x^PREFIXshr(x,27u),uvec2(0x133111ebu,0x94d049bbu));x=x^PREFIXshr(x,31u);return x.y>>8u;}
      """)
          .replace("PREFIX", prefix + "rng");
    }

    static String words(long x) {
      return "uvec2("
          + Long.toUnsignedString(x & 0xffffffffL)
          + "u,"
          + Long.toUnsignedString(x >>> 32)
          + "u)";
    }

    static int[] strides(Shape shape) {
      int[] strides = new int[shape.rank()];
      int stride = 1;
      for (int i = strides.length - 1; i >= 0; i--) {
        strides[i] = stride;
        stride *= shape.dim(i);
      }
      return strides;
    }
  }
}
