package com.nablatensor.jax4j.tensor;

import com.nablatensor.jax4j.tensor.spi.*;
import com.nablatensor.jax4j.tensor.tree.TreeUtil;
import java.lang.reflect.*;
import java.util.*;
import java.util.function.*;

/** Composable tensor autodiff. Rules execute on the input backend, including enclosing tracers. */
public final class Transforms {
  private Transforms() {}

  public record ValueAndGrad(Tensor value, Tensor grad) {}

  public record Jvp(Tensor value, Tensor tangent) {}

  public record Vjp<R, P>(R value, P cotangents) {}

  public static Tensor grad(UnaryOperator<Tensor> body, Tensor input) {
    ValueAndGrad result = valueAndGrad(body, input);
    if (result.value().buffer() != input.buffer()
        && result.value().buffer() != result.grad().buffer()) result.value().close();
    return result.grad();
  }

  public static UnaryOperator<Tensor> grad(UnaryOperator<Tensor> body) {
    return x -> grad(body, x);
  }

  public static ValueAndGrad valueAndGrad(UnaryOperator<Tensor> body, Tensor input) {
    return reverse(body, input, null, true);
  }

  private static ValueAndGrad reverse(
      UnaryOperator<Tensor> body, Tensor input, Tensor cotangent, boolean scalar) {
    Scope scope = new Scope(input.backend());
    AD ad = new AD(scope, false);
    Node leaf = new Node(ad, input.buffer(), null, List.of(), null);
    try {
      Tensor output = body.apply(new Tensor(ad.proxy, leaf));
      Node end = ad.node(output.buffer());
      if (scalar && end.shape().size() != 1)
        throw new IllegalArgumentException("grad requires scalar output");
      Tensor seed =
          cotangent == null ? scope.t(end.primal).fullLike(1) : scope.t(cotangent.buffer());
      if (!seed.shape().equals(end.shape())
          || seed.dtype() != end.dtype()
          || !seed.device().equals(end.device()))
        throw new IllegalArgumentException("cotangent signature mismatch");
      Map<Node, Tensor> grads = ad.backward(List.of(end), List.of(seed));
      Tensor gradient = grads.getOrDefault(leaf, null);
      if (gradient == null) gradient = scope.t(input.buffer()).fullLike(0);
      Tensor value = scope.independent(end.primal);
      gradient = scope.independent(gradient.buffer());
      scope.keep(value.buffer(), gradient.buffer());
      return new ValueAndGrad(
          new Tensor(scope.real, value.buffer()), new Tensor(scope.real, gradient.buffer()));
    } finally {
      scope.close();
    }
  }

  public static Jvp jvp(UnaryOperator<Tensor> body, Tensor input, Tensor tangent) {
    if (!input.shape().equals(tangent.shape())
        || input.backend() != tangent.backend()
        || input.dtype() != tangent.dtype()
        || !input.device().equals(tangent.device()))
      throw new IllegalArgumentException("tangent signature mismatch");
    Scope scope = new Scope(input.backend());
    AD ad = new AD(scope, true);
    try {
      Node leaf = new Node(ad, input.buffer(), scope.t(tangent.buffer()), List.of(), null);
      Node end = ad.node(body.apply(new Tensor(ad.proxy, leaf)).buffer());
      Tensor derivative = end.tangent == null ? scope.t(end.primal).fullLike(0) : end.tangent;
      Tensor value = scope.independent(end.primal);
      derivative = scope.independent(derivative.buffer());
      scope.keep(value.buffer(), derivative.buffer());
      return new Jvp(
          new Tensor(scope.real, value.buffer()), new Tensor(scope.real, derivative.buffer()));
    } finally {
      scope.close();
    }
  }

  public record ValueAndGradTree<P>(Tensor value, P grads) {}

  public static <P> ValueAndGradTree<P> valueAndGradTree(Function<P, Tensor> body, P input) {
    Vjp<Tensor, P> result = vjp(body, input, null);
    return new ValueAndGradTree<>(result.value(), result.cotangents());
  }

  public static <P> P gradTree(Function<P, Tensor> body, P input) {
    ValueAndGradTree<P> result = valueAndGradTree(body, input);
    boolean borrowed =
        Jit.tensorLeaves(TreeUtil.flatten(input).leaves()).stream()
            .anyMatch(t -> t.buffer() == result.value().buffer());
    if (!borrowed) result.value().close();
    return result.grads();
  }

  public record JvpTree<R>(R value, R tangent) {}

  /** Forward-mode over matching tensor pytrees. */
  @SuppressWarnings("unchecked")
  public static <P, R> JvpTree<R> jvpTree(Function<P, R> body, P inputs, P tangents) {
    var flat = TreeUtil.flatten(inputs);
    var dual = TreeUtil.flatten(tangents);
    if (!flat.def().equals(dual.def())) throw new IllegalArgumentException("tangent tree mismatch");
    List<Tensor> xs = Jit.tensorLeaves(flat.leaves()), ts = Jit.tensorLeaves(dual.leaves());
    if (xs.isEmpty()) throw new IllegalArgumentException("JVP requires inputs");
    Scope scope = new Scope(xs.getFirst().backend());
    AD ad = new AD(scope, true);
    try {
      List<Tensor> wrapped = new ArrayList<>();
      for (int i = 0; i < xs.size(); i++) {
        if (!xs.get(i).shape().equals(ts.get(i).shape())
            || xs.get(i).dtype() != ts.get(i).dtype()
            || !xs.get(i).device().equals(ts.get(i).device()))
          throw new IllegalArgumentException("tangent signature mismatch");
        wrapped.add(
            new Tensor(
                ad.proxy,
                new Node(ad, xs.get(i).buffer(), scope.t(ts.get(i).buffer()), List.of(), null)));
      }
      var output = TreeUtil.flatten(body.apply((P) TreeUtil.unflatten(flat.def(), wrapped)));
      List<Tensor> values = new ArrayList<>(), derivatives = new ArrayList<>();
      for (Object leaf : output.leaves()) {
        Node end = ad.node(((Tensor) leaf).buffer());
        Tensor t = scope.independent(ad.tangent(end).buffer());
        Tensor value = scope.independent(end.primal);
        scope.keep(value.buffer(), t.buffer());
        values.add(new Tensor(scope.real, value.buffer()));
        derivatives.add(new Tensor(scope.real, t.buffer()));
      }
      return new JvpTree<>(
          (R) TreeUtil.unflatten(output.def(), values),
          (R) TreeUtil.unflatten(output.def(), derivatives));
    } finally {
      scope.close();
    }
  }

  /** Structured VJP evaluated at explicit output cotangents. */
  @SuppressWarnings("unchecked")
  public static <P, R> Vjp<R, P> vjp(Function<P, R> body, P inputs, R cotangents) {
    var flat = TreeUtil.flatten(inputs);
    List<Tensor> tensors = Jit.tensorLeaves(flat.leaves());
    if (tensors.isEmpty()) throw new IllegalArgumentException("VJP requires inputs");
    Scope scope = new Scope(tensors.getFirst().backend());
    AD ad = new AD(scope, false);
    List<Node> leaves = new ArrayList<>();
    try {
      List<Tensor> wrapped = new ArrayList<>();
      for (Tensor t : tensors) {
        if (t.backend() != scope.real) throw new IllegalArgumentException("VJP placement mismatch");
        Node leaf = new Node(ad, t.buffer(), null, List.of(), null);
        leaves.add(leaf);
        wrapped.add(new Tensor(ad.proxy, leaf));
      }
      var output = TreeUtil.flatten(body.apply((P) TreeUtil.unflatten(flat.def(), wrapped)));
      var cot = TreeUtil.flatten(cotangents);
      if (cotangents != null && !cot.def().equals(output.def()))
        throw new IllegalArgumentException("cotangent tree mismatch");
      List<Node> ends = new ArrayList<>();
      List<Tensor> seeds = new ArrayList<>(), values = new ArrayList<>();
      for (int i = 0; i < output.leaves().size(); i++) {
        Node end = ad.node(((Tensor) output.leaves().get(i)).buffer());
        if (cotangents == null && (output.leaves().size() != 1 || end.shape().size() != 1))
          throw new IllegalArgumentException("grad requires scalar output");
        Tensor seed =
            cotangents == null ? scope.t(end.primal).fullLike(1) : (Tensor) cot.leaves().get(i);
        if (!seed.shape().equals(end.shape())
            || seed.dtype() != end.dtype()
            || !seed.device().equals(end.device()))
          throw new IllegalArgumentException("cotangent signature mismatch");
        ends.add(end);
        seeds.add(scope.t(seed.buffer()));
        Tensor value = scope.independent(end.primal);
        values.add(new Tensor(scope.real, value.buffer()));
        scope.keep(value.buffer());
      }
      var grads = ad.backward(ends, seeds);
      List<Tensor> results = new ArrayList<>();
      for (Node leaf : leaves) {
        Tensor g = grads.get(leaf);
        if (g == null) g = scope.t(leaf.primal).fullLike(0);
        g = scope.independent(g.buffer());
        scope.keep(g.buffer());
        results.add(new Tensor(scope.real, g.buffer()));
      }
      return new Vjp<>(
          (R) TreeUtil.unflatten(output.def(), values),
          (P) TreeUtil.unflatten(flat.def(), results));
    } finally {
      scope.close();
    }
  }

  private static Tensor basis(Tensor context, Shape shape, int coordinate) {
    int[] indices = new int[Math.toIntExact(shape.size())];
    Arrays.fill(indices, -1);
    indices[coordinate] = 0;
    try (Tensor one =
        new Tensor(
            context.backend(),
            context.backend().full(1, Shape.of(1), context.dtype(), context.device()))) {
      return one.reindex(shape, indices);
    }
  }

  /**
   * Forward Jacobian, output dimensions followed by input dimensions. Basis sweeps are explicit.
   */
  public static Tensor jacfwd(UnaryOperator<Tensor> body, Tensor input) {
    List<Tensor> columns = new ArrayList<>();
    Tensor output = null;
    try {
      for (int i = 0; i < input.shape().size(); i++) {
        try (Tensor basis = basis(input, input.shape(), i)) {
          Jvp result = jvp(body, input, basis);
          if (output != null && output != input) output.close();
          output = result.value();
          columns.add(result.tangent());
        }
      }
      if (columns.isEmpty()) throw new IllegalArgumentException("empty Jacobian input");
      try (Tensor stacked = Tensor.stackAxis0(columns)) {
        int[] axes = new int[stacked.shape().rank()];
        for (int i = 0; i < axes.length - 1; i++) axes[i] = i + 1;
        axes[axes.length - 1] = 0;
        try (Tensor permuted = stacked.permute(axes)) {
          int[] dims = new int[output.shape().rank() + input.shape().rank()];
          System.arraycopy(output.shape().dims(), 0, dims, 0, output.shape().rank());
          System.arraycopy(
              input.shape().dims(), 0, dims, output.shape().rank(), input.shape().rank());
          return permuted.reshape(dims);
        }
      }
    } finally {
      columns.forEach(Tensor::close);
      if (output != null && output != input) output.close();
    }
  }

  /** Reverse Jacobian; one pullback per output coordinate. */
  public static Tensor jacrev(UnaryOperator<Tensor> body, Tensor input) {
    Tensor output = Jit.apply(body, input);
    List<Tensor> rows = new ArrayList<>();
    try {
      for (int i = 0; i < output.shape().size(); i++) {
        try (Tensor seed = basis(input, output.shape(), i)) {
          ValueAndGrad r = reverse(body, input, seed, false);
          rows.add(r.grad());
          if (r.value() != input) r.value().close();
        }
      }
      try (Tensor stacked = Tensor.stackAxis0(rows)) {
        int[] dims = new int[output.shape().rank() + input.shape().rank()];
        System.arraycopy(output.shape().dims(), 0, dims, 0, output.shape().rank());
        System.arraycopy(
            input.shape().dims(), 0, dims, output.shape().rank(), input.shape().rank());
        return stacked.reshape(dims);
      }
    } finally {
      rows.forEach(Tensor::close);
      if (output != input) output.close();
    }
  }

  public static Tensor hessian(UnaryOperator<Tensor> body, Tensor input) {
    return jacfwd(grad(body), input);
  }

  private static final class Node implements TracingBuffer {
    public ComputeBackend tracingBackend() {
      return owner.proxy;
    }

    final AD owner;
    final DeviceBuffer primal;
    final Tensor tangent;
    final List<Node> parents;
    final Function<Tensor, List<Tensor>> pullback;

    Node(
        AD owner,
        DeviceBuffer primal,
        Tensor tangent,
        List<Node> parents,
        Function<Tensor, List<Tensor>> pullback) {
      this.owner = owner;
      this.primal = primal;
      this.tangent = tangent;
      this.parents = parents;
      this.pullback = pullback;
    }

    public Shape shape() {
      return primal.shape();
    }

    public DType dtype() {
      return primal.dtype();
    }

    public Device device() {
      return primal.device();
    }
  }

  /** Own all temporary forward/backward allocations, including rule-internal tensors. */
  private static final class Scope implements AutoCloseable, InvocationHandler {
    final ComputeBackend real, proxy;
    final Set<DeviceBuffer> owned = Collections.newSetFromMap(new IdentityHashMap<>()),
        kept = Collections.newSetFromMap(new IdentityHashMap<>());

    Scope(ComputeBackend real) {
      this.real = real;
      proxy =
          (ComputeBackend)
              Proxy.newProxyInstance(
                  ComputeBackend.class.getClassLoader(),
                  new Class<?>[] {TracingBackend.class},
                  this);
    }

    Tensor t(DeviceBuffer b) {
      return new Tensor(proxy, b);
    }

    Tensor independent(DeviceBuffer buffer) {
      Tensor tensor = t(buffer);
      return owned.contains(buffer) ? tensor : tensor.mul(1);
    }

    void keep(DeviceBuffer... bs) {
      kept.addAll(Arrays.asList(bs));
    }

    public Object invoke(Object p, Method m, Object[] args) throws Throwable {
      if (m.getName().equals("parentBackend")) return real;
      if (m.getName().equals("release")) return null;
      Object r = call(real, m, args);
      if (r instanceof DeviceBuffer b && !operand(b, args)) owned.add(b);
      return r;
    }

    public void close() {
      for (DeviceBuffer b : owned) if (!kept.contains(b)) real.release(b);
    }
  }

  private static boolean operand(DeviceBuffer b, Object[] args) {
    if (args != null)
      for (Object arg : args) {
        if (arg == b) return true;
        if (arg instanceof DeviceBuffer[] bs) for (var x : bs) if (x == b) return true;
      }
    return false;
  }

  private static Object call(ComputeBackend backend, Method method, Object[] args)
      throws Throwable {
    try {
      return method.invoke(backend, args);
    } catch (InvocationTargetException e) {
      throw e.getCause();
    }
  }

  private static final class AD implements InvocationHandler {
    final Scope scope;
    final ComputeBackend proxy;
    final boolean forward;

    AD(Scope scope, boolean forward) {
      this.scope = scope;
      this.forward = forward;
      proxy =
          (ComputeBackend)
              Proxy.newProxyInstance(
                  ComputeBackend.class.getClassLoader(),
                  new Class<?>[] {TracingBackend.class},
                  this);
    }

    Node node(DeviceBuffer b) {
      return b instanceof Node n && n.owner == this ? n : new Node(this, b, null, List.of(), null);
    }

    Tensor primal(Node n) {
      return scope.t(n.primal);
    }

    Tensor tangent(Node n) {
      return n.tangent == null ? primal(n).fullLike(0) : n.tangent;
    }

    public Object invoke(Object p, Method m, Object[] raw) throws Throwable {
      String name = m.getName();
      Object[] args = raw == null ? new Object[0] : raw.clone();
      if (name.equals("parentBackend")) return scope.proxy;
      if (name.equals("release")) return null;
      if (name.equals("download"))
        throw new IllegalStateException("Host reads inside differentiation are unsupported");
      if (name.equals("fused")) return InvocationHandler.invokeDefault(p, m, args);
      List<Node> parents = new ArrayList<>();
      for (int i = 0; i < args.length; i++) {
        if (args[i] instanceof DeviceBuffer b) {
          Node n = node(b);
          parents.add(n);
          args[i] = n.primal;
        } else if (args[i] instanceof DeviceBuffer[] bs) {
          DeviceBuffer[] actual = bs.clone();
          for (int j = 0; j < bs.length; j++) {
            Node n = node(bs[j]);
            parents.add(n);
            actual[j] = n.primal;
          }
          args[i] = actual;
        }
      }
      Object result = call(scope.proxy, m, args);
      if (!(result instanceof DeviceBuffer output)) return result;
      if (parents.isEmpty()) return new Node(this, output, null, List.of(), null);
      Tensor a = primal(parents.getFirst()),
          b = parents.size() > 1 ? primal(parents.get(1)) : null,
          y = scope.t(output);
      Rule rule = rule(name, args, a, b, y, parents);
      Tensor derivative =
          forward ? rule.push.apply(parents.stream().map(this::tangent).toList()) : null;
      if (derivative != null
          && (!derivative.shape().equals(output.shape())
              || derivative.dtype() != output.dtype()
              || !derivative.device().equals(output.device())))
        throw new IllegalArgumentException("JVP rule returned an incompatible tangent");
      return new Node(this, output, derivative, parents, rule.pull);
    }

    Map<Node, Tensor> backward(List<Node> ends, List<Tensor> seeds) {
      List<Node> topo = new ArrayList<>();
      Set<Node> visited = Collections.newSetFromMap(new IdentityHashMap<>());
      for (Node n : ends) visit(n, visited, topo);
      Map<Node, Tensor> grads = new IdentityHashMap<>();
      for (int i = 0; i < ends.size(); i++) add(grads, ends.get(i), seeds.get(i));
      for (int i = topo.size() - 1; i >= 0; i--) {
        Node n = topo.get(i);
        Tensor upstream = grads.get(n);
        if (upstream == null || n.pullback == null) continue;
        List<Tensor> gs = n.pullback.apply(upstream);
        if (gs.size() != n.parents.size()) throw new IllegalStateException("pullback arity");
        for (int j = 0; j < gs.size(); j++) {
          Node parent = n.parents.get(j);
          Tensor g = gs.get(j);
          if (!g.shape().equals(parent.shape())
              || g.dtype() != parent.dtype()
              || !g.device().equals(parent.device()))
            throw new IllegalArgumentException("VJP rule returned an incompatible cotangent");
          add(grads, parent, g);
        }
      }
      return grads;
    }

    void add(Map<Node, Tensor> grads, Node node, Tensor g) {
      grads.merge(node, g, Tensor::add);
    }

    void visit(Node n, Set<Node> visited, List<Node> topo) {
      if (visited.add(n)) {
        for (Node parent : n.parents) visit(parent, visited, topo);
        topo.add(n);
      }
    }

    record Rule(Function<List<Tensor>, Tensor> push, Function<Tensor, List<Tensor>> pull) {}

    Rule linear(Function<Tensor, Tensor> push, Function<Tensor, Tensor> pull) {
      return new Rule(ts -> push.apply(ts.getFirst()), g -> List.of(pull.apply(g)));
    }

    Rule local(Tensor derivative) {
      return linear(t -> t.mul(derivative), g -> g.mul(derivative));
    }

    Tensor reduceTo(Tensor g, Shape target) {
      while (g.shape().rank() > target.rank()) g = g.sum(0);
      for (int i = 0; i < target.rank(); i++)
        if (target.dim(i) == 1 && g.shape().dim(i) != 1) g = g.sum(i, true);
      return g.reshape(target.dims());
    }

    Tensor lastTranspose(Tensor t) {
      int[] axes = new int[t.shape().rank()];
      for (int i = 0; i < axes.length; i++) axes[i] = i;
      axes[axes.length - 1] = axes.length - 2;
      axes[axes.length - 2] = axes.length - 1;
      return t.permute(axes);
    }

    Rule rule(String name, Object[] args, Tensor a, Tensor b, Tensor y, List<Node> parents) {
      return switch (name) {
        case "program" -> {
          DeviceProgram program = (DeviceProgram) args[0];
          List<Tensor> xs = parents.stream().map(this::primal).toList();
          yield new Rule(ts -> program.push(xs, ts), g -> program.pull(xs, g));
        }
        case "custom" -> {
          CustomDerivative custom = (CustomDerivative) args[1];
          yield new Rule(
              ts -> {
                if (custom.jvp == null)
                  throw new UnsupportedOperationException("A custom VJP has no forward-mode rule");
                return custom.jvp.apply(a, ts.getFirst());
              },
              g -> {
                if (custom.vjp != null) return List.of(custom.vjp.apply(a, g));
                Tensor zero = a.fullLike(0);
                return List.of(Transforms.grad(t -> custom.jvp.apply(a, t).mul(g).sum(), zero));
              });
        }
        case "binary" -> {
          Op op = (Op) args[0];
          Tensor da, db;
          switch (op) {
            case ADD -> {
              da = a.fullLike(1);
              db = b.fullLike(1);
            }
            case SUB -> {
              da = a.fullLike(1);
              db = b.fullLike(-1);
            }
            case MUL -> {
              da = b;
              db = a;
            }
            case DIV -> {
              da = b.fullLike(1).div(b);
              db = a.neg().div(b.mul(b));
            }
            case MAX, MIN -> {
              Tensor sign = a.sub(b).sign().mul(op == Op.MAX ? 1 : -1);
              da = sign.add(1).mul(.5);
              db = sign.neg().add(1).mul(.5);
            }
            default -> throw new UnsupportedOperationException("binary derivative " + op);
          }
          yield new Rule(
              ts -> ts.get(0).mul(da).add(ts.get(1).mul(db)), g -> List.of(g.mul(da), g.mul(db)));
        }
        case "scalar" -> {
          double v = (double) args[2];
          Op op = (Op) args[0];
          yield switch (op) {
            case ADD, SUB -> linear(t -> t, g -> g);
            case MUL -> local(a.fullLike((float) v));
            case DIV -> local(a.fullLike((float) (1 / v)));
            case MAX, MIN -> local(a.sub(v).sign().mul(op == Op.MAX ? 1 : -1).add(1).mul(.5));
            default -> throw new UnsupportedOperationException("scalar derivative " + op);
          };
        }
        case "unary" ->
            local(
                switch ((Op) args[0]) {
                  case NEG -> a.fullLike(-1);
                  case EXP -> y;
                  case LOG -> a.fullLike(1).div(a);
                  case SQRT -> a.fullLike(1).div(y.mul(2));
                  case RSQRT -> y.mul(y).mul(y).mul(-.5);
                  case TANH -> y.mul(y).neg().add(1);
                  case SIGMOID -> y.mul(y.neg().add(1));
                  case RELU -> a.sign().maximum(0);
                  case ABS -> a.sign();
                  case SIGN -> a.fullLike(0);
                  default -> throw new UnsupportedOperationException("unary derivative");
                });
        case "reshape" ->
            linear(t -> t.reshape(y.shape().dims()), g -> g.reshape(a.shape().dims()));
        case "reindex" ->
            linear(
                t -> t.reindex(y.shape(), (int[]) args[2]),
                g -> g.scatterAdd(a.shape(), (int[]) args[2]));
        case "scatterAdd" ->
            linear(
                t -> t.scatterAdd(y.shape(), (int[]) args[2]),
                g -> g.reindex(a.shape(), (int[]) args[2]));
        case "transpose" -> linear(Tensor::transpose, Tensor::transpose);
        case "broadcastTo" -> linear(t -> t.broadcastTo(y.shape()), g -> reduceTo(g, a.shape()));
        case "reduceSum" -> linear(Tensor::sum, g -> g.broadcastTo(a.shape()));
        case "sumAxis0", "sumAxis" -> {
          int axis = name.equals("sumAxis0") ? 0 : (int) args[1];
          boolean keep = args.length > 2 && (boolean) args[2];
          int[] dims = a.shape().dims();
          dims[axis] = 1;
          yield linear(t -> t.sum(axis, keep), g -> g.reshape(dims).broadcastTo(a.shape()));
        }
        case "matmul", "batchedMatmul" -> {
          boolean batched = name.equals("batchedMatmul");
          yield new Rule(
              ts ->
                  batched
                      ? ts.get(0).batchedMatmul(b).add(a.batchedMatmul(ts.get(1)))
                      : ts.get(0).matmul(b).add(a.matmul(ts.get(1))),
              g ->
                  List.of(
                      reduceTo(
                          batched ? g.batchedMatmul(lastTranspose(b)) : g.matmul(b.transpose()),
                          a.shape()),
                      reduceTo(
                          batched ? lastTranspose(a).batchedMatmul(g) : a.transpose().matmul(g),
                          b.shape())));
        }
        case "conv2d", "conv2dGradInput", "conv2dGradWeight" -> {
          ConvSpec s = (ConvSpec) args[2];
          BiFunction<Tensor, Tensor, Tensor> op =
              switch (name) {
                case "conv2d" -> (x, w) -> x.conv2d(w, s);
                case "conv2dGradInput" -> (g, w) -> g.conv2dGradInput(w, s);
                default -> (x, g) -> x.conv2dGradWeight(g, s);
              };
          yield new Rule(
              ts -> op.apply(ts.get(0), b).add(op.apply(a, ts.get(1))),
              g ->
                  switch (name) {
                    case "conv2d" -> List.of(g.conv2dGradInput(b, s), a.conv2dGradWeight(g, s));
                    case "conv2dGradInput" -> List.of(g.conv2d(b, s), g.conv2dGradWeight(a, s));
                    default -> List.of(b.conv2dGradInput(g, s), a.conv2d(g, s));
                  });
        }
        case "sliceAxis0" -> {
          int index = (int) args[1], size = Math.toIntExact(y.shape().size());
          int[] map = new int[size];
          for (int i = 0; i < size; i++) map[i] = index * size + i;
          yield linear(t -> t.sliceAxis0(index), g -> g.scatterAdd(a.shape(), map));
        }
        case "stackAxis0" ->
            new Rule(
                Tensor::stackAxis0,
                g -> {
                  List<Tensor> parts = new ArrayList<>();
                  for (int i = 0; i < parents.size(); i++) parts.add(g.sliceAxis0(i));
                  return parts;
                });
        case "argMaxAxis" -> linear(t -> y.fullLike(0), g -> a.fullLike(0));
        case "reluBackward" -> {
          Tensor mask = b.sign().maximum(0);
          yield new Rule(ts -> ts.get(0).mul(mask), g -> List.of(g.mul(mask), b.fullLike(0)));
        }
        case "maxAxis", "reduceMax" -> {
          boolean all = name.equals("reduceMax");
          int ax = all ? 0 : (int) args[1];
          boolean keep = !all && (boolean) args[2];
          Tensor input = all ? a.reshape(Math.toIntExact(a.shape().size())) : a;
          Tensor mask = input.maxAxisBackward(y.fullLike(1), ax).reshape(a.shape().dims());
          int[] dims = a.shape().dims();
          if (!all) dims[ax] = 1;
          yield linear(
              t -> all ? t.mul(mask).sum() : t.mul(mask).sum(ax, keep),
              g ->
                  (all ? g.broadcastTo(a.shape()) : g.reshape(dims).broadcastTo(a.shape()))
                      .mul(mask));
        }
        case "maxAxisBackward" -> {
          int ax = (int) args[2];
          Tensor mask = b.maxAxisBackward(a.fullLike(1), ax);
          int[] dims = b.shape().dims();
          dims[ax] = 1;
          yield new Rule(
              ts -> ts.get(0).maxAxisBackward(b, ax),
              g -> List.of(g.mul(mask).sum(ax).reshape(a.shape().dims()), b.fullLike(0)));
        }
        default -> throw new UnsupportedOperationException("No tensor derivative rule for " + name);
      };
    }
  }
}
