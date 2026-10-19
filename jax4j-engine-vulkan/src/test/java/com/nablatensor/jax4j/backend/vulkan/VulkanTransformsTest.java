package com.nablatensor.jax4j.backend.vulkan;

import static org.junit.jupiter.api.Assertions.*;

import com.nablatensor.jax4j.tensor.*;
import java.util.*;
import java.util.function.UnaryOperator;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

@EnabledIfSystemProperty(named = "jax4j.vulkan.tests", matches = "true")
class VulkanTransformsTest {
  private Tensor array(float[] values, int... dims) {
    return Tensors.arrayOn(values, Shape.of(dims), Device.vulkan());
  }

  @Test
  void derivativesAndHigherOrders() {
    UnaryOperator<Tensor> cube = x -> x.mul(x).mul(x).sum();
    try (Tensor x = array(new float[] {1, 2, 3}, 3);
        Tensor t = x.fullLike(1)) {
      var jvp = Transforms.jvp(cube, x, t);
      try (Tensor value = jvp.value();
          Tensor tangent = jvp.tangent();
          Tensor g = Transforms.grad(cube, x);
          Tensor h = Transforms.hessian(cube, x);
          Tensor second = Transforms.grad(y -> Transforms.grad(cube, y).sum(), x)) {
        assertArrayEquals(new float[] {36}, value.toFloatArray(), 1e-5f);
        assertArrayEquals(new float[] {42}, tangent.toFloatArray(), 1e-5f);
        assertArrayEquals(new float[] {3, 12, 27}, g.toFloatArray(), 1e-5f);
        assertArrayEquals(new float[] {6, 0, 0, 0, 12, 0, 0, 0, 18}, h.toFloatArray(), 1e-5f);
        assertArrayEquals(new float[] {6, 12, 18}, second.toFloatArray(), 1e-5f);
      }
    }
  }

  @Test
  void jacobiansAndStructuredVjp() {
    try (Tensor x = array(new float[] {1, 2}, 2);
        Tensor seed = array(new float[] {2, 3}, 2);
        Tensor fwd = Transforms.jacfwd(t -> t.mul(t), x);
        Tensor rev = Transforms.jacrev(t -> t.mul(t), x)) {
      assertArrayEquals(new float[] {2, 0, 0, 4}, fwd.toFloatArray(), 1e-5f);
      assertArrayEquals(fwd.toFloatArray(), rev.toFloatArray(), 1e-5f);
      var result =
          Transforms.vjp(
              (List<Tensor> ts) -> Map.of("square", ts.get(0).mul(ts.get(0))),
              List.of(x),
              Map.of("square", seed));
      try (Tensor value = result.value().get("square");
          Tensor grad = result.cotangents().get(0)) {
        assertArrayEquals(new float[] {4, 12}, grad.toFloatArray(), 1e-5f);
      }
    }
  }

  @Test
  void compiledDerivativeChangesValues() {
    var plan = Jit.compile(Transforms.grad(x -> x.exp().sum()));
    try (Tensor x = array(new float[] {0, 1}, 2);
        Tensor y = array(new float[] {2, 3}, 2);
        Tensor first = plan.apply(x);
        Tensor second = plan.apply(y)) {
      assertArrayEquals(new float[] {1, (float) Math.E}, first.toFloatArray(), 1e-5f);
      assertArrayEquals(
          new float[] {(float) Math.exp(2), (float) Math.exp(3)}, second.toFloatArray(), 1e-4f);
      assertEquals(1, plan.cachedPlanCount());
    }
  }

  @Test
  void customRulesAndCheckpoint() {
    var custom = CustomDerivative.jvp(x -> x.mul(x), (x, t) -> x.mul(t).mul(3));
    var reverse = CustomDerivative.vjp(x -> x.mul(x), (x, g) -> x.mul(g).mul(4));
    try (Tensor x = array(new float[] {1, 2}, 2);
        Tensor g = Transforms.grad(t -> custom.apply(t).sum(), x);
        Tensor h = Transforms.grad(t -> reverse.apply(t).sum(), x);
        Tensor remat =
            Transforms.grad(
                t -> CustomDerivative.checkpoint(y -> y.mul(y).tanh()).apply(t).sum(), x);
        Tensor ordinary = Transforms.grad(t -> t.mul(t).tanh().sum(), x)) {
      assertArrayEquals(new float[] {3, 6}, g.toFloatArray(), 1e-5f);
      assertArrayEquals(new float[] {4, 8}, h.toFloatArray(), 1e-5f);
      assertArrayEquals(ordinary.toFloatArray(), remat.toFloatArray(), 1e-5f);
    }
  }

  @Test
  void arbitraryAxesAndMultipleInputs() {
    try (Tensor x = array(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor y = array(new float[] {10, 20}, 2);
        Tensor result =
            Vmap.apply(ts -> ts.get(0).add(ts.get(1)), List.of(x, y), Arrays.asList(1, null), 1)) {
      assertEquals(Shape.of(2, 3), result.shape());
      assertArrayEquals(new float[] {11, 12, 13, 24, 25, 26}, result.toFloatArray(), 1e-5f);
    }
  }

  @Test
  void mappedTransposeSliceStackAndGrad() {
    try (Tensor x = array(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2);
        Tensor out =
            Jit.apply(
                Vmap.of(
                    t -> Tensor.stackAxis0(List.of(t.transpose().sliceAxis0(0), t.sliceAxis0(1)))),
                x);
        Tensor grad =
            Jit.apply(Vmap.of(Transforms.grad(t -> t.transpose().sliceAxis0(0).sum())), x)) {
      assertArrayEquals(new float[] {1, 3, 3, 4, 5, 7, 7, 8}, out.toFloatArray(), 1e-5f);
      assertArrayEquals(new float[] {1, 0, 1, 0, 1, 0, 1, 0}, grad.toFloatArray(), 1e-5f);
    }
  }

  @Test
  void mappedConvolutionAndDerivatives() {
    ConvSpec spec = new ConvSpec(1, 2, 2, 1, 1, 1, 0);
    try (Tensor x = array(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, 2, 1, 4);
        Tensor w = array(new float[] {2, 3}, 2, 1, 1);
        Tensor out =
            Vmap.apply(ts -> ts.get(0).conv2d(ts.get(1), spec), List.of(x, w), List.of(0, 0), 0)) {
      assertArrayEquals(new float[] {2, 4, 6, 8, 15, 18, 21, 24}, out.toFloatArray(), 1e-5f);
    }
    try (Tensor x = array(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, 2, 1, 4);
        Tensor w = array(new float[] {2}, 1, 1);
        Tensor dx =
            Vmap.apply(
                ts -> Transforms.grad(t -> t.conv2d(ts.get(1), spec).sum(), ts.get(0)),
                List.of(x, w),
                Arrays.asList(0, null),
                0)) {
      assertArrayEquals(new float[] {2, 2, 2, 2, 2, 2, 2, 2}, dx.toFloatArray(), 1e-5f);
    }
  }

  @Test
  void indexingTransposeAccumulatesRepeatedIndices() {
    try (Tensor x = array(new float[] {2, 3}, 2);
        Tensor grad =
            Transforms.grad(t -> t.reindex(Shape.of(4), new int[] {0, 0, 1, -1}).sum(), x)) {
      assertArrayEquals(new float[] {2, 1}, grad.toFloatArray(), 1e-5f);
    }
  }

  @Test
  void deviceConditionalAndLoop() {
    try (Tensor x = array(new float[] {0}, 1);
        Tensor p = array(new float[] {1}, 1);
        Tensor yes = ControlFlow.cond(p, t -> t.add(3), t -> t.sub(9), x);
        Tensor loop =
            ControlFlow.whileLoop(t -> t.neg().add(5).sign().maximum(0), t -> t.add(1), x)) {
      assertArrayEquals(new float[] {3}, yes.toFloatArray(), 0);
      assertArrayEquals(new float[] {5}, loop.toFloatArray(), 0);
    }
  }

  @Test
  void compiledScanDoesNotUnrollBody() {
    java.util.concurrent.atomic.AtomicInteger calls =
        new java.util.concurrent.atomic.AtomicInteger();
    var plan =
        Jit.compileTree(
            (List<Tensor> args) ->
                ControlFlow.scan(
                    (c, x) -> {
                      calls.incrementAndGet();
                      Tensor next = c.add(x);
                      return new ControlFlow.ScanStep(next, next.mul(2));
                    },
                    args.get(0),
                    args.get(1)));
    try (Tensor initial = array(new float[] {0}, 1);
        Tensor xs = array(new float[] {1, 2, 3, 4}, 4, 1)) {
      var result = plan.apply(List.of(initial, xs));
      try (Tensor carry = result.carry();
          Tensor ys = result.outputs()) {
        assertArrayEquals(new float[] {10}, carry.toFloatArray(), 0);
        assertArrayEquals(new float[] {2, 6, 12, 20}, ys.toFloatArray(), 0);
      }
      assertEquals(1, calls.get());
      var again = plan.apply(List.of(initial, xs));
      again.carry().close();
      again.outputs().close();
      assertEquals(1, calls.get());
    }
  }

  @Test
  void differentiateDeviceRegions() {
    try (Tensor initial = array(new float[] {2}, 1);
        Tensor xs = array(new float[] {1, 2, 3}, 3, 1);
        Tensor p = array(new float[] {1}, 1);
        Tensor tangent = initial.fullLike(1)) {
      try (Tensor conditional =
              Transforms.grad(
                  t -> ControlFlow.cond(p, x -> x.mul(x), x -> x.mul(7), t).sum(), initial);
          Tensor scan =
              Transforms.grad(
                  t ->
                      ControlFlow.scan(
                              (c, x) -> {
                                Tensor n = c.mul(x);
                                return new ControlFlow.ScanStep(n, n);
                              },
                              t,
                              xs)
                          .carry()
                          .sum(),
                  initial)) {
        assertArrayEquals(new float[] {4}, conditional.toFloatArray(), 1e-5f);
        assertArrayEquals(new float[] {6}, scan.toFloatArray(), 1e-5f);
      }
      var loop =
          Transforms.jvp(
              t -> ControlFlow.whileLoop(x -> x.neg().add(10).sign().maximum(0), x -> x.mul(2), t),
              initial,
              tangent);
      try (Tensor value = loop.value();
          Tensor derivative = loop.tangent()) {
        assertArrayEquals(new float[] {16}, value.toFloatArray(), 1e-5f);
        assertArrayEquals(new float[] {8}, derivative.toFloatArray(), 1e-5f);
      }
    }
  }

  @Test
  void nestedVmapAndMappedControlFlow() {
    try (Tensor x = array(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2);
        Tensor nested = Vmap.apply(Vmap.of(t -> t.mul(2)), x);
        Tensor loop =
            Vmap.apply(
                t ->
                    ControlFlow.whileLoop(
                        v -> v.sum().neg().add(9).sign().maximum(0), v -> v.add(1), t),
                x);
        Tensor grad =
            Transforms.grad(
                t ->
                    Vmap.apply(
                            v ->
                                ControlFlow.scan(
                                        (c, z) -> new ControlFlow.ScanStep(c.mul(z), c),
                                        v,
                                        v.reshape(2, 2))
                                    .carry(),
                            t)
                        .sum(),
                x)) {
      assertArrayEquals(new float[] {2, 4, 6, 8, 10, 12, 14, 16}, nested.toFloatArray(), 0);
      assertArrayEquals(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, loop.toFloatArray(), 0);
      assertArrayEquals(new float[] {15, 32, 7, 20, 119, 160, 95, 132}, grad.toFloatArray(), 1e-5f);
    }
  }

  @Test
  void independentJaxFixtures() throws Exception {
    Properties fixture = new Properties();
    try (var stream = getClass().getResourceAsStream("/jax-parity.properties")) {
      fixture.load(stream);
    }
    java.util.function.Function<String, float[]> values =
        key -> {
          String[] parts = fixture.getProperty(key).split(",");
          float[] data = new float[parts.length];
          for (int j = 0; j < data.length; j++) data[j] = Float.parseFloat(parts[j]);
          return data;
        };
    UnaryOperator<Tensor> loss =
        p -> {
          Tensor y = p.mul(1.5).tanh();
          return y.mul(y).sum();
        };
    try (Tensor x = array(values.apply("input"), 4);
        Tensor t = array(values.apply("tangent"), 4);
        Tensor h = Transforms.hessian(loss, x);
        Tensor jac = Transforms.jacfwd(p -> p.mul(1.5).tanh(), x)) {
      var result = Transforms.jvp(loss, x, t);
      try (Tensor value = result.value();
          Tensor tangent = result.tangent()) {
        assertArrayEquals(values.apply("value"), value.toFloatArray(), 3e-5f);
        assertArrayEquals(values.apply("directional"), tangent.toFloatArray(), 3e-5f);
      }
      assertArrayEquals(values.apply("hessian"), h.toFloatArray(), 3e-5f);
      assertArrayEquals(values.apply("jacobian"), jac.toFloatArray(), 3e-5f);
    }
    try (Tensor initial = array(new float[] {2}, 1);
        Tensor xs = array(new float[] {1, 2, 3}, 3, 1);
        Tensor dx =
            Transforms.grad(
                t ->
                    ControlFlow.scan(
                            (c, v) -> {
                              Tensor n = c.mul(v);
                              return new ControlFlow.ScanStep(n, n);
                            },
                            initial,
                            t)
                        .carry()
                        .sum(),
                xs)) {
      assertArrayEquals(values.apply("scanGradInputs"), dx.toFloatArray(), 3e-5f);
    }
  }

  @Test
  void treeControlFlowAndNestedRegions() {
    try (Tensor x = array(new float[] {1}, 1);
        Tensor y = array(new float[] {2}, 1);
        Tensor xs = array(new float[] {1, 2, 3}, 3, 1)) {
      Map<String, Tensor> initial = Map.of("a", x, "b", y);
      var result =
          ControlFlow.whileTree(
              (Map<String, Tensor> state) -> state.get("a").neg().add(3).sign().maximum(0),
              state -> Map.of("a", state.get("a").add(1), "b", state.get("b").mul(2)),
              initial);
      try (Tensor a = result.get("a");
          Tensor b = result.get("b")) {
        assertArrayEquals(new float[] {3}, a.toFloatArray(), 0);
        assertArrayEquals(new float[] {8}, b.toFloatArray(), 0);
      }
      try (Tensor nested =
          ControlFlow.whileLoop(
              t -> t.neg().add(4).sign().maximum(0),
              t -> ControlFlow.cond(t.sign(), v -> v.add(1), v -> v.add(2), t),
              x)) {
        assertArrayEquals(new float[] {4}, nested.toFloatArray(), 0);
      }
      var scan =
          ControlFlow.scanTree(
              (Map<String, Tensor> c, Map<String, Tensor> v) ->
                  new ControlFlow.TreeScanStep<>(
                      Map.of("a", c.get("a").add(v.get("x")), "b", c.get("b").mul(2)),
                      Map.of("y", c.get("a"))),
              initial,
              Map.of("x", xs));
      try (Tensor a = scan.carry().get("a");
          Tensor b = scan.carry().get("b");
          Tensor outputs = scan.outputs().get("y")) {
        assertArrayEquals(new float[] {7}, a.toFloatArray(), 0);
        assertArrayEquals(new float[] {16}, b.toFloatArray(), 0);
        assertArrayEquals(new float[] {1, 2, 4}, outputs.toFloatArray(), 0);
      }
    }
  }

  @Test
  void randomGenerationStaysInTracingContext() {
    var key = PrngKey.of(123).split(2).get(0).foldIn(7);
    var plan = Jit.compile(t -> key.uniformLike(t, 3).add(t));
    try (Tensor x = array(new float[] {0, 0, 0}, 3);
        Tensor first = plan.apply(x);
        Tensor next = plan.apply(x);
        Tensor sequence = x.reshape(1, 3);
        Tensor batch = array(new float[6], 2, 3);
        Tensor mapped = Jit.apply(Vmap.of(t -> key.uniformLike(t, 3).add(t)), batch)) {
      var scan =
          ControlFlow.scan(
              (c, v) -> new ControlFlow.ScanStep(c.add(key.uniformLike(c, 3)), c), x, sequence);
      try (Tensor carry = scan.carry();
          Tensor outputs = scan.outputs()) {
        assertArrayEquals(first.toFloatArray(), next.toFloatArray(), 0);
        assertArrayEquals(first.toFloatArray(), carry.toFloatArray(), 0);
        float[] expected = new float[6], samples = first.toFloatArray();
        System.arraycopy(samples, 0, expected, 0, 3);
        System.arraycopy(samples, 0, expected, 3, 3);
        assertArrayEquals(expected, mapped.toFloatArray(), 0);
      }
    }
  }

  @Test
  void higherDerivativesThroughRegionsAndCustomVmap() {
    var custom = CustomDerivative.jvp(t -> t.mul(t), (t, dt) -> t.mul(dt).mul(3));
    try (Tensor x = array(new float[] {1, 2}, 2);
        Tensor xs = array(new float[] {1, 2}, 2, 1);
        Tensor mapped = Transforms.grad(t -> Vmap.apply(custom, t).sum(), x);
        Tensor second =
            Transforms.grad(
                t ->
                    Transforms.grad(
                            v ->
                                ControlFlow.scan(
                                        (c, z) -> {
                                          Tensor next = c.mul(c).add(z);
                                          return new ControlFlow.ScanStep(next, next);
                                        },
                                        v,
                                        xs)
                                    .carry()
                                    .sum(),
                            t)
                        .sum(),
                x);
        Tensor condSecond =
            Transforms.grad(
                t ->
                    Transforms.grad(
                            v ->
                                ControlFlow.cond(
                                        v.sum().sign(), a -> a.mul(a).mul(a), a -> a.mul(a), v)
                                    .sum(),
                            t)
                        .sum(),
                x)) {
      assertArrayEquals(new float[] {3, 6}, mapped.toFloatArray(), 1e-5f);
      // Two steps give (x^2+1)^2+2; second derivative is 12*x^2+4.
      assertArrayEquals(new float[] {16, 52}, second.toFloatArray(), 1e-5f);
      assertArrayEquals(new float[] {6, 12}, condSecond.toFloatArray(), 1e-5f);
    }
  }

  @Test
  void structuredGradientsAndRuleContracts() {
    try (Tensor x = array(new float[] {1, 2}, 2);
        Tensor y = array(new float[] {3, 4}, 2)) {
      var gradients =
          Transforms.gradTree(
              (Map<String, Tensor> ps) -> ps.get("a").mul(ps.get("b")).sum(),
              Map.of("a", x, "b", y));
      try (Tensor a = gradients.get("a");
          Tensor b = gradients.get("b")) {
        assertArrayEquals(new float[] {3, 4}, a.toFloatArray(), 0);
        assertArrayEquals(new float[] {1, 2}, b.toFloatArray(), 0);
      }
      var bad = CustomDerivative.jvp(t -> t.mul(t), (t, dt) -> dt.sum());
      assertThrows(IllegalArgumentException.class, () -> Transforms.jvp(bad, x, y));
      assertThrows(
          IllegalArgumentException.class,
          () -> ControlFlow.whileLoop(t -> t.sum(), t -> t.sum(), x));
      assertThrows(
          IllegalArgumentException.class,
          () -> Vmap.apply(ts -> ts.get(0), List.of(x, y), List.of(0, 1), 0));
    }
  }

  @Test
  void identityDerivativesDoNotOwnBorrowedArguments() {
    try (Tensor x = array(new float[] {1, 2}, 2);
        Tensor tangent = array(new float[] {3, 4}, 2)) {
      var result = Transforms.jvp(t -> t, x, tangent);
      result.value().close();
      result.tangent().close();
      assertArrayEquals(new float[] {1, 2}, x.toFloatArray(), 0);
      assertArrayEquals(new float[] {3, 4}, tangent.toFloatArray(), 0);
      var reverse = Transforms.vjp((Tensor t) -> t, x, tangent);
      reverse.value().close();
      reverse.cotangents().close();
      assertArrayEquals(new float[] {1, 2}, x.toFloatArray(), 0);
      assertArrayEquals(new float[] {3, 4}, tangent.toFloatArray(), 0);
      try (Tensor forward = Transforms.jacfwd(t -> t, x);
          Tensor backward = Transforms.jacrev(t -> t, x)) {
        assertArrayEquals(new float[] {1, 0, 0, 1}, forward.toFloatArray(), 0);
        assertArrayEquals(forward.toFloatArray(), backward.toFloatArray(), 0);
      }
    }
  }

  record Training<P>(P parameters, com.nablatensor.jax4j.nn.FunctionalAdam.State<P> optimizer) {}

  @Test
  void completeCompiledTrainingStepUsesRuntimeOptimizerState() throws Exception {
    var optimizer = new com.nablatensor.jax4j.nn.FunctionalAdam(.01);
    java.util.concurrent.atomic.AtomicInteger captures =
        new java.util.concurrent.atomic.AtomicInteger();
    var train =
        Jit.compileTree(
            (Training<List<Tensor>> input) -> {
              captures.incrementAndGet();
              var gradients =
                  Transforms.gradTree(
                      (List<Tensor> ps) -> ps.getFirst().mul(ps.getFirst()).sum(),
                      input.parameters());
              var update = optimizer.update(input.parameters(), gradients, input.optimizer());
              com.nablatensor.jax4j.tensor.tree.TreeUtil.closeAll(gradients);
              return new Training<>(update.parameters(), update.state());
            });
    Tensor x = array(new float[] {1, 2}, 2);
    Training<List<Tensor>> state = new Training<>(List.of(x), optimizer.initialize(List.of(x)));
    try {
      for (int i = 0; i < 3; i++) {
        var previous = state;
        state = train.apply(previous);
        com.nablatensor.jax4j.tensor.tree.TreeUtil.closeAll(previous);
      }
      assertEquals(1, captures.get());
      assertEquals(3, state.optimizer().step().item(), 0);
      Properties fixture = new Properties();
      try (var stream = getClass().getResourceAsStream("/jax-parity.properties")) {
        fixture.load(stream);
      }
      String[] parts = fixture.getProperty("adamParameters").split(",");
      assertArrayEquals(
          new float[] {Float.parseFloat(parts[0]), Float.parseFloat(parts[1])},
          state.parameters().getFirst().toFloatArray(),
          3e-6f);
    } finally {
      com.nablatensor.jax4j.tensor.tree.TreeUtil.closeAll(state);
    }
  }

  @Test
  void remainingBatchingRulesAndSharedOutputs() {
    try (Tensor x = array(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 1, 2);
        Tensor w = array(new float[] {1, 1}, 1, 2, 1);
        Tensor output =
            Vmap.apply(
                ts -> ts.get(0).batchedMatmul(ts.get(1)),
                List.of(x, w),
                Arrays.asList(0, null),
                0)) {
      assertEquals(Shape.of(2, 2, 1, 1), output.shape());
      assertArrayEquals(new float[] {3, 7, 11, 15}, output.toFloatArray(), 0);
      assertArrayEquals(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, x.toFloatArray(), 0);
      assertArrayEquals(new float[] {1, 1}, w.toFloatArray(), 0);
    }
    ConvSpec spec = new ConvSpec(1, 2, 2, 1, 1, 1, 0);
    try (Tensor x = array(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, 2, 1, 4);
        Tensor upstream = array(new float[] {1, 1, 1, 1}, 1, 4);
        Tensor dw =
            Vmap.apply(
                ts -> ts.get(0).conv2dGradWeight(ts.get(1), spec),
                List.of(x, upstream),
                Arrays.asList(0, null),
                0)) {
      assertArrayEquals(new float[] {10, 26}, dw.toFloatArray(), 0);
      List<Tensor> results =
          Vmap.applyTree(
              (List<Tensor> args) -> List.of(args.get(0).mul(2), args.get(1).sum()),
              List.of(x, upstream),
              Arrays.asList(0, null),
              Arrays.asList(0, null));
      try (Tensor mapped = results.get(0);
          Tensor shared = results.get(1)) {
        assertEquals(x.shape(), mapped.shape());
        assertArrayEquals(new float[] {4}, shared.toFloatArray(), 0);
      }
    }
  }

  @Test
  void checkpointsRecomputeAndMutableOptimizersRejectCapture() {
    var calls = new java.util.concurrent.atomic.AtomicInteger();
    var checkpoint =
        CustomDerivative.checkpoint(
            t -> {
              calls.incrementAndGet();
              return t.mul(t).tanh();
            });
    try (Tensor x = array(new float[] {.2f, .3f}, 2);
        Tensor gradient = Transforms.grad(t -> checkpoint.apply(t).sum(), x);
        var mutable = new com.nablatensor.jax4j.nn.Adam(.01f)) {
      assertEquals(2, calls.get());
      assertThrows(
          IllegalStateException.class,
          () -> Jit.apply(t -> mutable.update(List.of(t), List.of(t)).getFirst(), x));
      assertEquals(2, gradient.toFloatArray().length);
    }
  }
}
