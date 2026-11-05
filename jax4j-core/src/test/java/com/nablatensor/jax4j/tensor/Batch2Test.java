package com.nablatensor.jax4j.tensor;

import static org.junit.jupiter.api.Assertions.*;

import com.nablatensor.jax4j.tensor.tree.TreeUtil;
import java.util.List;
import org.junit.jupiter.api.Test;

class Batch2Test {
  private static final Device DEVICE = Device.cpu();

  private Tensor input(float... values) {
    return Tensors.arrayOn(values, Shape.of(values.length), DEVICE);
  }

  private Tensor matrix(float[] values, int... dims) {
    return Tensors.arrayOn(values, Shape.of(dims), DEVICE);
  }

  private void equal(float[] expected, Tensor actual) {
    assertEquals(DEVICE, actual.device());
    assertArrayEquals(expected, actual.toFloatArray(), 2e-5f);
  }

  private void close(Object tree) {
    TreeUtil.closeAll(tree);
  }

  @Test
  void comparisonsAndLogicalMasks() {
    try (Tensor x = input(-1, 0, 2, Float.NaN, Float.POSITIVE_INFINITY);
        Tensor finite = x.isfinite();
        Tensor nan = x.isnan();
        Tensor less = x.lt(0);
        Tensor equal = x.eq(0);
        Tensor both = less.or(equal);
        Tensor not = both.not();
        Tensor f = Jit.compile(t -> t.isfinite().and(t.ge(0))).apply(x)) {
      equal(new float[] {1, 1, 1, 0, 0}, finite);
      equal(new float[] {0, 0, 0, 1, 0}, nan);
      equal(new float[] {1, 0, 0, 0, 0}, less);
      equal(new float[] {0, 1, 0, 0, 0}, equal);
      equal(new float[] {1, 1, 0, 0, 0}, both);
      equal(new float[] {0, 0, 1, 1, 1}, not);
      equal(new float[] {0, 1, 1, 0, 0}, f);
    }
    try (Tensor x = matrix(new float[] {1, 2}, 2, 1);
        Tensor y = input(0, 2, 3);
        Tensor b = x.lt(y)) {
      assertEquals(Shape.of(2, 3), b.shape());
      equal(new float[] {0, 1, 1, 0, 0, 1}, b);
    }
  }

  @Test
  void whereBroadcastAndUnselectedNan() {
    try (Tensor mask = input(1, 0);
        Tensor yes = input(5, Float.NaN);
        Tensor no = input(Float.NaN, 7);
        Tensor result = Tensor.where(mask, yes, no)) {
      equal(new float[] {5, 7}, result);
      equal(new float[] {1, 0}, mask);
      equal(new float[] {5, Float.NaN}, yes);
    }
    try (Tensor x = matrix(new float[] {-1, 2}, 2, 1);
        Tensor bias = input(10, 20, 30);
        Tensor zeros = x.fullLike(0);
        Tensor mask = x.gt(0);
        Tensor result = Tensors.where(mask, bias, zeros);
        Tensor gradient =
            Transforms.grad(t -> Tensor.where(t.gt(0), t.square(), t.fullLike(0)).sum(), x)) {
      equal(new float[] {0, 0, 0, 10, 20, 30}, result);
      equal(new float[] {0, 4}, gradient);
    }
  }

  @Test
  void trigPowerAndHypot() {
    try (Tensor x = input(-2, -.5f, 0, .5f, 2);
        Tensor sin = x.sin();
        Tensor cos = x.cos();
        Tensor power = x.pow(3);
        Tensor js = Jit.compile(t -> t.sin().add(t.cos())).apply(x);
        Tensor gradient = Transforms.grad(t -> t.sin().sum(), x)) {
      float[] values = x.toFloatArray(),
          s = sin.toFloatArray(),
          c = cos.toFloatArray(),
          pow = power.toFloatArray(),
          compiled = js.toFloatArray(),
          g = gradient.toFloatArray();
      for (int i = 0; i < values.length; i++) {
        assertEquals(Math.sin(values[i]), s[i], 2e-6);
        assertEquals(Math.cos(values[i]), c[i], 2e-6);
        assertEquals(Math.pow(values[i], 3), pow[i], 2e-5);
        assertEquals(s[i] + c[i], compiled[i], 2e-5);
        assertEquals(c[i], g[i], 2e-5);
      }
    }
    try (Tensor x = input(3, 1e30f, 0, Float.POSITIVE_INFINITY);
        Tensor y = input(4, 1e30f, 0, Float.NaN);
        Tensor h = x.hypot(y)) {
      float[] v = h.toFloatArray();
      assertEquals(5, v[0], 1e-6);
      assertEquals(Math.hypot(1e30f, 1e30f), v[1], 2e24);
      assertEquals(0, v[2]);
      assertEquals(Float.POSITIVE_INFINITY, v[3]);
    }
    try (Tensor x = input(1, 2, 3);
        Tensor p = input(2, 3, 2);
        Tensor g = Transforms.grad(t -> t.pow(p).sum(), x)) {
      equal(new float[] {2, 12, 6}, g);
    }
  }

  @Test
  void slicesConcatenationAndStack() {
    try (Tensor x = matrix(new float[] {0, 1, 2, 3, 4, 5}, 2, 3);
        Tensor s = x.slice(1, 0, 3, 2);
        Tensor reverse = x.slice(1, 2, -4, -1);
        Tensor cat = Tensor.concatenate(List.of(x, x), 0);
        Tensor stack = Tensor.stack(List.of(x, x), -1)) {
      equal(new float[] {0, 2, 3, 5}, s);
      equal(new float[] {2, 1, 0, 5, 4, 3}, reverse);
      assertEquals(Shape.of(4, 3), cat.shape());
      assertEquals(Shape.of(2, 3, 2), stack.shape());
      equal(new float[] {0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5}, stack);
      assertThrows(IllegalArgumentException.class, () -> x.slice(0, 0, 2, 0));
    }
    try (Tensor x = input(1, 2, 3);
        Tensor g = Transforms.grad(t -> Tensor.concatenate(List.of(t, t), 0).sum(), x)) {
      equal(new float[] {2, 2, 2}, g);
    }
  }

  record Gather(Tensor values, Tensor indices) {}

  @Test
  void dynamicTakeAndRepeatedGradient() {
    var take = Jit.compileTree((Gather g) -> g.values().take(g.indices(), 0));
    try (Tensor values = input(10, 20, 30);
        Tensor ix = input(2, 0, 2, -1, 99, 1.5f, Float.NaN);
        Tensor result = take.apply(new Gather(values, ix))) {
      equal(new float[] {30, 10, 30, 30, Float.NaN, Float.NaN, Float.NaN}, result);
    }
    try (Tensor values = input(10, 20, 30);
        Tensor ix = input(1, 1, 0, 0, 0, 0, 0);
        Tensor result = take.apply(new Gather(values, ix))) {
      equal(new float[] {20, 20, 10, 10, 10, 10, 10}, result);
    }
    try (Tensor values = input(1, 2, 3);
        Tensor ix = input(2, 0, 2);
        Tensor g = Transforms.grad(t -> t.take(ix, 0).sum(), values)) {
      equal(new float[] {1, 0, 2}, g);
    }
  }

  record Region(Tensor base, Tensor update, Tensor starts) {}

  @Test
  void runtimeSlicesUpdatesAndGradients() {
    var update = Jit.compileTree((Region s) -> s.base().dynamicUpdateSlice(s.update(), s.starts()));
    try (Tensor base = matrix(new float[] {0, 1, 2, 3, 4, 5}, 2, 3);
        Tensor patch = matrix(new float[] {8, 9}, 1, 2);
        Tensor starts = input(1, 1);
        Tensor result = update.apply(new Region(base, patch, starts));
        Tensor slice = base.dynamicSlice(starts, 1, 2);
        Tensor gradient = Transforms.grad(t -> t.dynamicUpdateSlice(patch, starts).sum(), base)) {
      equal(new float[] {0, 1, 2, 3, 8, 9}, result);
      equal(new float[] {4, 5}, slice);
      equal(new float[] {1, 1, 1, 1, 0, 0}, gradient);
    }
    try (Tensor base = matrix(new float[] {0, 1, 2, 3, 4, 5}, 2, 3);
        Tensor patch = matrix(new float[] {8, 9}, 1, 2);
        Tensor starts = input(99, 99);
        Tensor result = update.apply(new Region(base, patch, starts))) {
      equal(new float[] {0, 1, 2, 3, 8, 9}, result);
    }
    try (Tensor x = input(1, 2, 3, 4);
        Tensor starts = input(1);
        Tensor g = Transforms.grad(t -> t.dynamicSlice(starts, 2).sum(), x)) {
      equal(new float[] {0, 1, 1, 0}, g);
    }
  }

  @Test
  void productsWithZerosAndEmptyIdentity() {
    try (Tensor x = matrix(new float[] {2, 0, 4, 0, 0, 3}, 2, 3);
        Tensor p = x.prod(1);
        Tensor gradient = Transforms.grad(t -> t.prod(1).sum(), x);
        Tensor compiled = Jit.compile(t -> t.prod(1)).apply(x)) {
      equal(new float[] {0, 0}, p);
      equal(new float[] {0, 8, 0, 0, 0, 0}, gradient);
      equal(p.toFloatArray(), compiled);
    }
    try (Tensor x = input(-2, 3, 4);
        Tensor p = x.prod();
        Tensor g = Transforms.grad(Tensor::prod, x)) {
      assertEquals(-24, p.item(), 1e-5);
      equal(new float[] {12, -8, -6}, g);
    }
    try (Tensor empty = Tensors.zeros(DEVICE, 2, 0);
        Tensor p = empty.prod(1)) {
      equal(new float[] {1, 1}, p);
    }
  }

  @Test
  void topKAndArgmin() {
    try (Tensor x = input(2, 5, 5, Float.NaN, 1)) {
      var top = x.topK(3);
      try {
        equal(new float[] {Float.NaN, 5, 5}, top.values());
        equal(new float[] {3, 1, 2}, top.indices());
      } finally {
        close(top);
      }
      try (Tensor g = Transforms.grad(t -> t.topK(3).values().sum(), x)) {
        equal(new float[] {0, 1, 1, 1, 0}, g);
      }
    }
    try (Tensor x = input(4, -2, -2, 3);
        Tensor min = x.argmin(0)) {
      assertEquals(1, min.item());
    }
  }

  @Test
  void segmentSumsAndOneHot() {
    try (Tensor data = matrix(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, 4, 2);
        Tensor ids = input(0, 1, 0, -1);
        Tensor sum = data.segmentSum(ids, 3);
        Tensor g = Transforms.grad(t -> t.segmentSum(ids, 3).sum(), data);
        Tensor hot = ids.oneHot(3)) {
      equal(new float[] {6, 8, 3, 4, 0, 0}, sum);
      equal(new float[] {1, 1, 1, 1, 1, 1, 0, 0}, g);
      equal(new float[] {1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0}, hot);
    }
    try (Tensor ids = input(1.5f, Float.NaN, 5);
        Tensor hot = Jit.compile(t -> t.oneHot(3)).apply(ids)) {
      equal(new float[9], hot);
    }
  }

  @Test
  void differencesRollAndPadding() {
    try (Tensor x = input(1, 3, 6, 10);
        Tensor d = x.diff(0);
        Tensor dd = x.diff(2, 0);
        Tensor rolled = x.roll(-1, 0);
        Tensor padded = x.pad(new int[][] {{1, 2}}, 9);
        Tensor grad = Transforms.grad(t -> t.pad(new int[][] {{1, 2}}, 9).sum(), x)) {
      equal(new float[] {2, 3, 4}, d);
      equal(new float[] {1, 1}, dd);
      equal(new float[] {3, 6, 10, 1}, rolled);
      equal(new float[] {9, 1, 3, 6, 10, 9, 9}, padded);
      equal(new float[] {1, 1, 1, 1}, grad);
    }
  }

  @Test
  void constructorsAndGrids() {
    try (Tensor eye = Tensors.eye(DEVICE, 2, 3, 1);
        Tensor logs = Tensors.logspace(DEVICE, 0, 2, 3);
        Tensor geom = Tensors.geomspace(DEVICE, -1, -100, 3);
        Tensor x = input(10, 20);
        Tensor y = input(1, 2, 3)) {
      equal(new float[] {0, 1, 0, 0, 0, 1}, eye);
      equal(new float[] {1, 10, 100}, logs);
      equal(new float[] {-1, -10, -100}, geom);
      var grid = Tensors.meshgrid(List.of(x, y), "xy", false);
      try {
        assertEquals(Shape.of(3, 2), grid.get(0).shape());
        equal(new float[] {10, 20, 10, 20, 10, 20}, grid.get(0));
        equal(new float[] {1, 1, 2, 2, 3, 3}, grid.get(1));
      } finally {
        close(grid);
      }
      var sparse = Tensors.meshgrid(List.of(x, y), "ij", true);
      try {
        assertEquals(Shape.of(2, 1), sparse.get(0).shape());
        assertEquals(Shape.of(1, 3), sparse.get(1).shape());
        equal(new float[] {10, 20}, sparse.get(0));
        equal(new float[] {1, 2, 3}, sparse.get(1));
      } finally {
        close(sparse);
      }
      var ix = Tensors.indices(DEVICE, 2, 3);
      try {
        equal(new float[] {0, 0, 0, 1, 1, 1}, ix.get(0));
        equal(new float[] {0, 1, 2, 0, 1, 2}, ix.get(1));
      } finally {
        close(ix);
      }
      assertThrows(IllegalArgumentException.class, () -> Tensors.geomspace(DEVICE, -1, 1, 3));
    }
  }

  @Test
  void arrayKernelJvpAndEmptyCases() {
    try (Tensor x = input(2, 0, 4);
        Tensor tangent = input(1, 1, 1)) {
      var jvp = Transforms.jvp(Tensor::prod, x, tangent);
      try {
        assertEquals(8, jvp.tangent().item(), 2e-5);
      } finally {
        close(jvp);
      }
    }
    try (Tensor empty = Tensors.zeros(DEVICE, 0);
        Tensor ix = input(0);
        Tensor taken = empty.take(ix, 0);
        Tensor zeroHot = ix.oneHot(0)) {
      equal(new float[] {Float.NaN}, taken);
      assertEquals(0, zeroHot.toFloatArray().length);
      var top = empty.topK(0);
      try {
        assertEquals(0, top.values().toFloatArray().length);
      } finally {
        close(top);
      }
    }
  }

  @Test
  void compiledArrayGradientsAndValidation() {
    var gradient = Jit.compile(Transforms.grad(t -> t.prod(0)));
    try (Tensor x = input(2, 3, 4);
        Tensor y = input(1, 5, 2);
        Tensor gx = gradient.apply(x);
        Tensor gy = gradient.apply(y)) {
      equal(new float[] {12, 8, 6}, gx);
      equal(new float[] {10, 2, 5}, gy);
      try (Tensor ids = input(1);
          Tensor bad = input(Float.NaN);
          Tensor patch = input(9);
          Tensor ignored = x.dynamicUpdateSlice(patch, bad);
          Tensor nan = x.dynamicSlice(bad, 1)) {
        equal(x.toFloatArray(), ignored);
        equal(new float[] {Float.NaN}, nan);
        assertThrows(IllegalArgumentException.class, () -> x.dynamicSlice(ids, 5));
        assertThrows(IllegalArgumentException.class, () -> x.topK(4));
        assertThrows(IllegalArgumentException.class, () -> x.oneHot(-1));
        assertThrows(IllegalArgumentException.class, () -> x.pad(new int[][] {{-1, 0}}));
      }
    }
    try (Tensor x = input(0, 1);
        Tensor y = input(1, 1);
        Tensor xor = x.xor(y);
        Tensor unequal = x.ne(y);
        Tensor le = x.le(y)) {
      equal(new float[] {1, 0}, xor);
      equal(new float[] {1, 0}, unequal);
      equal(new float[] {1, 1}, le);
    }
  }

  @Test
  void selectionInsideScanAndVmap() {
    try (Tensor initial = input(-1, 2);
        Tensor samples = matrix(new float[] {2, -3, -4, 1}, 2, 2)) {
      var scan =
          ControlFlow.scan(
              (carry, sample) -> {
                Tensor sum = carry.add(sample);
                Tensor next = Tensor.where(sum.gt(0), sum, sum.fullLike(0));
                return new ControlFlow.ScanStep(next, next);
              },
              initial,
              samples);
      try {
        equal(new float[] {0, 1}, scan.carry());
      } finally {
        close(scan);
      }
    }
    try (Tensor x = matrix(new float[] {1, -2, 3, -4, 5, -6}, 2, 3);
        Tensor y = Vmap.apply(t -> Tensor.where(t.gt(0), t, t.fullLike(0)), x)) {
      equal(new float[] {1, 0, 3, 0, 5, 0}, y);
    }
  }

  @Test
  void mappedArrayGradientsAndPowerDomains() {
    try (Tensor x = matrix(new float[] {1, -2, 3, -4, 5, -6}, 2, 3);
        Tensor mapped = Vmap.apply(t -> Tensor.where(t.gt(0), t, t.fullLike(0)), x);
        Tensor gradient =
            Transforms.grad(
                t -> Vmap.apply(v -> Tensor.where(v.gt(0), v, v.fullLike(0)), t).sum(), x)) {
      equal(new float[] {1, 0, 3, 0, 5, 0}, mapped);
      equal(new float[] {1, 0, 1, 0, 1, 0}, gradient);
    }
    try (Tensor x = input(-2, -2, 0, Float.NaN);
        Tensor exponent = input(Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, Float.NaN, 0);
        Tensor p = x.pow(exponent)) {
      equal(new float[] {Float.POSITIVE_INFINITY, 0, Float.NaN, 1}, p);
    }
    try (Tensor x = input(1, 2, 3)) {
      assertThrows(
          UnsupportedOperationException.class,
          () -> Transforms.grad(t -> Transforms.grad(Tensor::prod, t).sum(), x));
    }
  }

  @Test
  void powerInfiniteBasesAndSignedZero() {
    try (Tensor x = input(Float.NEGATIVE_INFINITY, Float.NEGATIVE_INFINITY, -0.0f, -0.0f);
        Tensor exponent = input(.5f, -3, 3, -3);
        Tensor result = x.pow(exponent)) {
      float[] v = result.toFloatArray();
      assertEquals(Float.POSITIVE_INFINITY, v[0]);
      assertEquals(Float.floatToRawIntBits(-0.0f), Float.floatToRawIntBits(v[1]));
      assertEquals(Float.floatToRawIntBits(-0.0f), Float.floatToRawIntBits(v[2]));
      assertEquals(Float.NEGATIVE_INFINITY, v[3]);
    }
  }
}
