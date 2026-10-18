package com.jax4j.backend.wgpu;

import static org.junit.jupiter.api.Assertions.*;

import com.jax4j.nn.FunctionalAdam;
import com.jax4j.tensor.*;
import com.jax4j.tensor.tree.TreeUtil;
import java.util.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

@EnabledIfSystemProperty(named = "jax4j.wgpu.tests", matches = "true")
class WgpuScaleTest {
  private Tensor array(float[] data, int... shape) {
    return Tensors.arrayOn(data, Shape.of(shape), Device.wgpu());
  }

  @Test
  void multiWorkgroupReductionsAreDeterministic() {
    for (int size : new int[] {257, 65536, 262144}) {
      float[] data = new float[size];
      for (int i = 0; i < size; i++) data[i] = i % 7 - 3;
      double expected = 0;
      for (float value : data) expected += value;
      try (Tensor x = array(data, size);
          Tensor zeros = x.fullLike(0)) {
        for (int repeat = 0; repeat < 5; repeat++)
          try (Tensor sum = x.sum();
              Tensor zero = zeros.sum();
              Tensor max = x.max()) {
            assertEquals((float) expected, sum.item(), 1e-5f);
            assertEquals(0, zero.item(), 0);
            assertEquals(3, max.item(), 0);
          }
      }
    }
  }

  @Test
  void tiledMatmulHasCorrectTailAndCrossWaveDependencies() {
    int m = 65, k = 67, n = 70;
    float[] a = new float[m * k], b = new float[k * n], expected = new float[m * n];
    for (int i = 0; i < a.length; i++) a[i] = (i * 17 % 31 - 15) * .01f;
    for (int i = 0; i < b.length; i++) b[i] = (i * 11 % 29 - 14) * .02f;
    for (int row = 0; row < m; row++)
      for (int column = 0; column < n; column++) {
        float sum = 0;
        for (int j = 0; j < k; j++) sum += a[row * k + j] * b[j * n + column];
        expected[row * n + column] = sum;
      }
    try (Tensor x = array(a, m, k);
        Tensor y = array(b, k, n)) {
      for (int repeat = 0; repeat < 5; repeat++)
        try (Tensor result = x.matmul(y)) {
          assertArrayEquals(expected, result.toFloatArray(), 1e-5f);
        }
    }
  }

  @Test
  void largeFunctionalTrainingHasFiniteBiasCorrection() {
    float[] data = new float[262144];
    Arrays.fill(data, .25f);
    var optimizer = new FunctionalAdam(.01);
    try (Tensor x = array(data, data.length)) {
      var state = optimizer.initialize(List.of(x));
      try {
        assertEquals(0, state.step().item(), 0);
        try (Tensor gradient = Transforms.grad(t -> t.mul(t).sum(), x)) {
          var result = optimizer.update(List.of(x), List.of(gradient), state);
          try {
            assertEquals(1, result.state().step().item(), 0);
            for (float value : result.parameters().getFirst().toFloatArray())
              assertEquals(.24f, value, 5e-6f);
          } finally {
            TreeUtil.closeAll(result);
          }
        }
      } finally {
        TreeUtil.closeAll(state);
      }
    }
  }
}
