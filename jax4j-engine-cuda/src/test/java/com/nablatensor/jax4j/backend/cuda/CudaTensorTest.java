package com.nablatensor.jax4j.backend.cuda;

import com.nablatensor.jax4j.tensor.*;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;
import static org.junit.jupiter.api.Assertions.*;

@EnabledIfSystemProperty(named = "jax4j.cuda.tests", matches = "true")
class CudaTensorTest {
  @BeforeAll static void requireCuda() {
    assertTrue(new CudaBackend().isAvailable(), "CUDA explicitly required");
  }

  private Tensor array(float[] values, int... dims) {
    return Tensors.arrayOn(values, Shape.of(dims), Device.cuda());
  }

  @Test void uploadElementwiseAndReduction() {
    try (Tensor x = array(new float[] {1, 2, 3, 4}, 4);
         Tensor y = x.mul(x).add(3); Tensor sum = y.sum()) {
      assertArrayEquals(new float[] {4, 7, 12, 19}, y.toFloatArray(), 1e-5f);
      assertEquals(42, sum.item(), 1e-5f);
    }
  }

  @Test void matrixMultiplication() {
    try (Tensor a = array(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
         Tensor b = array(new float[] {7, 8, 9, 10, 11, 12}, 3, 2);
         Tensor c = a.matmul(b)) {
      assertArrayEquals(new float[] {58, 64, 139, 154}, c.toFloatArray(), 1e-5f);
    }
  }

  @Test void deviceFill() {
    try (Tensor x = array(new float[] {1, 2, 3}, 3);
         Tensor filled = x.fullLike(-2.5f)) {
      assertArrayEquals(new float[] {-2.5f, -2.5f, -2.5f}, filled.toFloatArray(), 0f);
    }
  }

  @Test void compiledReplayUsesNewInputs() {
    var plan = Jit.compile(x -> x.mul(x).add(3));
    try (Tensor x = array(new float[] {1, 2}, 2);
         Tensor y = array(new float[] {3, 4}, 2);
         Tensor first = plan.apply(x); Tensor second = plan.apply(y)) {
      assertArrayEquals(new float[] {4, 7}, first.toFloatArray(), 1e-5f);
      assertArrayEquals(new float[] {12, 19}, second.toFloatArray(), 1e-5f);
      assertEquals(1, plan.cachedPlanCount());
    }
  }

  @Test void compiledGradient() {
    try (Tensor x = array(new float[] {1, 2, 3}, 3);
         Tensor grad = Jit.apply(t -> Transforms.grad(v -> v.mul(v).sum(), t), x)) {
      assertArrayEquals(new float[] {2, 4, 6}, grad.toFloatArray(), 1e-5f);
    }
  }
}
