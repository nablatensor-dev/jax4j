package com.nablatensor.jax4j.tensor;
import com.nablatensor.jax4j.tensor.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
class ArrayCreationTest {
  private static final Device DEVICE = Device.cpu();

  @Test void sequencesAndConstants() {
    try (Tensor range = Tensors.arange(DEVICE, 0, 5, 2);
         Tensor descending = Tensors.arange(DEVICE, 3, -1, -1);
         Tensor line = Tensors.linspace(DEVICE, 0, 1, 5);
         Tensor excluded = Tensors.linspace(DEVICE, 0, 1, 4, false);
         Tensor ones = Tensors.ones(DEVICE, 2, 3);
         Tensor full = Tensors.full(DEVICE, 7, 2, 3);
         Tensor zero = full.zerosLike(); Tensor one = full.onesLike()) {
      assertArrayEquals(new float[]{0,2,4}, range.toFloatArray(), 0);
      assertArrayEquals(new float[]{3,2,1,0}, descending.toFloatArray(), 0);
      assertArrayEquals(new float[]{0,.25f,.5f,.75f,1}, line.toFloatArray(), 0);
      assertArrayEquals(new float[]{0,.25f,.5f,.75f}, excluded.toFloatArray(), 0);
      assertArrayEquals(new float[]{1,1,1,1,1,1}, ones.toFloatArray(), 0);
      assertArrayEquals(new float[]{7,7,7,7,7,7}, full.toFloatArray(), 0);
      assertArrayEquals(new float[6], zero.toFloatArray(), 0);
      assertArrayEquals(ones.toFloatArray(), one.toFloatArray(), 0);
      for (Tensor t : new Tensor[]{range, descending, line, excluded, ones, full, zero, one}) {
        assertEquals(DEVICE, t.device()); assertEquals(DType.F32, t.dtype());
      }
      assertEquals(full.shape(), zero.shape());
    }
  }
  @Test void boundariesAndValidation() {
    try (Tensor empty = Tensors.arange(DEVICE, 5, 0, 1);
         Tensor none = Tensors.linspace(DEVICE, 2, 3, 0);
         Tensor single = Tensors.linspace(DEVICE, 2, 3, 1);
         Tensor scalar = Tensors.full(DEVICE, 4)) {
      assertEquals(0, empty.toFloatArray().length);
      assertEquals(0, none.toFloatArray().length);
      assertArrayEquals(new float[]{2}, single.toFloatArray(), 0);
      assertEquals(4, scalar.item());
    }
    assertThrows(IllegalArgumentException.class, () -> Tensors.arange(DEVICE, 0, 2, 0));
    assertThrows(IllegalArgumentException.class, () -> Tensors.arange(DEVICE, Double.NaN));
    assertThrows(IllegalArgumentException.class, () -> Tensors.linspace(DEVICE, 0, 1, -1));
    assertThrows(IllegalArgumentException.class, () -> Tensors.arange(DEVICE, 0, 1e20, 1));
  }
  @Test void likeConstantsInsideJit() {
    var plan = Jit.compile(x -> x.onesLike());
    try (Tensor input = Tensors.arange(DEVICE, 3); Tensor result = plan.apply(input)) {
      assertArrayEquals(new float[]{1,1,1}, result.toFloatArray(), 0);
      assertEquals(DEVICE, result.device());
    }
  }
}
