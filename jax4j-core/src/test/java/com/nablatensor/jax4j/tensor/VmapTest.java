package com.nablatensor.jax4j.tensor;

import com.nablatensor.jax4j.autodiff.AutoDiff;
import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;

class VmapTest {
  @Test void batchRulesPreserveSampleReductionsAndGradients() {
    AtomicInteger calls=new AtomicInteger();
    try(Tensor input=Tensors.arrayOn(new float[]{1,2,3,4,5,6},Shape.of(2,3),Device.cpu());
        Tensor output=Jit.apply(Vmap.of(x->{calls.incrementAndGet();return x.mul(x).sum();}),input);
        Tensor gradients=Jit.apply(Vmap.of(x->AutoDiff.grad(v->v.get(0).mul(v.get(0)).sum(),List.of(x)).get(0)),input)) {
      assertEquals(1,calls.get());
      assertEquals(Shape.of(2,1),output.shape());
      assertArrayEquals(new float[]{14,77},output.toFloatArray(),1e-6f);
      assertArrayEquals(new float[]{2,4,6,8,10,12},gradients.toFloatArray(),1e-6f);
    }
  }
}
