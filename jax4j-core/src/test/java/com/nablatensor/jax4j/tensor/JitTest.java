package com.nablatensor.jax4j.tensor;

import com.nablatensor.jax4j.autodiff.AutoDiff;
import com.nablatensor.jax4j.backend.cpu.CpuBackend;
import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;

class JitTest {
  private static Tensor array(float... values) {
    var backend=new CpuBackend();
    return new Tensor(backend,backend.upload(values,Shape.of(values.length),DType.F32,Device.cpu()));
  }
  @Test void reusablePlanUsesNewValuesWithoutRetracing() {
    AtomicInteger traces=new AtomicInteger();
    var plan=Jit.compile(x->{traces.incrementAndGet(); return x.mul(x).add(3).tanh();});
    // Same backend instance is part of the signature.
    var backend=new CpuBackend();
    try(Tensor x=new Tensor(backend,backend.upload(new float[]{1,2},Shape.of(2),DType.F32,Device.cpu()));
        Tensor y=new Tensor(backend,backend.upload(new float[]{3,4},Shape.of(2),DType.F32,Device.cpu()));
        Tensor a=plan.apply(x); Tensor b=plan.apply(y)) {
      assertArrayEquals(new float[]{(float)Math.tanh(12),(float)Math.tanh(19)},b.toFloatArray(),1e-6f);
      assertEquals(1,traces.get()); assertEquals(1,plan.cachedPlanCount());
    }
  }
  @Test void capturedInputsAndAxisBoundaries() {
    try(Tensor bias=array(10,20);Tensor x=array(1,2);
        Tensor y=Jit.apply(t->t.mul(t).add(bias).reshape(1,2).sum(0).add(1),x)) {
      assertArrayEquals(new float[]{12,25},y.toFloatArray(),1e-6f);
    }
  }
  @Test void compiledFunctionsRejectHostReads() {
    var compiled=Jit.compile(x->{x.toFloatArray();return x.add(1);});
    try(Tensor x=array(1)) {
      assertThrows(IllegalStateException.class,()->compiled.apply(x));
      assertEquals(0,compiled.cachedPlanCount());
    }
  }
  @Test void gradientComposesWithFusion() {
    try(Tensor x=array(1,2,3);Tensor grad=Jit.apply(t->AutoDiff.grad(v->v.get(0).mul(v.get(0)).sum(),List.of(t)).get(0),x)) {
      assertArrayEquals(new float[]{2,4,6},grad.toFloatArray(),1e-6f);
    }
  }
  @Test void capturedConstantDoesNotBecomeADynamicInputWhenItAliasesFirstArgument() {
    var backend=new CpuBackend();
    try(Tensor constant=new Tensor(backend,backend.upload(new float[]{2},Shape.of(1),DType.F32,Device.cpu()));
        Tensor next=new Tensor(backend,backend.upload(new float[]{5},Shape.of(1),DType.F32,Device.cpu()))) {
      var compiled=Jit.compile(x->x.add(constant));
      try(Tensor first=compiled.apply(constant);Tensor second=compiled.apply(next)) {
        assertEquals(4,first.item(),1e-6f);assertEquals(7,second.item(),1e-6f);
      }
    }
  }
}
