package com.jax4j.tensor;

import com.jax4j.autodiff.AutoDiff;
import org.junit.jupiter.api.Test;
import java.util.List;
import java.util.Properties;
import java.util.concurrent.atomic.AtomicInteger;
import static org.junit.jupiter.api.Assertions.*;

class JaxParityTest {
  private static float[] values(Properties p,String key) {
    String[] parts=p.getProperty(key).split(",");float[] x=new float[parts.length];
    for(int i=0;i<x.length;i++)x[i]=Float.parseFloat(parts[i]);return x;
  }
  @Test void compiledValueAndGradMatchesPinnedJaxAndNewParameters() throws Exception {
    Properties fixture=new Properties();
    try(var stream=getClass().getResourceAsStream("/jax-parity.properties")){fixture.load(stream);}
    assertEquals("0.11.2",fixture.getProperty("jax.version"));
    AtomicInteger captures=new AtomicInteger();
    var compiled=Jit.<List<Tensor>,AutoDiff.ValueAndGrad>compileTree(parameters->{
      captures.incrementAndGet();
      return AutoDiff.valueAndGrad(vars->{var y=vars.get(0).scaled(1.5).tanh();return y.mul(y).sum();},parameters);
    });
    try(Tensor input=Tensors.arrayOn(values(fixture,"input"),Shape.of(4),Device.cpu());
        Tensor second=Tensors.arrayOn(new float[]{0,0,0,0},Shape.of(4),Device.cpu())) {
      var first=compiled.apply(List.of(input));
      try(Tensor loss=first.value();Tensor gradient=first.grads().get(0)) {
        assertArrayEquals(values(fixture,"value"),loss.toFloatArray(),2e-6f);
        assertArrayEquals(values(fixture,"gradient"),gradient.toFloatArray(),2e-6f);
      }
      var next=compiled.apply(List.of(second));
      try(Tensor loss=next.value();Tensor gradient=next.grads().get(0)) {
        assertEquals(0,loss.toFloatArray()[0],1e-6f);
        assertArrayEquals(new float[4],gradient.toFloatArray(),1e-6f);
      }
      assertEquals(1,captures.get());
    }
  }
}
