package com.nablatensor.jax4j.tensor;
import com.nablatensor.jax4j.tensor.*;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;
class StableMathTest {
  private static final Device DEVICE=Device.cpu();

  private Tensor input(float... values) { return Tensors.arrayOn(values, Shape.of(values.length), DEVICE); }
  private void close(Object tree) { com.nablatensor.jax4j.tensor.tree.TreeUtil.closeAll(tree); }

  @Test void stableNearZeroEagerAndFused() {
    float[] values = {-.5f, -.499f, -.125f, -.1249f, .1249f, .125f, -0.1f, -1e-7f, -1e-12f, 0, 1e-12f, 1e-7f, .1f, .49f, .5f, 1};
    var logarithm = Jit.compile(Tensor::log1p);
    var exponential = Jit.compile(Tensor::expm1);
    try (Tensor x = input(values); Tensor l = x.log1p(); Tensor e = x.expm1();
         Tensor jl = logarithm.apply(x); Tensor je = exponential.apply(x)) {
      float[] lv=l.toFloatArray(), ev=e.toFloatArray(), jlv=jl.toFloatArray(), jev=je.toFloatArray();
      for (int i=0;i<values.length;i++) {
        double a = Math.log1p(values[i]), b = Math.expm1(values[i]);
        double lt = Math.max(1e-18, Math.abs(a)*2e-6), et = Math.max(1e-18,Math.abs(b)*2e-6);
        assertEquals(a,lv[i],lt); assertEquals(a,jlv[i],lt);
        assertEquals(b,ev[i],et); assertEquals(b,jev[i],et);
      }
    }
  }
  @Test void unaryDomainsAndExtremeSoftplus() {
    try (Tensor x=input(-1000,-100,-2,0,2,100,1000,Float.NEGATIVE_INFINITY,Float.POSITIVE_INFINITY,Float.NaN);
         Tensor y=x.softplus(); Tensor jy=Jit.compile(Tensor::softplus).apply(x); Tensor squares=x.square()) {
      float[] out=y.toFloatArray(), compiled=jy.toFloatArray(), sq=squares.toFloatArray();
      for(int i=0;i<7;i++) {
        double v=x.toFloatArray()[i];
        double expected=Math.max(v,0)+Math.log1p(Math.exp(-Math.abs(v)));
        assertEquals(expected,out[i],Math.max(1e-40,Math.abs(expected)*3e-6));
        assertEquals(expected,compiled[i],Math.max(1e-40,Math.abs(expected)*3e-6));
      }
      assertEquals(0,out[7]); assertEquals(Float.POSITIVE_INFINITY,out[8]); assertTrue(Float.isNaN(out[9]));
      assertEquals(1000000,sq[0]);
    }
    try(Tensor x=input(-2,-1,Float.POSITIVE_INFINITY,Float.NaN);Tensor y=x.log1p()) {
      float[] v=y.toFloatArray(); assertTrue(Float.isNaN(v[0]));
      assertEquals(Float.NEGATIVE_INFINITY,v[1]); assertEquals(Float.POSITIVE_INFINITY,v[2]); assertTrue(Float.isNaN(v[3]));
    }
  }
  @Test void axisStatisticsAndCorrection() {
    try(Tensor x=Tensors.arrayOn(new float[]{1,2,3,4,5,6},Shape.of(2,3),DEVICE);
        Tensor m=x.mean(-1,true);Tensor v=x.var(1);Tensor sample=x.var(1,1,true);
        Tensor sd=x.std(0);Tensor all=x.var();Tensor allStd=x.std();
        Tensor invalid=x.var(1,3,false);Tensor jit=Jit.compile(t->t.var(-1,1,true)).apply(x)) {
      assertEquals(Shape.of(2,1),m.shape());assertArrayEquals(new float[]{2,5},m.toFloatArray(),1e-6f);
      assertArrayEquals(new float[]{2f/3,2f/3},v.toFloatArray(),1e-6f);
      assertArrayEquals(new float[]{1,1},sample.toFloatArray(),1e-6f);
      assertArrayEquals(new float[]{1.5f,1.5f,1.5f},sd.toFloatArray(),1e-6f);
      assertEquals(35.0/12,all.item(),1e-6); assertEquals(Math.sqrt(35.0/12),allStd.item(),1e-6);
      assertTrue(Float.isNaN(invalid.toFloatArray()[0]));assertArrayEquals(sample.toFloatArray(),jit.toFloatArray(),1e-6f);
      assertThrows(IllegalArgumentException.class,()->x.var(1,-1,false));
      assertThrows(IllegalArgumentException.class,()->x.mean(4));
    }
    try(Tensor x=Tensors.full(DEVICE,10000,2,3);Tensor v=x.var(1);Tensor sd=x.std(1)) {
      assertArrayEquals(new float[]{0,0},v.toFloatArray(),2e-6f);assertArrayEquals(new float[]{0,0},sd.toFloatArray(),.002f);
    }
  }
  @Test void probabilityReductionsAndNonfiniteInputs() {
    try(Tensor x=Tensors.arrayOn(new float[]{1000,1001,1002,-1000,-1001,-1002},Shape.of(2,3),DEVICE);
        Tensor p=x.softmax(-1);Tensor logp=x.logSoftmax(1);Tensor lse=x.logsumexp(1);
        Tensor keep=x.logsumexp(1,true);Tensor all=x.logsumexp();
        Tensor j=Jit.compile(t->t.logSoftmax(-1)).apply(x);
        Tensor shifted=x.add(4096);Tensor shiftedP=shifted.softmax(1)) {
      double[] a={1,2,3}; double denom=Math.exp(-2)+Math.exp(-1)+1;
      float[] probs=p.toFloatArray(),logs=logp.toFloatArray();
      assertEquals(1,probs[0]+probs[1]+probs[2],2e-6);
      assertEquals(1/denom,probs[2],2e-6);assertEquals(-2-Math.log(denom),logs[0],2e-6);
      assertEquals(1002+Math.log(denom),lse.toFloatArray()[0],2e-4);
      assertEquals(1002+Math.log(denom),all.item(),2e-4);
      assertEquals(Shape.of(2,1),keep.shape());
      assertArrayEquals(probs,shiftedP.toFloatArray(),2e-6f);assertArrayEquals(logs,j.toFloatArray(),2e-6f);
    }
    for(float[] values:new float[][]{{Float.NEGATIVE_INFINITY,Float.NEGATIVE_INFINITY},{Float.POSITIVE_INFINITY,0},{Float.NaN,0}}) {
      try(Tensor x=input(values);Tensor l=x.logsumexp();Tensor axis=x.logsumexp(0);
          Tensor p=x.softmax(0);Tensor lp=x.logSoftmax(0);Tensor j=Jit.compile(t->t.logsumexp(0)).apply(x)) {
        float expected=values[0];
        if(Float.isNaN(expected)) {assertTrue(Float.isNaN(l.item()));assertTrue(Float.isNaN(axis.item()));}
        else {assertEquals(expected,l.item());assertEquals(expected,axis.item());assertEquals(expected,j.item());}
        assertTrue(Float.isNaN(p.toFloatArray()[0]));assertTrue(Float.isNaN(lp.toFloatArray()[0]));
      }
    }
  }
  @Test void emptyStatistics() {
    try(Tensor x=Tensors.zeros(DEVICE,0,3);Tensor m=x.mean(0);Tensor v=x.var(0);
        Tensor sd=x.std(0);Tensor l=x.logsumexp(0);Tensor all=x.logsumexp()) {
      for(float f:m.toFloatArray()) assertTrue(Float.isNaN(f));
      for(float f:v.toFloatArray()) assertTrue(Float.isNaN(f));
      for(float f:sd.toFloatArray()) assertTrue(Float.isNaN(f));
      for(float f:l.toFloatArray()) assertEquals(Float.NEGATIVE_INFINITY,f);
      assertEquals(Float.NEGATIVE_INFINITY,all.item());
    }
  }
  @Test void analyticGradientsAndJvp() {
    java.util.List<java.util.function.UnaryOperator<Tensor>> functions=java.util.List.of(
      t->t.log1p().sum(),t->t.expm1().sum(),t->t.softplus().sum(),t->t.square().sum(),
      Tensor::logsumexp,t->t.var(0).sum());
    float[] points={-.2f,.3f,.7f};
    for(int f=0;f<functions.size();f++) {
      var fun=functions.get(f);
      try(Tensor x=input(points);Tensor tangent=Tensors.ones(DEVICE,3);
          Tensor gradient=Transforms.grad(fun,x);Tensor jg=Jit.compile(Transforms.grad(fun)).apply(x)) {
        var jvp=Transforms.jvp(fun,x,tangent);
        try {
          float[] g=gradient.toFloatArray();double sum=0;
          for(int i=0;i<3;i++) {
            double expected=switch(f) {
              case 0->1/(1.0+points[i]);case 1->Math.exp(points[i]);
              case 2->1/(1+Math.exp(-points[i]));case 3->2*points[i];
              case 4->Math.exp(points[i])/(Math.exp(points[0])+Math.exp(points[1])+Math.exp(points[2]));
              default->2*(points[i]-(points[0]+points[1]+points[2])/3.0)/3;
            };
            assertEquals(expected,g[i],2e-5);sum+=expected;
          }
          assertEquals(sum,jvp.tangent().item(),3e-5);assertArrayEquals(g,jg.toFloatArray(),2e-5f);
        } finally { close(jvp); }
      }
    }
  }
  @Test void tiedLogitsAndFiniteDifference() {
    java.util.function.UnaryOperator<Tensor> fun=t->t.logSoftmax(0).square().sum();
    try(Tensor x=input(1,1,1);Tensor gradient=Transforms.grad(Tensor::logsumexp,x)) {
      assertArrayEquals(new float[]{1f/3,1f/3,1f/3},gradient.toFloatArray(),1e-6f);
    }
    try(Tensor x=input(-.2f,.3f,.7f);Tensor g=Transforms.grad(fun,x)) {
      float[] expected=g.toFloatArray();
      for(int i=0;i<3;i++) {
        float[] plus={-.2f,.3f,.7f},minus=plus.clone();plus[i]+=.001f;minus[i]-=.001f;
        try(Tensor p=input(plus);Tensor m=input(minus);Tensor fp=Jit.compile(fun).apply(p);Tensor fm=Jit.compile(fun).apply(m)) {
          assertEquals((fp.item()-fm.item())/.002,expected[i],.002);
        }
      }
    }
  }
  @Test void replayAndStructuredRegion() {
    var plan=Jit.compile(t->t.log1p().expm1().softplus());
    for(float value:new float[]{.01f,.2f,.7f}) {
      try(Tensor x=input(value);Tensor y=plan.apply(x)) {
        assertEquals(Math.log1p(Math.exp(value)),y.item(),2e-6);
      }
    }
    assertEquals(1,plan.cachedPlanCount());
    try(Tensor initial=input(.1f);Tensor samples=input(.1f,.2f,.3f)) {
      var scan=ControlFlow.scan((carry, sample)-> {
        Tensor next=carry.add(sample).log1p().expm1();
        return new ControlFlow.ScanStep(next,next);
      },initial,samples);
      try {assertEquals(.7,scan.carry().item(),2e-5);}
      finally {close(scan);}
    }
  }
  @Test void signedZeroAndSoftmaxGradient() {
    try(Tensor x=input(-0.0f);Tensor l=x.log1p();Tensor e=x.expm1()) {
      assertEquals(Float.floatToRawIntBits(-0.0f),Float.floatToRawIntBits(l.item()));
      assertEquals(Float.floatToRawIntBits(-0.0f),Float.floatToRawIntBits(e.item()));
    }
    try(Tensor x=input(-.2f,.3f,.7f);Tensor w=input(1,2,4)) {
      java.util.function.UnaryOperator<Tensor> loss=t->t.softmax(0).mul(w).sum();
      try(Tensor g=Transforms.grad(loss,x);Tensor p=x.softmax(0)) {
        float[] probabilities=p.toFloatArray();
        double weighted=probabilities[0]+2*probabilities[1]+4*probabilities[2];
        float[] gradient=g.toFloatArray();
        for(int i=0;i<3;i++) assertEquals(probabilities[i]*((i==2?4:i+1)-weighted),gradient[i],2e-5);
      }
    }
  }
}
