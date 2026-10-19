package com.nablatensor.jax4j.backend.vulkan;

import com.nablatensor.jax4j.tensor.*;

/** Synchronized eager/fused chain benchmark; no accelerator auto-selection. */
public final class VulkanBenchmark {
  private VulkanBenchmark() {}
  private static Tensor chain(Tensor x) {
    Tensor y=x;
    for(int i=0;i<12;i++) y=y.mul(1.0001).add(.0001).tanh();
    return y;
  }
  private static Tensor eager(Tensor x) {
    Tensor y=x;
    for(int i=0;i<12;i++) {
      Tensor a=y.mul(1.0001),b=a.add(.0001),c=b.tanh();
      a.close();b.close();if(y!=x)y.close();y=c;
    }
    return y;
  }
  public static void main(String[] args) {
    int size=Integer.getInteger("size",262144),iterations=Integer.getInteger("iterations",20);
    VulkanBackend backend=new VulkanBackend();
    if(!backend.isAvailable())throw new IllegalStateException("Vulkan required");
    var compiled=Jit.compile(VulkanBenchmark::chain);
    float[] data=new float[size];java.util.Arrays.fill(data,.5f);
    try(Tensor input=Tensors.arrayOn(data,Shape.of(size),Device.vulkan())) {
      long start=System.nanoTime();
      try(Tensor first=compiled.apply(input)){first.blockUntilReady();}
      double cold=(System.nanoTime()-start)/1e6;
      for(int i=0;i<5;i++){try(Tensor y=compiled.apply(input)){y.blockUntilReady();}try(Tensor y=eager(input)){y.blockUntilReady();}}
      var before=backend.diagnostics();start=System.nanoTime();
      for(int i=0;i<iterations;i++)try(Tensor y=compiled.apply(input)){y.blockUntilReady();}
      double fused=(System.nanoTime()-start)/1e6/iterations;
      var fusion=backend.diagnostics().minus(before);
      before=backend.diagnostics();start=System.nanoTime();
      for(int i=0;i<iterations;i++)try(Tensor y=eager(input)){y.blockUntilReady();}
      double eager=(System.nanoTime()-start)/1e6/iterations;
      var baseline=backend.diagnostics().minus(before);
      try(Tensor y=compiled.apply(input)){
        float ref=.5f;for(int i=0;i<12;i++)ref=(float)Math.tanh((float)(ref*1.0001)+.0001);
        float[] actual=y.toFloatArray();for(float value:actual)if(Math.abs(value-ref)>3e-5f)throw new AssertionError("numerical mismatch");
      }
      System.out.printf(java.util.Locale.ROOT,"""
          {"device":"%s","dtype":"F32","elements":%d,"iterations":%d,
           "chainOperations":36,"coldCaptureAndCompileMs":%.6f,
           "fusedMs":%.6f,"eagerMs":%.6f,
           "fusedDispatches":%d,"eagerDispatches":%d,
           "fusedAllocations":%d,"eagerAllocations":%d,
           "fusedHostToDeviceBytes":%d,"fusedDeviceToHostBytes":%d,
           "peakDeviceBytes":%d,"numericalTolerance":0.00003}
          """,backend.deviceName().replace("\"","'"),size,iterations,cold,fused,eager,
          fusion.dispatches(),baseline.dispatches(),fusion.allocations(),baseline.allocations(),
          fusion.hostToDeviceBytes(),fusion.deviceToHostBytes(),backend.diagnostics().peakDeviceBytes());
    }
  }
}
