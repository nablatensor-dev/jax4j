package com.jax4j.backend.wgpu;

import static org.junit.jupiter.api.Assertions.*;

import com.jax4j.tensor.*;
import java.util.*;
import java.util.concurrent.*;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

@EnabledIfSystemProperty(named = "jax4j.wgpu.tests", matches = "true")
class WgpuExecutionTest {
  private Tensor array(float[] x, int... dims) {
    return Tensors.arrayOn(x, Shape.of(dims), Device.wgpu());
  }

  @Test
  void compiledScanDispatchCountIsIndependentOfLength() {
    var backend = new WgpuBackend();
    for (int length : new int[] {4, 64}) {
      var plan =
          Jit.compileTree(
              (List<Tensor> args) ->
                  ControlFlow.scan(
                      (c, x) -> {
                        Tensor next = c.add(x);
                        return new ControlFlow.ScanStep(next, next);
                      },
                      args.get(0),
                      args.get(1)));
      float[] values = new float[length];
      Arrays.fill(values, 1);
      try (Tensor c = array(new float[] {0}, 1);
          Tensor xs = array(values, length, 1)) {
        var warm = plan.apply(List.of(c, xs));
        warm.carry().blockUntilReady();
        warm.carry().close();
        warm.outputs().close();
        backend.sync();
        var before = backend.diagnostics();
        var result = plan.apply(List.of(c, xs));
        try (Tensor carry = result.carry();
            Tensor ys = result.outputs()) {
          ys.blockUntilReady();
          var delta = backend.diagnostics().minus(before);
          assertEquals(3, delta.dispatches());
          assertEquals(0, delta.compilations());
          assertEquals(0, delta.hostToDeviceBytes());
          assertEquals(0, delta.deviceToHostBytes());
          assertEquals(length, carry.item(), 0);
        }
      }
    }
  }

  @Test
  void derivativeAndMappedPlansReleaseTemporaryAllocations() {
    var backend = new WgpuBackend();
    backend.sync();
    long baseline = backend.diagnostics().liveDeviceBytes();
    var plan = Jit.compile(Transforms.grad(x -> x.exp().mul(x).tanh().sum()));
    try (Tensor x = array(new float[] {.1f, .2f, .3f, .4f}, 2, 2)) {
      for (int i = 0; i < 10; i++)
        try (Tensor g = plan.apply(x)) {
          g.blockUntilReady();
        }
    }
    backend.sync();
    assertEquals(baseline, backend.diagnostics().liveDeviceBytes());
  }

  @Test
  void concurrentCallsAndEarlyReleasePreserveResults() throws Exception {
    var backend = new WgpuBackend();
    backend.sync();
    long baseline = backend.diagnostics().liveDeviceBytes();
    var plan = Jit.compile(x -> x.mul(x).add(1));
    try (var executor = Executors.newFixedThreadPool(4)) {
      List<Future<?>> tasks = new ArrayList<>();
      for (int worker = 0; worker < 4; worker++) {
        int id = worker;
        tasks.add(
            executor.submit(
                () -> {
                  for (int iteration = 0; iteration < 12; iteration++) {
                    float value = id + iteration;
                    Tensor x = array(new float[] {value, value + 1}, 2);
                    Tensor output = plan.apply(x);
                    x.close();
                    try (output) {
                      output.blockUntilReady();
                      assertArrayEquals(
                          new float[] {value * value + 1, (value + 1) * (value + 1) + 1},
                          output.toFloatArray(),
                          1e-5f);
                    }
                  }
                }));
      }
      for (var task : tasks) task.get(30, TimeUnit.SECONDS);
    }
    backend.sync();
    assertEquals(baseline, backend.diagnostics().liveDeviceBytes());
  }

  @Test
  void reverseScanTransfersNoValuesToHost() {
    var backend = new WgpuBackend();
    try (Tensor initial = array(new float[] {2}, 1);
        Tensor xs = array(new float[] {1, 2, 3}, 3, 1)) {
      var plan =
          Jit.compile(
              t ->
                  Transforms.grad(
                      values ->
                          ControlFlow.scan(
                                  (c, x) -> {
                                    Tensor next = c.mul(x);
                                    return new ControlFlow.ScanStep(next, next);
                                  },
                                  initial,
                                  values)
                              .carry()
                              .sum(),
                      t));
      try (Tensor warm = plan.apply(xs)) {
        warm.blockUntilReady();
      }
      var before = backend.diagnostics();
      try (Tensor result = plan.apply(xs)) {
        result.blockUntilReady();
        var delta = backend.diagnostics().minus(before);
        assertEquals(0, delta.hostToDeviceBytes());
        assertEquals(0, delta.deviceToHostBytes());
        assertEquals(0, delta.compilations());
        assertArrayEquals(new float[] {12, 6, 4}, result.toFloatArray(), 2e-5f);
      }
    }
  }
}
