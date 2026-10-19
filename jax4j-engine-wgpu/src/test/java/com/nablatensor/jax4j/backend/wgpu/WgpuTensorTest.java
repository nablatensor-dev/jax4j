package com.nablatensor.jax4j.backend.wgpu;

import static org.junit.jupiter.api.Assertions.*;

import com.nablatensor.jax4j.autodiff.AutoDiff;
import com.nablatensor.jax4j.tensor.*;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfSystemProperty;

@EnabledIfSystemProperty(named = "jax4j.wgpu.tests", matches = "true")
class WgpuTensorTest {
  private Tensor array(float[] x, int... dims) {
    return Tensors.arrayOn(x, Shape.of(dims), Device.wgpu());
  }

  @Test
  void fusedPlanHasOneDispatchAndNoTransfers() {
    var backend = new WgpuBackend();
    assertTrue(backend.isAvailable(), "Wgpu explicitly required");
    var plan =
        Jit.compile(
            x -> {
              Tensor square = x.mul(x);
              return square.add(square).add(3).tanh();
            });
    try (Tensor x = array(new float[] {1, 2, 3, 4}, 4);
        Tensor warm = plan.apply(x)) {
      warm.blockUntilReady();
      var before = backend.diagnostics();
      try (Tensor y = plan.apply(x)) {
        y.blockUntilReady();
        var delta = backend.diagnostics().minus(before);
        assertEquals(1, delta.dispatches());
        assertEquals(1, delta.allocations());
        assertEquals(0, delta.hostToDeviceBytes());
        assertEquals(0, delta.deviceToHostBytes());
        assertEquals(0, delta.compilations());
        assertArrayEquals(
            new float[] {(float) Math.tanh(5), (float) Math.tanh(11), (float) Math.tanh(21), 1},
            y.toFloatArray(),
            2e-5f);
      }
    }
  }

  @Test
  void shapeOperationsStayOnDevice() {
    var backend = new WgpuBackend();
    try (Tensor x = array(new float[] {1, 2, 3, 4}, 2, 2)) {
      var before = backend.diagnostics();
      try (Tensor flat = x.reshape(4);
          Tensor row = x.sliceAxis0(1);
          Tensor stacked = Tensor.stackAxis0(List.of(row, row))) {
        stacked.blockUntilReady();
        var delta = backend.diagnostics().minus(before);
        assertEquals(0, delta.hostToDeviceBytes());
        assertEquals(0, delta.deviceToHostBytes());
        assertArrayEquals(new float[] {3, 4, 3, 4}, stacked.toFloatArray(), 0f);
      }
    }
  }

  @Test
  void batchedMatmulAndGradient() {
    try (Tensor a = array(new float[] {1, 2, 3, 4, 5, 6, 7, 8}, 2, 2, 2);
        Tensor b = array(new float[] {1, 0, 0, 1}, 1, 2, 2);
        Tensor out = a.batchedMatmul(b)) {
      assertArrayEquals(a.toFloatArray(), out.toFloatArray(), 1e-6f);
    }
    try (Tensor x = array(new float[] {1, 2, 3}, 3);
        Tensor grad =
            Jit.apply(
                t -> AutoDiff.grad(v -> v.get(0).mul(v.get(0)).sum(), List.of(t)).get(0), x)) {
      assertArrayEquals(new float[] {2, 4, 6}, grad.toFloatArray(), 1e-5f);
    }
  }

  @Test
  void mappedGradientsAndAdamStayResident() {
    var backend = new WgpuBackend();
    var gradient =
        Jit.compile(
            Vmap.of(x -> AutoDiff.grad(v -> v.get(0).mul(v.get(0)).sum(), List.of(x)).get(0)));
    try (Tensor x = array(new float[] {1, 2, 3, 4, 5, 6}, 2, 3);
        Tensor warm = gradient.apply(x);
        com.nablatensor.jax4j.nn.Adam optimizer = new com.nablatensor.jax4j.nn.Adam(.01f)) {
      warm.blockUntilReady();
      var before = backend.diagnostics();
      try (Tensor grad = gradient.apply(x)) {
        var updated = optimizer.update(List.of(x), List.of(grad));
        try (Tensor next = updated.get(0)) {
          next.blockUntilReady();
          var delta = backend.diagnostics().minus(before);
          assertEquals(0, delta.hostToDeviceBytes());
          assertEquals(0, delta.deviceToHostBytes());
          assertArrayEquals(new float[] {2, 4, 6, 8, 10, 12}, grad.toFloatArray(), 1e-5f);
          assertArrayEquals(
              new float[] {.99f, 1.99f, 2.99f, 3.99f, 4.99f, 5.99f}, next.toFloatArray(), 2e-5f);
        }
      }
    }
  }

  @Test
  void randomKeysHaveMatchingStreamsWithoutHostUploads() {
    var backend = new WgpuBackend();
    long seed = 0x1020304050607080L, counter = 0x1fffffffdL;
    var cpu = new com.nablatensor.jax4j.backend.cpu.CpuBackend();
    float[] uniform = cpu.download(cpu.randomUniform(seed, counter, Shape.of(257), Device.cpu()));
    float[] normal = cpu.download(cpu.randomNormal(seed, counter, Shape.of(257), Device.cpu()));
    var before = backend.diagnostics();
    try (Tensor u = new PrngKey(seed, counter).uniform(Device.wgpu(), 257);
        Tensor n = new PrngKey(seed, counter).normal(Device.wgpu(), 257)) {
      n.blockUntilReady();
      assertEquals(0, backend.diagnostics().minus(before).hostToDeviceBytes());
      assertArrayEquals(uniform, u.toFloatArray(), 0f);
      assertArrayEquals(normal, n.toFloatArray(), 3e-5f);
    }
  }

  @Test
  void queuedBuffersAreFreedAfterCompletion() {
    var backend = new WgpuBackend();
    backend.sync();
    long baseline = backend.diagnostics().liveDeviceBytes();
    try (Tensor x = array(new float[] {1, 2, 3, 4}, 4)) {
      try (Tensor y = Jit.apply(t -> t.mul(t).add(1), x)) {
        // Closing wrappers before submission must defer their native frees.
      }
    }
    backend.sync();
    assertEquals(baseline, backend.diagnostics().liveDeviceBytes());
  }
}
