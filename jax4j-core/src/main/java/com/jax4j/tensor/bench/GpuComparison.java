/*
 * Copyright 2026 The NablaTensor Authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.jax4j.tensor.bench;

import com.jax4j.nn.FunctionalAdam;
import com.jax4j.tensor.*;
import com.jax4j.tensor.spi.*;
import com.jax4j.tensor.tree.TreeUtil;
import java.util.*;
import java.util.function.*;

/**
 * Same-process, alternating-order, synchronized tensor benchmark. Accelerators are discovered via
 * SPI.
 */
public final class GpuComparison {
  private GpuComparison() {}

  private record Training(List<Tensor> parameters, FunctionalAdam.State<List<Tensor>> optimizer) {}

  private record Counters(
      long dispatches,
      long allocations,
      long uploads,
      long downloads,
      long compilations,
      long parameters) {
    Counters minus(Counters x) {
      return new Counters(
          dispatches - x.dispatches,
          allocations - x.allocations,
          uploads - x.uploads,
          downloads - x.downloads,
          compilations - x.compilations,
          parameters - x.parameters);
    }
  }

  private static Counters counters(ComputeBackend backend) {
    try {
      Object snapshot = backend.getClass().getMethod("diagnostics").invoke(backend);
      return new Counters(
          value(snapshot, "dispatches"),
          value(snapshot, "allocations"),
          value(snapshot, "hostToDeviceBytes"),
          value(snapshot, "deviceToHostBytes"),
          value(snapshot, "compilations"),
          value(snapshot, "parameterUploadBytes"));
    } catch (ReflectiveOperationException e) {
      throw new IllegalStateException(e);
    }
  }

  private static long value(Object record, String method) throws ReflectiveOperationException {
    try {
      return ((Number) record.getClass().getMethod(method).invoke(record)).longValue();
    } catch (NoSuchMethodException missing) {
      if (method.equals("parameterUploadBytes")) return 0;
      throw missing;
    }
  }

  private static Tensor first(Object tree) {
    return (Tensor) TreeUtil.flatten(tree).leaves().getFirst();
  }

  private static Tensor array(Device device, float value, int... dims) {
    Shape shape = Shape.of(dims);
    float[] data = new float[Math.toIntExact(shape.size())];
    Arrays.fill(data, value);
    return Tensors.arrayOn(data, shape, device);
  }

  private static Tensor chain(Tensor x) {
    Tensor y = x;
    for (int i = 0; i < 12; i++) y = y.mul(1.0001).add(.0001).tanh();
    return y;
  }

  private static Tensor eager(Tensor x) {
    Tensor y = x;
    for (int i = 0; i < 12; i++) {
      Tensor a = y.mul(1.0001), b = a.add(.0001), next = b.tanh();
      a.close();
      b.close();
      if (y != x) y.close();
      y = next;
    }
    return y;
  }

  private static final class Scenario implements AutoCloseable {
    final ComputeBackend backend;
    final Object inputs;
    final Supplier<Object> action;
    final float expected, tolerance;

    Scenario(
        ComputeBackend backend,
        Object inputs,
        Supplier<Object> action,
        float expected,
        float tolerance) {
      this.backend = backend;
      this.inputs = inputs;
      this.action = action;
      this.expected = expected;
      this.tolerance = tolerance;
    }

    double invoke(boolean verify) {
      long start = System.nanoTime();
      Object result = action.get();
      try {
        first(result).blockUntilReady();
        double elapsed = (System.nanoTime() - start) / 1e6;
        if (verify)
          for (float value : first(result).toFloatArray())
            if (!Float.isFinite(value) || Math.abs(value - expected) > tolerance)
              throw new AssertionError(
                  "Benchmark numerical mismatch on "
                      + backend.name()
                      + ": "
                      + value
                      + " vs "
                      + expected);
        return elapsed;
      } finally {
        TreeUtil.closeAll(result);
      }
    }

    public void close() {
      TreeUtil.closeAll(inputs);
      backend.sync();
    }
  }

  private static Scenario scenario(
      String workload, Device device, int size, int matrix, int scanLength) {
    ComputeBackend backend = BackendRegistry.forDevice(device);
    if (workload.equals("matmul")) {
      Tensor a = array(device, .01f, matrix, matrix), b = array(device, .02f, matrix, matrix);
      var plan = Jit.compileTree((List<Tensor> xs) -> xs.get(0).matmul(xs.get(1)));
      return new Scenario(
          backend, List.of(a, b), () -> plan.apply(List.of(a, b)), matrix * .0002f, 2e-4f);
    }
    if (workload.equals("scan")) {
      Tensor initial = array(device, .25f, 32), xs = array(device, 1, scanLength, 32);
      var plan =
          Jit.compileTree(
              (List<Tensor> args) ->
                  ControlFlow.scan(
                      (c, x) -> {
                        Tensor next = c.mul(.99).add(x);
                        return new ControlFlow.ScanStep(next, next);
                      },
                      args.get(0),
                      args.get(1)));
      float expected = .25f;
      for (int i = 0; i < scanLength; i++) expected = expected * .99f + 1;
      return new Scenario(
          backend, List.of(initial, xs), () -> plan.apply(List.of(initial, xs)), expected, 2e-3f);
    }
    Tensor input = array(device, .25f, size);
    if (workload.equals("train-step")) {
      var optimizer = new FunctionalAdam(.01);
      var state = new Training(List.of(input), optimizer.initialize(List.of(input)));
      var plan =
          Jit.compileTree(
              (Training previous) -> {
                var gradients =
                    Transforms.gradTree(
                        (List<Tensor> ps) -> ps.getFirst().mul(ps.getFirst()).sum(),
                        previous.parameters());
                var update =
                    optimizer.update(previous.parameters(), gradients, previous.optimizer());
                TreeUtil.closeAll(gradients);
                return new Training(update.parameters(), update.state());
              });
      return new Scenario(backend, state, () -> plan.apply(state), .24f, 5e-6f);
    }
    if (workload.equals("gradient")) {
      var plan =
          Jit.compile(
              Transforms.grad(
                  x -> {
                    Tensor y = x.mul(1.5).tanh();
                    return y.mul(y).sum();
                  }));
      float y = (float) Math.tanh(.375);
      return new Scenario(backend, input, () -> plan.apply(input), 3 * y * (1 - y * y), 4e-5f);
    }
    float expected = .25f;
    for (int i = 0; i < 12; i++) expected = (float) Math.tanh((float) (expected * 1.0001) + .0001);
    if (workload.equals("eager-chain"))
      return new Scenario(backend, input, () -> eager(input), expected, 4e-5f);
    var plan = Jit.compile(GpuComparison::chain);
    return new Scenario(backend, input, () -> plan.apply(input), expected, 4e-5f);
  }

  private static Map<String, Object> stats(
      double cold, List<Double> samples, Counters delta, int calls) {
    List<Double> sorted = new ArrayList<>(samples);
    Collections.sort(sorted);
    Map<String, Object> out = new LinkedHashMap<>();
    out.put("coldFunctionMs", cold);
    out.put("medianMs", sorted.get(sorted.size() / 2));
    out.put("minMs", sorted.getFirst());
    out.put(
        "p90Ms", sorted.get(Math.min(sorted.size() - 1, (int) Math.ceil(.9 * sorted.size()) - 1)));
    out.put("dispatchesPerCall", (double) delta.dispatches / calls);
    out.put("allocationsPerCall", (double) delta.allocations / calls);
    out.put("hostToDeviceBytes", delta.uploads);
    out.put("deviceToHostBytes", delta.downloads);
    out.put("newCompilations", delta.compilations);
    out.put("parameterUploadBytes", delta.parameters);
    return out;
  }

  private static String json(Object value) {
    if (value instanceof String s)
      return "\"" + s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n") + "\"";
    if (value instanceof Number || value instanceof Boolean) return value.toString();
    if (value instanceof Map<?, ?> map) {
      List<String> pairs = new ArrayList<>();
      map.forEach((k, v) -> pairs.add(json(k.toString()) + ":" + json(v)));
      return "{" + String.join(",", pairs) + "}";
    }
    if (value instanceof List<?> list)
      return "[" + String.join(",", list.stream().map(GpuComparison::json).toList()) + "]";
    throw new IllegalArgumentException("Unsupported JSON value " + value);
  }

  public static void main(String[] args) {
    int size = Integer.getInteger("size", 262144),
        matrix = Integer.getInteger("matrix", 256),
        length = Integer.getInteger("scanLength", 128),
        warmup = Integer.getInteger("warmup", 10),
        samples = Integer.getInteger("samples", 15),
        iterations = Integer.getInteger("iterations", 5);
    if (size < 1 || matrix < 1 || length < 1 || warmup < 1 || samples < 3 || iterations < 1)
      throw new IllegalArgumentException("Invalid benchmark dimensions or repetitions");
    var vk = BackendRegistry.forDevice(Device.vulkan());
    var wg = BackendRegistry.forDevice(Device.wgpu());
    Map<String, Object> result = new LinkedHashMap<>();
    result.put("vulkanDevice", ((GpuTensorBackend) vk).deviceName());
    result.put("wgpuDevice", ((GpuTensorBackend) wg).deviceName());
    result.put("dtype", "F32");
    result.put("elements", size);
    result.put("matrix", matrix);
    result.put("scanLength", length);
    result.put("scanWidth", 32);
    result.put("warmupCalls", warmup);
    result.put("samples", samples);
    result.put("iterationsPerSample", iterations);
    result.put(
        "method",
        "Alternating backend order, synchronized per call; input uploads and result downloads"
            + " outside warm timing; same mathematical outputs verified before timing. Final-output"
            + " cleanup is outside timing for both backends; intermediate releases are included in"
            + " execution.");
    List<Object> workloads = new ArrayList<>();
    for (String name :
        List.of("fused-chain", "eager-chain", "matmul", "gradient", "train-step", "scan")) {
      try (Scenario vulkan = scenario(name, Device.vulkan(), size, matrix, length);
          Scenario wgpu = scenario(name, Device.wgpu(), size, matrix, length)) {
        double coldVk = vulkan.invoke(true), coldWg = wgpu.invoke(true);
        for (int i = 0; i < warmup; i++) {
          vulkan.invoke(false);
          wgpu.invoke(false);
        }
        Counters beforeVk = counters(vk), beforeWg = counters(wg);
        List<Double> vkSamples = new ArrayList<>(), wgSamples = new ArrayList<>();
        for (int sample = 0; sample < samples; sample++) {
          for (int backend = 0; backend < 2; backend++) {
            boolean useVk = (sample + backend) % 2 == 0;
            Scenario scenario = useVk ? vulkan : wgpu;
            double total = 0;
            for (int i = 0; i < iterations; i++) total += scenario.invoke(false);
            (useVk ? vkSamples : wgSamples).add(total / iterations);
          }
        }
        var vkStats = stats(coldVk, vkSamples, counters(vk).minus(beforeVk), samples * iterations);
        var wgStats = stats(coldWg, wgSamples, counters(wg).minus(beforeWg), samples * iterations);
        Map<String, Object> entry = new LinkedHashMap<>();
        entry.put("workload", name);
        entry.put("vulkan", vkStats);
        entry.put("wgpu", wgStats);
        entry.put(
            "wgpuOverVulkan", (double) wgStats.get("medianMs") / (double) vkStats.get("medianMs"));
        workloads.add(entry);
      }
    }
    result.put("workloads", workloads);
    System.out.println(json(result));
  }
}
